# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.


"""MNIST example.
Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

from functools import partial
import math
from typing import Any, Callable, Sequence, Tuple
import warnings

from absl import logging
from absl import app
from absl import flags

import dp_accounting

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp

import matplotlib

from ml_collections import config_dict
from ml_collections import config_flags

import numpy as onp
import optax

import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm
import wandb

from cnqr.layers import StiefelDense, Normalized2ToInftyDense
from cnqr.layers import RKOConv, l2norm_pool, global_l2norm_pool
from cnqr.layers import fullsort, groupsort2
from cnqr.losses import multiclass_hinge


cfg = config_dict.ConfigDict()
cfg.augment = True
cfg.learning_rate = 1e-3
cfg.num_gradient_steps = 10 * 1000
cfg.num_epochs = -1
cfg.batch_size = 128
cfg.batch_size_test = 1024
cfg.net_scale = 64
cfg.use_global_pool = False
cfg.use_bias = False
cfg.deep = True
cfg.loss_fn = "multiclass_hinge"
cfg.temperature = 16.
cfg.margin = 5e-3
cfg.dpsgd = True
cfg.noise_multiplier = -1.
cfg.tan_noise_multiplier = 0.6
cfg.l2_norm_clip = 4.0
cfg.delta = 1e-5
cfg.log_wandb = 'disabled'
cfg.sweep_count = 40

NUM_EXAMPLES = 50 * 1000
MAX_EPSILON = 20.

_CONFIG = config_flags.DEFINE_config_dict('cfg', cfg)

project_name = "cifar10_dpsgd"
sweep_config = {
  'method': 'bayes',
  'name': 'default',
  'metric': {'goal': 'maximize', 'name': 'test_accuracy'},
  'early_terminate': {'type': 'hyperband', 'min_iter': 10, 'eta': 2},
  'parameters': {
      'learning_rate': {
        'max': 5e-1,
        'min': 1e-5,
        'distribution': 'log_uniform_values'},
      'l2_norm_clip': {
        'max': 3e2,
        'min': 1e-1,
        'distribution': 'log_uniform_values'},
      'margin': {
        'max': 1.,
        'min': 1e-3,
        'distribution': 'log_uniform_values'},
      'use_global_pool': {
        'values': [True, False],
        'distribution': 'categorical'},
      'use_bias': {
        'values': [True, False],
        'distribution': 'categorical'},
      'deep': {
        'values': [True, False],
        'distribution': 'categorical'},
      'net_scale': {
        'values': [16, 32, 64, 128, 256],
        'distribution': 'categorical'},
      'batch_size': {
        'values': [64, 128, 256, 512, 1024],
        'distribution': 'categorical'},
  }
}


def compute_epsilon(steps, target_delta=1e-5):
  if NUM_EXAMPLES * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = cfg.batch_size / float(NUM_EXAMPLES)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(dp_accounting.PoissonSampledDpEvent(
      q, dp_accounting.GaussianDpEvent(cfg.noise_multiplier)), steps)
  return accountant.get_epsilon(target_delta)


def loss_on_batch(logits, labels):
  one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
  if cfg.loss_fn == "multiclass_hinge":
    return multiclass_hinge(logits, one_hot_labels, margin=cfg.margin)
  elif cfg.loss_fn == "softmax_crossentropy":
    return optax.softmax_cross_entropy(logits * cfg.temperature, one_hot_labels)
  else:
    raise ValueError("Unknown loss function.")


class LipschitzCNN(nn.Module):
  """A simple Lipschitz CNN model."""
  num_classes: int
  scale = cfg.net_scale
  hidden_widths: Sequence[int] = (1, 1, 1, 2)

  @nn.compact
  def __call__(self, x, train):
    rko_conv = partial(RKOConv, kernel_size=(3, 3), use_bias=cfg.use_bias)

    widths = onp.array(self.hidden_widths) * self.scale

    for width in widths[:-1]:
      x = rko_conv(features=width)(x, train=train)
      x = groupsort2(x)
      if cfg.deep:
        x = rko_conv(features=width)(x, train=train)
        x = groupsort2(x)
      x = l2norm_pool(x, window_shape=(2, 2), strides=(2, 2))

    if cfg.use_global_pool:
      x = rko_conv(features=widths[-1])(x, train=train)
      x = groupsort2(x)
      x = global_l2norm_pool(x)
    else:
      x = jnp.reshape(x, (x.shape[0], -1))
      x = StiefelDense(features=widths[-1], use_bias=cfg.use_bias)(x, train=train)
      x = fullsort(x)

    x = Normalized2ToInftyDense(features=self.num_classes, use_bias=cfg.use_bias)(x, train=train)

    return x


class LipschitzTrainState(train_state.TrainState):
  """Train state with Lipschitz constraint."""
  lip_state: Any

def create_train_state(rng, num_classes):
  """Creates initial `TrainState`."""
  model = LipschitzCNN(num_classes=num_classes)
  keys = dict(zip(['params', 'lip'], jax.random.split(rng, 2)))
  dummy_batch = jnp.zeros([cfg.batch_size, 32, 32, 3])
  model_params = model.init(keys, dummy_batch, train=True)
  params, lip_state = model_params['params'], model_params['lip']

  if cfg.dpsgd:
    tx = optax.dpsgd(learning_rate=cfg.learning_rate,
                     l2_norm_clip=cfg.l2_norm_clip,
                     noise_multiplier=cfg.noise_multiplier,
                     seed=12345,
                     momentum=0.9,
                     nesterov=True)
  else:
    tx = optax.adam(learning_rate=cfg.learning_rate)

  return LipschitzTrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    lip_state=lip_state)


@jax.jit
def update_model(state, grads, lip_vars):
  """Updates model parameters."""
  return state.apply_gradients(grads=grads, lip_state=lip_vars)

@jax.jit
def predict_model(train_state, points):
  """Predicts on a batch of points."""
  model_params = {'params': train_state.params, 'lip': train_state.lip_state}
  preds = train_state.apply_fn(model_params, points, train=False)
  return preds

def tf_to_jax(arr):
  return jnp.array(arr)

def predict_ds(train_state, ds):
  """Predicts on a dataset."""
  jnp_preds = [predict_model(train_state, tf_to_jax(batch)) for batch, _ in ds]
  jnp_preds = jnp.concatenate(jnp_preds)
  return jnp_preds


def evaluate_model(train_state, ds):
  """Evaluates model on a dataset."""
  jnp_preds = predict_ds(train_state, ds)
  jnp_labels = jnp.concatenate([tf_to_jax(label) for _, label in ds], axis=0)
  accuracy = jnp.mean(jnp.argmax(jnp_preds, -1) == jnp_labels)
  loss = jnp.mean(loss_on_batch(logits=jnp_preds, labels=jnp_labels))
  return loss, accuracy


@jax.jit
def apply_model(state, batch):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params, example):
    image, label = example['image'], example['label']
    all_params = {'params': params, 'lip': state.lip_state}
    logits, variables = state.apply_fn(all_params, image, train=True, mutable='lip')
    loss = jnp.mean(loss_on_batch(logits=logits, labels=label))
    return loss, (variables['lip'], logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  if cfg.dpsgd:
    # Insert dummy dimension in axis 1 to use jax.vmap over the batch
    batch = jax.tree_util.tree_map(lambda x: x[:, None], batch)
    # Use jax.vmap across the batch to extract per-example gradients
    grad_fn = jax.vmap(grad_fn, in_axes=(None, 0), out_axes=0)

  aux, grads = grad_fn(state.params, batch)
  (losses, (lip_vars, logits)) = aux

  loss = jnp.mean(losses)  # Average loss across batch.

  if cfg.dpsgd:
    # Remove dummy dimension on logits and loss, and Lipschitz variables
    logits = jnp.squeeze(logits, axis=1)
    batch = jax.tree_util.tree_map(lambda x: x.squeeze(axis=1), batch)
    lip_vars = jax.tree_util.tree_map(lambda x: x[0,...], lip_vars)

  accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
  return grads, (lip_vars, accuracy), loss

@jax.jit
def update_model(state, grads, lip_vars):
  return state.apply_gradients(grads=grads, lip_state=lip_vars)


def train_epoch(state, train_ds,):
  """Train for a single epoch."""
  epoch_loss = []
  epoch_accuracy = []

  for batch_images, batch_labels in (pb := tqdm(train_ds)):
    batch_images = tf_to_jax(batch_images)
    batch_labels = tf_to_jax(batch_labels)
    batch = {'image': batch_images, 'label': batch_labels}
    grads, aux, loss = apply_model(state, batch)
    lip_vars, accuracy = aux
    state = update_model(state, grads, lip_vars=lip_vars)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
    pb.set_postfix({'loss': float(loss), 'accuracy': float(accuracy*100)})
  
  train_loss = jnp.array(epoch_loss).mean()
  train_accuracy = jnp.array(epoch_accuracy).mean()
  return state, train_loss, train_accuracy


def augment_cifar10(image, label):
  """Augment CIFAR-10 images."""
  image = tf.image.random_flip_left_right(image)
  return image, label


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  data_dir = '/data/datasets/'
  train_ds = tfds.load('cifar10', split='train', shuffle_files=True, data_dir=data_dir)
  test_ds = tfds.load('cifar10', split='test', shuffle_files=False, data_dir=data_dir)

  def normalize_img(img_label):
    """Normalizes images: `uint8` -> `float32`."""
    image, label = img_label['image'], img_label['label']
    image = tf.cast(image, dtype=tf.dtypes.float32) / 255.
    # image = image * 2 - 1  # image in [-1, 1]
    return image, label
  
  train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.shuffle(1024).batch(cfg.batch_size)
  test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.batch(cfg.batch_size_test)

  if cfg.augment:
    train_ds = train_ds.map(augment_cifar10, num_parallel_calls=tf.data.AUTOTUNE)

  return train_ds, test_ds


def get_noise_multiplier():
  """ See [1]] for details on how to scale the noise multiplier.
  This heuristic ensures an approximately constant DP guarantee
  when rescaling the batch size (for the same number of gradient steps).
  
  [1] Sander, T., Stock, P. and Sablayrolles, A., 2022.
    TAN without a burn: Scaling Laws of DP-SGD.
    arXiv preprint arXiv:2210.03403.
  """
  if not cfg.dpsgd:
    return 0.
  REF_BATCH_SIZE = 256
  batch_size_ratio = cfg.batch_size / REF_BATCH_SIZE
  noise_multiplier = cfg.tan_noise_multiplier * batch_size_ratio
  return noise_multiplier


def init_wandb():
  if cfg.log_wandb == 'run':
    # Log all hyper-parameters because config=cfg.
    wandb.init(project=project_name, mode="online", config=cfg)
  elif cfg.log_wandb == 'disabled':
    wandb.init(project=project_name, mode="disabled", config=cfg)
  else:  # this is a sweep.
    wandb.init()  # warning: do not log all hyper-parameters!
    # instead wandb.config contains the assignments of hyper_parameters.
    # made by the sweep agent. We retrieve them, put them in the config dict,
    # and log them manually.
    for param in sweep_config['parameters']:
      cfg[param] = wandb.config[param]

  # Set number of epochs.
  if cfg.num_gradient_steps != -1 and cfg.num_epochs != -1:
    raise ValueError('Only one of num_gradient_steps and num_epochs can be set.')

  if cfg.num_epochs == -1:
    cfg.num_epochs = math.ceil((cfg.batch_size * cfg.num_gradient_steps) / NUM_EXAMPLES)
  else:
    cfg.num_gradient_steps = cfg.num_epochs * NUM_EXAMPLES // cfg.batch_size

  # Set noise multiplier.
  if cfg.noise_multiplier != -1 and cfg.tan_noise_multiplier != -1:
    raise ValueError('Only one of noise_multiplier and tan_noise_multiplier can be set.')

  if cfg.noise_multiplier == -1:
    cfg.noise_multiplier = get_noise_multiplier()

  # log all hyper-parameters in every case!
  df = pd.DataFrame.from_dict(data={k: [v] for k, v in cfg.items()}, orient='columns')
  print(df)
  hyper_params_table = wandb.Table(data=df)
  wandb.log({"hyper_params": hyper_params_table})


def compute_TAN(epoch):
  """Compute Total Amount of Noise of the model.
  
  Inspire from paper:
    Sander, T., Stock, P. and Sablayrolles, A., 2022.
    TAN without a burn: Scaling Laws of DP-SGD.
    arXiv preprint arXiv:2210.03403.
  """
  q = cfg.batch_size / NUM_EXAMPLES
  sigma = cfg.noise_multiplier
  num_steps_per_epoch = NUM_EXAMPLES // cfg.batch_size
  S = epoch * num_steps_per_epoch
  eta = (q / sigma) * (S / 2)**0.5
  tan = 1 / eta
  return tan


def log_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy):
  """Log metrics to console and wandb."""

  # Determine privacy loss so far
  if cfg.dpsgd:
    steps = epoch * NUM_EXAMPLES // cfg.batch_size
    eps = compute_epsilon(steps, cfg.delta)
    dp_sgd_msg = f'DP: (delta={cfg.delta:.0e}, epsilon={eps:.2f}) | '
    tan = compute_TAN(epoch)
  else:
    eps = 0
    dp_sgd_msg = 'DP: (delta=∞, epsilon=∞) | '
    tan = 0
  
  tan_msg = f'TAN: {tan:.3f}'

  logging.info(
        f'epoch:{epoch:3d} | '
        f'train_loss: {train_loss:.5f} '
        f'train_accuracy: {train_accuracy*100:.2f}% | '
        f'test_loss: {test_loss:.5f} '
        f'test_accuracy: {test_accuracy*100:.2f}% | '
        + dp_sgd_msg
        + tan_msg)

  if cfg.log_wandb != 'disabled':
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "epsilon": eps,
        "delta": cfg.delta,
        "TAN": tan
    })

  return eps


def train() :
  """Execute model training and evaluation loop."""
  init_wandb()

  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, num_classes=10)

  for epoch in range(1, cfg.num_epochs + 1):
    state, train_loss, train_accuracy = train_epoch(state, train_ds)
    test_loss, test_accuracy = evaluate_model(state, test_ds)

    epsilon = log_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy)

    if epsilon >= MAX_EPSILON:
      logging.info(f'Epsilon exceeded {MAX_EPSILON}. Stopping training because guarantees are not meaningful.')
      break


def main(_):
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Handle weird bug in dependencies.
  matplotlib.use('Agg')

  wandb.login()
  
  if cfg.log_wandb in ['run', 'disabled']:
    train()
    return

  if cfg.log_wandb.startswith('sweep_'):
    sweep_name = cfg.log_wandb[len('sweep_'):]
    sweep_config['name'] = sweep_name
    for key, value in cfg.items():
      if key not in sweep_config['parameters']:
        sweep_config['parameters'][key] = dict(value=value, distribution='constant')
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)      

  if cfg.log_wandb.startswith('resume_'):
    sweep_id = cfg.log_wandb[len('resume_'):]
  
  wandb.agent(sweep_id, function=train, count=cfg.sweep_count)


if __name__ == '__main__':
  app.run(main)
