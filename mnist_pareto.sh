# iterate over valeus of noise multiplier
# and launch the script mnist_pareto.py
for tan_noise_multiplier in 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2
do
  python mnist_dpsgd.py --cfg.dpsgd=True --cfg.tan_noise_multiplier=$tan_noise_multiplier --cfg.log_wandb=sweep_tan_$tan_noise_multiplier
done