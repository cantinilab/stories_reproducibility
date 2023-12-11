poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_joker wandb.mode="offline" checkpoint_path=$(ls -d tmp/* | grep zesta | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_big wandb.mode="offline" checkpoint_path=$(ls -d tmp/* | grep mosta | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_joker wandb.mode="offline" checkpoint_path=$(ls -d tmp/* | grep artista | tr '\n' ',' | sed 's/,$/\n/')
