poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_joker wandb.mode="offline"     rank=-1     checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep zesta | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_joker wandb.mode="offline"     rank=-1     checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep artista | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate.py --multirun hydra/launcher=gpu_big wandb.mode="offline"       rank=500     checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep mosta | tr '\n' ',' | sed 's/,$/\n/')

poetry run python3 scripts/evaluate_multi.py --multirun hydra/launcher=gpu_joker wandb.mode="offline"     checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep zesta | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate_multi.py --multirun hydra/launcher=gpu_joker wandb.mode="offline"     checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep artista | tr '\n' ',' | sed 's/,$/\n/')
poetry run python3 scripts/evaluate_multi.py --multirun hydra/launcher=gpu_largemem wandb.mode="offline"       checkpoint_path=$(ls -d /pasteur/appa/homes/ghuizing/space-time/tmp/* | grep mosta | tr '\n' ',' | sed 's/,$/\n/')
