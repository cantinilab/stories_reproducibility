poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=zesta model.quadratic=true model.quadratic_weight=1e-5,1e-4,1e-3,1e-2,1e-1 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_quadratic" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_quadratic.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=zesta model.quadratic=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=zesta model.quadratic=false model.teacher_forcing=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_notf" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_notf.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=zesta model.quadratic=false model.n_steps=10 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_tensteps" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_tensteps.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=zesta model.quadratic=false step.type="icnn_implicit" optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_implicit" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/zesta_may_linear_implicit.log 2>&1 &

poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=artista_growth model.quadratic=true model.quadratic_weight=1e-5,1e-4,1e-3,1e-2,1e-1 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_growth_may_quadratic" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_growth_may_quadratic.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=artista_growth model.quadratic=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_growth_may_linear" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_growth_may_linear.log 2>&1 &

poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=artista model.quadratic=true model.quadratic_weight=1e-5,1e-4,1e-3,1e-2,1e-1 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_may_quadratic" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_may_quadratic.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=artista model.quadratic=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_may_linear" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/artista_may_linear.log 2>&1 &

poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=midbrain model.quadratic=true model.quadratic_weight=1e-5,1e-4,1e-3,1e-2,1e-1 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/midbrain_may_quadratic" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/midbrain_may_quadratic.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu +organism=midbrain model.quadratic=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/midbrain_may_linear" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/midbrain_may_linear.log 2>&1 &

poetry run python3 scripts/train.py --multirun hydra/launcher=gpu_largemem +organism=mosta model.quadratic=true model.quadratic_weight=1e-5,1e-4,1e-3,1e-2,1e-1 optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/mosta_may_quadratic" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/mosta_may_quadratic.log 2>&1 &
poetry run python3 scripts/train.py --multirun hydra/launcher=gpu_largemem +organism=mosta model.quadratic=false optimizer.learning_rate=1e-2 model.epsilon=0.1 model.seed=17158,20181,12409,5360,21712,21781,24802,13630,9668,651 checkpoint_path="/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/mosta_may_linear" > /pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/tmp/mosta_may_linear.log 2>&1 &
