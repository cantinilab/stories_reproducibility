# Learning cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein

This is the code used to produce the experiments for STORIES.

- `spacetime/`: the source code for the package at the time of writing
- `scripts/`: contains the code to train and evaluate the model
  - `evaluate.sh`: commands used to train the model on a SLURM cluster using Hydra and Weights&Biases
  - `train.sh`: commands used to evaluate the model on a SLURM cluster using Hydra and Weights&Biases
  - `configs/`: configuration files for datasets, scripts, and GPUs
- `notebooks/`: Jupyter notebooks for preprocessing and plotting
  - `preprocess_*.ipynb`: Preprocessing scripts
  - `vis_runs.ipynb`: Scripts to generate plots in Fig 2.B
  - `*_matching.ipynb`: Scripts to generate plots in Fig 2.C
  - `blender_render.py`: Script to generate the 3D plots, to be used in Blender
  - `fig3.ipynb`: Script to generate the plots in Fig. 3
  - `fig4.ipynb`: Script to generate the plots in Fig. 4
