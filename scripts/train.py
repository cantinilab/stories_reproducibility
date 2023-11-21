import hydra
from omegaconf import DictConfig, OmegaConf


# This script loads the data, initializes the model, and fits the model to the data.
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import os

    import anndata as ad
    import jax
    import numpy as np
    import optax
    import orbax.checkpoint
    import wandb
    from flax.linen.activation import gelu
    from flatten_dict import flatten

    import spacetime
    from spacetime import potentials, steps

    # Setup Weights & Biases.
    config = flatten(OmegaConf.to_container(cfg, resolve=True), reducer="dot")
    wandb.init(project="train_spacetime", config=config, mode=cfg.wandb.mode)

    # Print the JAX device type.
    print(config, f"JAX device type: {jax.devices()[0].device_kind}")

    # Save config as yaml file, creating subfolders if needed. This is useful for
    # the evaluation script, which will retrieve information from this file.
    config_file_name = f"{cfg.checkpoint_path}_{cfg.model.seed}/config.yaml"
    os.makedirs(os.path.dirname(config_file_name), exist_ok=True)
    with open(config_file_name, "w") as f:
        OmegaConf.save(cfg, f)

    # Load the data.
    adata = ad.read_h5ad(cfg.organism.dataset_path)

    # Select a given number of principal components then normalize the embedding.
    adata.obsm[cfg.organism.obsm] = adata.obsm[cfg.organism.obsm][:, : cfg.n_pcs]
    adata.obsm[cfg.organism.obsm] /= adata.obsm[cfg.organism.obsm].max()

    # Remove the last timepoint if needed.
    timepoints = np.sort(np.unique(adata.obs[cfg.organism.time_obs]))
    if cfg.skip_last:
        adata = adata[adata.obs[cfg.organism.time_obs] != timepoints[-1]]

    # Make the space random if needed.
    if cfg.random_space:
        adata.obsm[cfg.organism.space_obsm] = np.random.randn(
            adata.obsm[cfg.organism.space_obsm].shape[0],
            adata.obsm[cfg.organism.space_obsm].shape[1],
        )

    # Intialize keyword arguments for the proximal step.
    step_kwargs = {}

    # If the proximal step is implicit, add the appropriate keyword arguments.
    if "implicit" in cfg.step.type:
        step_kwargs["implicit_diff"] = cfg.step.implicit_diff
        step_kwargs["maxiter"] = cfg.step.maxiter
        step_kwargs["log_callback"] = lambda x: wandb.log(x)

    # Choose the proximal step.
    if cfg.step.type == "linear_explicit":
        step = steps.LinearExplicitStep()
    elif cfg.step.type == "monge_linear_implicit":
        step = steps.MongeLinearImplicitStep(**step_kwargs)
    elif cfg.step.type == "icnn_linear_implicit":
        step = steps.ICNNLinearImplicitStep(**step_kwargs)
    else:
        raise ValueError(f"Step {cfg.step.type} not recognized.")

    # Initialize the model.
    my_model = spacetime.SpaceTime(
        potential=potentials.MLPPotential(cfg.potential.features, activation=gelu),
        proximal_step=step,
        tau=cfg.model.tau,
        tau_auto=cfg.model.tau_auto,
        teacher_forcing=cfg.model.teacher_forcing,
        quadratic=cfg.model.quadratic,
        epsilon=cfg.model.epsilon,
        log_callback=lambda x: wandb.log(x),
        fused_penalty=cfg.model.fused,
    )

    # Define the checkpoint manager.
    options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=cfg.optimizer.checkpoint_interval,
        max_to_keep=1,
        best_fn=lambda x: x["loss"],
        best_mode="min",
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        f"{cfg.checkpoint_path}_{cfg.model.seed}/checkpoints",
        orbax.checkpoint.PyTreeCheckpointer(),
        options=options,
    )

    scheduler = optax.constant_schedule(cfg.optimizer.learning_rate)
    scheduler = optax.cosine_decay_schedule(cfg.optimizer.learning_rate, 10_000)

    # Fit the model.
    my_model.fit(
        adata=adata,
        time_obs=cfg.organism.time_obs,
        x_obsm=cfg.organism.obsm,
        space_obsm=cfg.organism.space_obsm,
        optimizer=optax.chain(
            optax.adamw(scheduler),
            optax.clip_by_global_norm(10.0),
        ),
        max_iter=cfg.optimizer.max_iter,
        batch_size=cfg.optimizer.batch_size,
        train_val_split=cfg.optimizer.train_val_split,
        min_delta=cfg.optimizer.min_delta,
        patience=cfg.optimizer.patience,
        checkpoint_manager=checkpoint_manager,
        key=jax.random.PRNGKey(cfg.model.seed),
    )

    wandb.finish()


if __name__ == "__main__":
    main()
