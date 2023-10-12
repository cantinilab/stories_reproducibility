import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import os

    import anndata as ad
    import jax
    import numpy as np
    import optax
    import orbax.checkpoint
    import wandb
    from flatten_dict import flatten

    import spacetime
    from spacetime import potentials, steps

    # Setup Weights & Biases.
    config = flatten(OmegaConf.to_container(cfg, resolve=True), reducer="dot")
    wandb.init(project="train_spacetime", config=config, mode=cfg.wandb.mode)
    print(config, f"JAX device type: {jax.devices()[0].device_kind}")

    # Save config as yaml file, creating subfolders if needed
    config_file_name = f"{cfg.checkpoint_path}_{cfg.model.seed}/config.yaml"
    os.makedirs(os.path.dirname(config_file_name), exist_ok=True)
    with open(config_file_name, "w") as f:
        OmegaConf.save(cfg, f)

    # Load the data.
    adata = ad.read_h5ad(cfg.organism.dataset_path)

    # Normalize the obsm.
    adata.obsm[cfg.organism.obsm] /= adata.obsm[cfg.organism.obsm].max()

    # Center the space.
    timepoints = np.sort(np.unique(adata.obs[cfg.organism.time_obs]))
    for timepoint in timepoints:
        idx = adata.obs[cfg.organism.time_obs] == timepoint
        mean_space = adata.obsm[cfg.organism.space_obsm][idx].mean(axis=0)
        adata.obsm[cfg.organism.space_obsm][idx] -= mean_space.reshape(1, 2)

    # Remove the last timepoint if needed.
    if cfg.skip_last:
        idx = adata.obs[cfg.organism.time_obs] != timepoints[-1]
        adata = adata[idx]

    # Intialize keyword arguments for the proximal step.
    step_kwargs = {}

    # If the proximal step is quadratic, add the appropriate keyword arguments.
    if "quadratic" in cfg.step.type:
        step_kwargs["fused"] = cfg.step.fused
        step_kwargs["cross"] = cfg.step.cross
        step_kwargs["straight"] = cfg.step.straight

    # If the proximal step is implicit, add the appropriate keyword arguments.
    if "implicit" in cfg.step.type:
        step_kwargs["implicit_diff"] = cfg.step.implicit_diff
        step_kwargs["maxiter"] = cfg.step.maxiter
        step_kwargs["log_callback"] = lambda x: wandb.log(x)

    # Choose the proximal step.
    if cfg.step.type == "linear_explicit":
        step = steps.LinearExplicitStep()
    elif cfg.step.type == "quadratic_explicit":
        step = steps.QuadraticExplicitStep(**step_kwargs)
    elif cfg.step.type == "monge_linear_implicit":
        step = steps.MongeLinearImplicitStep(**step_kwargs)
    elif cfg.step.type == "monge_quadratic_implicit":
        step = steps.MongeQuadraticImplicitStep(**step_kwargs)
    else:
        raise ValueError(f"Step {cfg.step.type} not recognized.")

    # Initialize the model.
    my_model = spacetime.SpaceTime(
        potential=potentials.MLPPotential(cfg.potential.features),
        proximal_step=step,
        tau=cfg.model.tau,
        quadratic=cfg.model.quadratic,
        epsilon=cfg.model.epsilon,
        teacher_forcing=cfg.model.teacher_forcing,
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

    # Fit the model.
    my_model.fit(
        adata=adata,
        time_obs=cfg.organism.time_obs,
        x_obsm=cfg.organism.obsm,
        space_obsm=cfg.organism.space_obsm,
        optimizer=optax.adamw(cfg.optimizer.learning_rate),
        max_epochs=cfg.optimizer.max_epochs,
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
