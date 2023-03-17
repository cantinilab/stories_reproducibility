# Imports
import hydra
import matplotlib.pyplot as plt
import wandb
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs/main", config_name="config")
def main(cfg: DictConfig) -> None:

    import anndata as ad
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax
    import scanpy as sc
    from scipy import interpolate
    from space_time import explicit_steps, implicit_steps, model, potentials

    # start a new wandb run to track this script
    print(dict(cfg))
    print(f"JAX device type: {jax.devices()[0].device_kind}")

    wandb.init(
        project="spacetime-main",
        config=flatten(OmegaConf.to_container(cfg), reducer="dot"),
        mode=cfg.wandb.mode,
    )

    # Load the data.
    print("Loading data")
    adata = ad.read_h5ad(cfg.organism.dataset_path)

    sc.pp.subsample(adata, fraction=cfg.organism.subset)

    # Define the protential.
    print("Defining potential")
    potential = potentials.MLPPotential(
        features=cfg.potential.features,
        act_fn=nn.softplus,
    )

    # Define the proximal step.
    print("Defining proximal step")
    if cfg.proximal_step.type == "linear_explicit":
        proximal_step = explicit_steps.linear.LinearExplicitStep()
    elif cfg.proximal_step.type == "quadratic_explicit":
        proximal_step = explicit_steps.quadratic.QuadraticExplicitStep(
            fused=cfg.proximal_step.fused,
            cross=cfg.proximal_step.cross,
            straight=cfg.proximal_step.straight,
        )
    elif cfg.proximal_step.type == "linear_implicit":
        proximal_step = implicit_steps.linear.LinearImplicitStep(
            epsilon=cfg.proximal_step.epsilon,
            maxiter=cfg.proximal_step.maxiter,
            sinkhorn_iter=cfg.proximal_step.sinkhorn_iter,
            implicit_diff=cfg.proximal_step.implicit_diff,
            wb=True,
        )
    elif cfg.proximal_step.type == "quadratic_implicit":
        proximal_step = implicit_steps.quadratic.QuadraticImplicitStep(
            sinkhorn_iter=cfg.proximal_step.sinkhorn_iter,
            epsilon=cfg.proximal_step.epsilon,
            fused=cfg.proximal_step.fused,
            maxiter=cfg.proximal_step.maxiter,
            implicit_diff=cfg.proximal_step.implicit_diff,
            wb=True,
        )
    elif cfg.proximal_step.type == "monge_linear_implicit":
        proximal_step = implicit_steps.monge_linear.MongeLinearImplicitStep(
            maxiter=cfg.proximal_step.maxiter,
            implicit_diff=cfg.proximal_step.implicit_diff,
            wb=True,
        )
    elif cfg.proximal_step.type == "monge_quadratic_implicit":
        proximal_step = implicit_steps.monge_quadratic.MongeQuadraticImplicitStep(
            fused=cfg.proximal_step.fused,
            cross=cfg.proximal_step.cross,
            straight=cfg.proximal_step.straight,
            implicit_diff=cfg.proximal_step.implicit_diff,
            wb=True,
            maxiter=cfg.proximal_step.maxiter,
        )
    else:
        raise ValueError(f"Step {cfg.proximal_step.type} not recognized.")

    # Define the model.
    print("Defining model")
    my_model = model.SpaceTime(
        potential=potential,
        proximal_step=proximal_step,
        tau=cfg.proximal_step.tau,
        quadratic=cfg.model.quadratic,
        epsilon=cfg.model.epsilon,
        teacher_forcing=cfg.model.teacher_forcing,
        wb=True,
    )

    # Fit the model.
    my_model.fit_adata(
        adata=adata,
        time_obs=cfg.organism.time_obs,
        obsm=cfg.organism.obsm,
        space_obsm=cfg.organism.space_obsm,
        optimizer=optax.adam(cfg.optimizer.learning_rate),
        n_iter=cfg.optimizer.n_iter,
        batch_iter=cfg.optimizer.batch_iter,
        batch_size=cfg.optimizer.batch_size,
        key=jax.random.PRNGKey(cfg.model.seed),
    )

    # Use the trained model to create a potential function.
    potential_fn = lambda x: my_model.potential.apply(my_model.params, x)

    # Try inference.
    max_time = max(adata.obs[cfg.organism.time_obs])
    idx = adata.obs[cfg.organism.time_obs] == max_time
    x = adata[idx].obsm[cfg.organism.obsm]
    space = adata[idx].obsm[cfg.organism.space_obsm]
    my_model.transform(x, space)

    # Make a 3d scatter plot of the data.
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d", elev=30, azim=40)
    zz = potential_fn(jnp.array(adata.obsm[cfg.organism.obsm]))
    xx = adata.obsm[cfg.organism.viz_obsm][:, 0]
    yy = adata.obsm[cfg.organism.viz_obsm][:, 1]
    ax.scatter(
        xs=xx,
        ys=yy,
        zs=zz,
        s=cfg.plot.scatter.size,
        c=adata.obs[cfg.organism.time_obs],
        cmap=cfg.plot.scatter.cmap,
        zorder=1,
        alpha=cfg.plot.scatter.alpha,
    )

    # Interpolate the potential function.
    xx_surface = jnp.linspace(min(xx), max(xx), 50)
    yy_surface = jnp.linspace(min(yy), max(yy), 50)
    xx_surface, yy_surface = jnp.meshgrid(xx_surface, yy_surface, indexing="xy")
    interp = interpolate.RBFInterpolator(
        jnp.vstack((xx.ravel(), yy.ravel())).T, zz, smoothing=1e-4
    )
    zz_surface = interp(jnp.vstack((xx_surface.ravel(), yy_surface.ravel())).T)
    zz_surface = zz_surface.reshape(xx_surface.shape)

    # Plot the potential function as a surface.
    ax.plot_surface(xx_surface, yy_surface, zz_surface, alpha=0.5, zorder=-1)
    ax.contour(
        xx_surface,
        yy_surface,
        zz_surface,
        zdir="z",
        offset=min(zz_surface.min(), zz.min()),
        **cfg.plot.contour,
    )

    # # Decorate the plot.
    ax.set_xlabel(f"{cfg.organism.viz_obsm} 1")
    ax.set_ylabel(f"{cfg.organism.viz_obsm} 2")
    ax.set_zlabel("Interpolated potential")

    plt.title(cfg.plot.title)
    image = wandb.Image(plt)
    wandb.log({"3d_potential": image})
    plt.show()


if __name__ == "__main__":
    main()
