# Imports
import anndata as ad
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from omegaconf import DictConfig
from space_time import explicit_steps, implicit_steps, model, potentials


@hydra.main(version_base=None, config_path="configs", config_name="celegans")
def main(config: DictConfig) -> None:

    # Load the data.
    adata = ad.read_h5ad(config.dataset.path)

    # Define the protential.
    potential = potentials.MLPPotential(features=config.potential.features)

    # Define the proximal step.
    if config.model.proximal_step == "linear_explicit":
        proximal_step = explicit_steps.linear.LinearExplicitStep()
    elif config.model.proximal_step == "quadratic_explicit":
        proximal_step = explicit_steps.quadratic.QuadraticExplicitStep()
    elif config.model.proximal_step == "linear_implicit":
        proximal_step = implicit_steps.linear.LinearImplicitStep()
    elif config.model.proximal_step == "quadratic_implicit":
        proximal_step = implicit_steps.quadratic.QuadraticImplicitStep()
    else:
        raise ValueError(f"Proximal step {config.model.proximal_step} not recognized.")

    # Define the model.
    my_model = model.SpaceTime(
        potential=potential,
        proximal_step=proximal_step,
        tau=config.model.tau,
        debias=config.model.debias,
        epsilon=config.model.epsilon,
        teacher_forcing=config.model.teacher_forcing,
    )

    # Fit the model.
    my_model.fit_adata(
        adata=adata,
        time_obs=config.dataset.time_obs,
        obsm=config.dataset.obsm,
        space_obsm=config.dataset.space_obsm,
        optimizer=optax.adam(config.optimizer.learning_rate),
        n_iter=config.optimizer.n_iter,
        batch_iter=config.optimizer.batch_iter,
        batch_size=config.optimizer.batch_size,
        key=jax.random.PRNGKey(config.model.seed),
    )

    # Use the trained model to create a potential function.
    potential_fn = lambda x: my_model.potential.apply(my_model.params, x)

    # Try inference.
    max_time = max(adata.obs[config.dataset.time_obs])
    idx = adata.obs[config.dataset.time_obs] == max_time
    x = adata[idx].obsm[config.dataset.obsm]
    space = adata[idx].obsm[config.dataset.space_obsm]
    my_model.transform(x, space)

    # Make a 3d scatter plot of the data.
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d", elev=30, azim=40)
    zz = potential_fn(jnp.array(adata.obsm[config.dataset.obsm]))
    xx = adata.obsm[config.dataset.obsm][:, 0]
    yy = adata.obsm[config.dataset.obsm][:, 1]
    ax.scatter(
        xs=xx,
        ys=yy,
        zs=zz,
        s=config.plot.scatter.size,
        c=adata.obs[config.dataset.time_obs],
        cmap=config.plot.scatter.cmap,
        zorder=1,
        alpha=config.plot.scatter.alpha,
    )

    # Plot the potential function as a surface.
    xx, yy = jnp.linspace(min(xx), max(xx), 50), jnp.linspace(min(yy), max(yy), 50)
    xx, yy = jnp.meshgrid(xx, xx, indexing="xy")
    zz = potential_fn(jnp.stack([xx, yy], axis=-1))
    ax.plot_surface(xx, yy, zz, alpha=0.5, zorder=-1)
    ax.contour(xx, yy, zz, zdir="z", offset=zz.min(), **config.plot.contour)

    # Decorate the plot.
    ax.set_xlabel(f"{config.dataset.obsm} 1")
    ax.set_ylabel(f"{config.dataset.obsm} 2")
    ax.set_zlabel("Potential")
    plt.title(config.plot.title)
    plt.show()


if __name__ == "__main__":
    main()
