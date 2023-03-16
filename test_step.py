import hydra
import matplotlib.pyplot as plt
import wandb
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="configs/test_step", config_name="config")
def main(cfg: DictConfig) -> None:

    import jax
    import jax.numpy as jnp
    import optax
    import space_time
    from jax import vmap
    from space_time import explicit_steps, implicit_steps

    # start a new wandb run to track this script
    print(dict(cfg))
    print(f"JAX device type: {jax.devices()[0].device_kind}")

    wandb.init(
        project="spacetime-step",
        config=flatten(OmegaConf.to_container(cfg), reducer="dot"),
        mode=cfg.wandb.mode,
    )

    # Define the Styblinski potential.
    print("Defining potential")
    if cfg.potential.ground_truth == "styblinski":
        potential_fn = space_time.potentials.styblinski_potential
    else:
        raise ValueError(f"Potential {cfg.potential} not recognized.")

    # Compute the potential over a grid.
    print("Computing potential over a grid")
    xx, yy = jnp.meshgrid(
        jnp.linspace(-cfg.grid.scale, cfg.grid.scale, cfg.grid.number),
        jnp.linspace(-cfg.grid.scale, cfg.grid.scale, cfg.grid.number),
    )
    zz = vmap(potential_fn)(jnp.stack((xx.flatten(), yy.flatten()), axis=1))

    # Random number generators.
    print("Generating initial population")
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)

    # Generate omics data.
    n_cells = cfg.population.n_cells
    n_genes = cfg.population.n_genes
    x = 0.5 * jax.random.normal(key1, (n_cells, n_genes))

    # Generate spatial data.
    n_dims = cfg.population.n_dims
    space = 0.2 * jax.random.normal(key2, (n_cells, n_dims))
    space += jnp.arange(n_cells).reshape(-1, 1) % 2 - 0.5

    # Generate marginals.
    marginals = jax.random.uniform(key3, (n_cells,))
    marginals = marginals / jnp.sum(marginals)

    # Define the proximal step.
    print("Defining proximal step")
    if cfg.proximal_step.type == "linear_explicit":
        proximal_step = explicit_steps.linear.LinearExplicitStep()
    elif cfg.proximal_step.type == "quadratic_explicit":
        proximal_step = space_time.explicit_steps.quadratic.QuadraticExplicitStep(
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

    # Run the gradient flow.
    print("Running gradient flow")
    n_iter, tau = cfg.evolution.n_timepoints - 1, cfg.proximal_step.tau
    x_list, space_list = [x], [space]
    for i in tqdm(range(n_iter)):
        next_x, next_space = proximal_step.inference_step(
            x_list[-1],
            space_list[-1],
            marginals,
            potential_fn,
            tau=tau,
        )
        x_list.append(next_x)
        space_list.append(next_space)

    # Define the model.
    print("Defining model")
    features = cfg.potential.features
    potential_network = space_time.potentials.MLPPotential(features=features)
    my_model = space_time.model.SpaceTime(
        potential=potential_network,
        proximal_step=proximal_step,
        tau=cfg.model.tau,
        quadratic=cfg.model.quadratic,
        teacher_forcing=cfg.model.teacher_forcing,
        wb=True,
    )

    # Fit the model.
    print("Fitting model")
    input_distributions = [
        {"x": x, "space": space} for x, space in zip(x_list, space_list)
    ]
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=cfg.model.learning_rate),
    )
    my_model.fit(
        input_distributions=input_distributions,
        n_iter=cfg.model.n_iter,
        batch_iter=cfg.model.batch_iter,
        batch_size=cfg.model.batch_size,
        optimizer=optimizer,
    )

    # Use the trained model to create a potential function.
    print("Computing potential over a grid")
    trained_potential_fn = lambda x: my_model.potential.apply(my_model.params, x)
    zz_pred = vmap(trained_potential_fn)(jnp.stack((xx.ravel(), yy.ravel()), axis=1))

    # Plot the gradient flow on the ground truth potential.
    print("Plotting gradient flow on the ground truth potential")
    cmap = plt.get_cmap("coolwarm")
    plt.contourf(xx, yy, zz.reshape(xx.shape), levels=20, alpha=0.8)
    for i in range(len(x_list)):
        plt.scatter(
            x_list[i][:, 0],
            x_list[i][:, 1],
            color=cmap(i / len(x_list)),
            edgecolor="black",
            s=1e3 * marginals,
            alpha=0.8,
        )
    plt.title("Cells on the Styblinski omics potential")
    image = wandb.Image(plt)
    wandb.log({"omics_evolution": image})
    plt.clf()

    # Compute the colors for the cells in physical space.
    r = x_list[-1][:, 0]
    r -= jnp.min(r)
    r /= jnp.max(r)
    g = x_list[-1][:, 1]
    g -= jnp.min(g)
    g /= jnp.max(g)
    b = 0.5 * jnp.ones(n_cells)
    colors = list(zip(r, g, b))

    # Plot the cells in physical space.
    print("Plotting cells in physical space")
    for i in range(len(space_list)):
        plt.scatter(
            space_list[i][:, 0],
            space_list[i][:, 1],
            c=colors,
            edgecolor="black",
            s=2e3 * marginals,
            alpha=i * 0.8 / len(space_list),
        )
    plt.title("Cells in physical space, colored by omics at last timepoint")
    image = wandb.Image(plt)
    wandb.log({"space_evolution": image})
    plt.clf()

    # Plot the trained potential.
    print("Plotting trained potential")
    plt.contourf(xx, yy, zz_pred.reshape(xx.shape), levels=20, alpha=0.8)
    plt.title("Trained potential")
    image = wandb.Image(plt)
    wandb.log({"learned_potential": image})
    plt.clf()

    plt.close()


if __name__ == "__main__":
    main()
