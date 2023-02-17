import jax.numpy as jnp
from jax import vmap
import space_time
import jax

key = jax.random.PRNGKey(0)
n_cells, n_dims = 7, 2
cells = 0.5 * jax.random.normal(key, shape=(n_cells, n_dims))


def test_implicit_wass_gradient_flow():
    """Test the gradient flow with an implicit Wasserstein proximal step."""

    proximal_step = space_time.implicit_steps.implicit_wasserstein_proximal_step
    E = space_time.potentials.styblinski_potential
    n_proximal_steps = 10
    predictions = space_time.gradient_flow.gradient_flow(
        cells,
        E,
        proximal_step,
        n_proximal_steps=n_proximal_steps,
        tau=0.1,
    )

    assert len(predictions) == n_proximal_steps


def test_explicit_wass_gradient_flow():
    """Test the gradient flow with an explicit Wasserstein proximal step."""

    proximal_step = space_time.explicit_steps.explicit_wasserstein_proximal_step
    E = space_time.potentials.styblinski_potential
    n_proximal_steps = 50
    predictions = space_time.gradient_flow.gradient_flow(
        cells,
        E,
        proximal_step,
        n_proximal_steps=n_proximal_steps,
        tau=1e-2,
    )

    assert len(predictions) == n_proximal_steps


def test_implicit_gw_gradient_flow():
    """Test the gradient flow with an implicit Gromov-Wasserstein proximal step."""

    proximal_step = space_time.implicit_steps.implicit_gromov_wasserstein_proximal_step
    E = space_time.potentials.styblinski_potential
    n_proximal_steps = 5
    predictions = space_time.gradient_flow.gradient_flow(
        cells,
        E,
        proximal_step,
        n_proximal_steps=n_proximal_steps,
        tau=0.1,
    )

    assert len(predictions) == n_proximal_steps


def test_explicit_gw_gradient_flow():
    """Test the gradient flow with an explicit Gromov-Wasserstein proximal step."""

    proximal_step = space_time.explicit_steps.explicit_gromov_wasserstein_proximal_step
    E = space_time.potentials.styblinski_potential
    n_proximal_steps = 50
    predictions = space_time.gradient_flow.gradient_flow(
        cells,
        E,
        proximal_step,
        n_proximal_steps=n_proximal_steps,
        tau=1e-2,
        fused=1.0,
    )

    assert len(predictions) == n_proximal_steps
