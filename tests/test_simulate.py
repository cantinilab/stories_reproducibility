import jax
import space_time

def test_euler_maruyama():
    """Test the Euler-Maruyama method."""

    key = jax.random.PRNGKey(0)
    n_cells, n_dims = 7, 2
    cells = 0.5*jax.random.normal(key, shape=(n_cells, n_dims))

    n_steps = 20
    populations = space_time.simulate.euler_maruyama(
        key=key,
        potential=space_time.potentials.styblinski_potential,
        x=cells,
        dt = 0.02,
        n_steps = n_steps,
        sd = .2,
    )

    assert len(populations) == n_steps + 1
