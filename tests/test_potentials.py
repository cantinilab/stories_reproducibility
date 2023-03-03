import jax.numpy as jnp
from jax import vmap
import space_time


def test_styblinski():
    """Test the styblinski potential."""

    E = space_time.potentials.styblinski_potential

    # Create a meshgrid
    xx, yy = jnp.meshgrid(
        jnp.linspace(-4, 4, 50), jnp.linspace(-4, 4, 50), indexing="xy"
    )

    # Can we compute the potential on the meshgrid?
    vmap(vmap(E))(jnp.stack([xx, yy], axis=-1))
