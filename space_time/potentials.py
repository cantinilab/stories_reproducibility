import jax.numpy as jnp

def styblinski_potential(u: jnp.ndarray) -> jnp.ndarray:
    """The Styblinski potential has 4 minima.

    Args:
        u (jnp.ndarray): A 2D point or a pointcloud of size (N, 2).

    Returns:
        jnp.ndarray: _description_
    """    
    return jnp.sum(u**4 - 16 * u**2 + 5 * u) / 2