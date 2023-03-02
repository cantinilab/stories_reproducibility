from typing import Callable, List
from jax import grad, vmap
import jax.numpy as jnp
import jax


def get_drift(potential: Callable) -> Callable:
    """Get the drift function for a potential.

    Args:
        potential (Callable): A potential function.

    Returns:
        Callable: The drift function.
    """
    return vmap(grad(potential))


def euler_maruyama(
    key: jax.random.PRNGKey,
    potential: Callable,
    x: jnp.ndarray,
    dt: float,
    sd: float,
    n_steps: int,
) -> List[jnp.array]:
    """Generate a sequence of populations using the Euler-Maruyama method.

    Args:
        key (jax.random.PRNGKey): The PRNGKey.
        potential (Callable): A potential function.
        x (jnp.ndarray): The initial distribution.
        dt (float): The time step.
        sd (float): The standard deviation of the noise.
        n_steps (int): The number of steps.

    Returns:
        List[jnp.array]: A list of distributions.
    """

    keys = jax.random.split(key, n_steps)

    # Get the drift function from the potential.
    drift = get_drift(potential)

    # Set the initial population.
    y = x.copy()

    # Perform the Euler-Maruyama steps.
    yy = [y]
    for i in range(n_steps):

        y = y - drift(y) * dt + jnp.sqrt(dt) * sd * jax.random.normal(keys[i], y.shape)
        yy.append(y)

    return yy
