from typing import Callable, List
import jax.numpy as jnp


def gradient_flow(
    x: jnp.array,
    potential_fun: Callable,
    proximal_step: Callable,
    n_proximal_steps: int,
    tau: float,
    **kwargs
) -> List[jnp.array]:
    """Perform a gradient flow.

    Args:
        x (jnp.array): The input distribution, size (N, d).
        potential_fun (Callable): The potential function, taking a pointcloud of size (N, d) as input.
        proximal_step (Callable): The proximal step function
        n_proximal_steps (int): The number of proximal steps
        tau (float): The time step
        kwargs: Additional arguments to the proximal step function

    Returns:
        List[jnp.array]: A list of the intermediate distributions, each of size (N, d)
    """
    y = x.copy()
    yy = []
    for _ in range(n_proximal_steps):
        y = proximal_step(y, potential_fun, tau, **kwargs)
        yy.append(y)
    return yy
