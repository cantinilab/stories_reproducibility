from typing import Callable, List, Union

import jax.numpy as jnp
from jax import jit
from space_time import explicit_steps, implicit_steps
from tqdm import tqdm


def gradient_flow(
    x: jnp.array,
    potential_fun: Callable,
    proximal_step: Union[implicit_steps.ImplicitStep, explicit_steps.ExplicitStep],
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
    yy = [y]
    print("Jitting the proximal step, this may take a while...")
    jitted_inf_step = jit(
        lambda u: proximal_step.inference_step(
            u, potential_fun=potential_fun, tau=tau, **kwargs
        )
    )
    for _ in tqdm(range(n_proximal_steps)):
        y = jitted_inf_step(y)
        yy.append(y)
    return yy
