from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
from space_time import implicit_steps


class MongeLinearImplicitStep(implicit_steps.ImplicitStep):
    """Implicit proximal step with the Wasserstein distance, learning the velocity field."""

    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
    ):
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> jnp.array:
        """Implicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        def proximal_cost(v, inner_x, inner_a):

            # Compute the Wasserstein distance
            cost = tau**2 * jnp.sum(inner_a * v**2)

            # Return the proximal cost
            y = inner_x + tau * v
            return tau * jnp.sum(potential_fun(y)) + cost

        gd = jaxopt.GradientDescent(
            fun=proximal_cost, maxiter=self.maxiter, implicit_diff=self.implicit_diff
        )
        v, _ = gd.run(jnp.zeros(x.shape), inner_x=x, inner_a=a)
        y = x + tau * v
        return y, space

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> jnp.array:
        """Implicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        def proximal_cost(v, inner_x, inner_potential_params, inner_a):

            # Compute the Wasserstein distance
            cost = tau**2 * jnp.sum(inner_a * v**2)

            # Return the proximal cost
            y = inner_x + tau * v
            potential_fun = lambda u: potential_network.apply(inner_potential_params, u)
            return tau * jnp.sum(potential_fun(y)) + cost

        gd = jaxopt.GradientDescent(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )
        y, _ = gd.run(
            jnp.zeros(x.shape),
            inner_x=x,
            inner_potential_params=potential_params,
            inner_a=a,
        )
        return y, space
