from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import grad, vmap
from space_time import explicit_steps


class LinearExplicitStep(explicit_steps.ExplicitStep):
    """Explicit proximal step with the Wasserstein distance."""

    def inference_step(
        self,
        x: jnp.array,
        potential_fun: Callable,
        tau: float,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Explicit proximal step with the Wasserstein distance. This "inference" step uses a
        callable potential function, while the "training" step requires a neural network and
        its parameters to perform the step.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        return x - tau * vmap(grad(potential_fun))(x)


    def training_step(
        self,
        x: jnp.array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Explicit proximal step with the Wasserstein distance. This "training" step uses a
        neural network and its parameters to perform the step, while the "inference" step
        requires a callable potential function.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_network (nn.Module): A neural network, taking a pointcloud of size (N, d) as input.
            potential_params (optax.Params): Parameters of the neural network.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        potential_fun = lambda u: potential_network.apply(potential_params, u)
        return x - tau * vmap(grad(potential_fun))(x)