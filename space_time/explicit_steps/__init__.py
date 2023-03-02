# init
import abc
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import optax


class ExplicitStep(abc.ABC):
    def inference_step(
        self,
        x: jnp.array,
        potential_fun: Callable,
        tau: float,
    ) -> jnp.array:
        """Explicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """
        pass

    def training_step(
        self,
        x: jnp.array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> jnp.array:
        """Explicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """
        pass