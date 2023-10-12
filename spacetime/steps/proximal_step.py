from abc import ABC, abstractmethod
from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import optax


class ProximalStep(ABC):
    """This abstract class defines the interface for proximal steps. Given a potential
    function, the proximal step updates the input distribution :math:`\\mu_t`.

    A proximal step should implement both an inference step and a training step. The
    inference step is used to generate samples from the model, while the training step
    should eb differentiable and is used to train the model parameters."""

    @abstractmethod
    def inference_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray]:
        """Given a distribution of cells :math:`\\mu_t` and a potential function, this
        function returns :math:`\\mu_{t+\\tau}`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2)
        """
        pass

    @abstractmethod
    def training_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray]:
        """Given a distribution of cells :math:`\\mu_t` and a potential function
        parameterized by a neural network, this function returns :math:`\\mu_{t+\\tau}`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2)
        """
        pass
