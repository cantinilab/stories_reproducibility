from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
import wandb
from space_time import implicit_steps


class MongeLinearImplicitStep(implicit_steps.ImplicitStep):
    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
        wb: bool = False,
    ):
        """Proximal step with the squared Wasserstein distance, assuming the
        transportation plan is the identity (each cell mapped to itself).

        Args:
            maxiter (int, optional): The mamximum number of iterations for the optimization loop. Defaults to 100.
            implicit_diff (bool, optional): Whether to differentiate implicitly through the optimization loop. Defaults to True.
            wb (bool, optional): Whether to log the proximal loss using wandb. Defaults to False.
        """
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.wb = wb

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the squared Wasserstein distance. This
        "inference step" takes a potential function as an input, and will not be
        differentiated through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_fun (Callable): The potential function.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics coordinates and the new
            spatial coordinates. The spatial coordinates are not updated.
        """

        def proximal_cost(v, inner_x, inner_a):
            """Helper function to compute the proximal cost."""

            # Compute the squared Wasserstein term.
            wass_term = jnp.sum(inner_a.reshape(-1, 1) * v**2)

            # Compute the potential term.
            y = inner_x + tau * v
            potential_term = jnp.sum(inner_a * potential_fun(y))

            # Return the proximal cost
            return potential_term + tau * wass_term

        # Define the optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the gradient descent, and log the proximal cost.
        v, state = jnp.zeros(x.shape), opt.init_state(
            jnp.zeros(x.shape), inner_x=x, inner_a=a
        )
        for _ in range(self.maxiter):
            v, state = opt.update(v, state, inner_x=x, inner_a=a)
            if self.wb:
                wandb.log({"proximal_cost": state.error})

        # Return the new omics coordinates and spatial coordinates. The spatial
        # coordinates are not updated.
        return x + tau * v, space

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the squared Wasserstein distance. This
        "training step" takes a potential network as an input, and will be
        differentiated through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_network (nn.Module): The potential neural network.
            potential_params (optax.Params): The potential network's parameters.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics coordinates and the new
            spatial coordinates. The spatial coordinates are not updated.
        """

        def proximal_cost(v, inner_x, inner_potential_params, inner_a):
            """Helper function to compute the proximal cost."""

            # Compute the squared Wasserstein term.
            wass_term = jnp.sum(inner_a.reshape(-1, 1) * v**2)

            # Compute the potential term.
            fun = lambda u: potential_network.apply(inner_potential_params, u)
            potential_term = jnp.sum(inner_a * fun(inner_x + tau * v))

            # Return the proximal cost
            return potential_term + tau * wass_term

        # Define the optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the optimization loop.
        v, state = opt.run(
            jnp.zeros(x.shape),
            inner_x=x,
            inner_potential_params=potential_params,
            inner_a=a,
        )

        # Return the new omics coordinates and spatial coordinates. The spatial
        # coordinates are not updated.
        return x + tau * v, space
