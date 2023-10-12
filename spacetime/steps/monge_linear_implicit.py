from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
import optax
from .proximal_step import ProximalStep


class MongeLinearImplicitStep(ProximalStep):
    """This class defines an implicit proximal step corresponding to the squared
    Wasserstein distance, assuming the transportation plan is the identity (each cell
    mapped to itself). This step is "implicit" in the sense that instead of computing a
    velocity field it predicts the next timepoint as an argmin and thus requires solving
    an optimization problem.

    Args:
        maxiter (int, optional): The maximum number of iterations for the optimization
            loop. Defaults to 100.
        implicit_diff (bool, optional): Whether to differentiate implicitly through the
            optimization loop. Defaults to True.
        log_callback (Callable, optional): A callback used to log the proximal loss.
            Defaults to None.
        tol (float, optional): The tolerance for the optimization loop. Defaults to 1e-8.
    """

    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
        log_callback: Callable = None,
        tol: float = 1e-8,
    ):
        self.log_callback = log_callback
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.tol = tol

        self.opt_hyperparams = {
            "maxiter": maxiter,
            "implicit_diff": implicit_diff,
            "tol": tol,
        }

    def inference_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs a linear implicit step on the input distribution and returns the
        updated distribution, given a potential function. The spatial coordinates
        remain unchanged in a linear proximal step. If logging is available, logs the
        proximal cost.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2).
        """

        # Define a helper function to compute the proximal cost.
        def proximal_cost(y, inner_x):
            potential_term = jnp.sum(potential_fun(y))
            return potential_term + tau * 0.5 * jnp.linalg.norm(inner_x - y) ** 2

        # Define the optimizer.
        opt = jaxopt.LBFGS(fun=proximal_cost, **self.opt_hyperparams)

        @jax.jit
        def jitted_update(y, state):
            return opt.update(y, state, inner_x=x)

        # Run the gradient descent, and log the proximal cost.
        y = x.copy()
        state = opt.init_state(y, inner_x=x)
        for _ in range(self.maxiter):
            y, state = jitted_update(y, state)
            if self.log_callback:
                self.log_callback({"proximal_cost": state.error})
            if state.error < self.tol:
                break

        # Return the new omics coordinates and spatial coordinates. The spatial
        # coordinates are not updated.
        return y, space

    def training_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs a linear implicit step on the input distribution and returns the
        updated distribution. This function differs from the inference step in that it
        takes a potential network as input and returns the updated distribution. The
        spatial coordinates remain unchanged in a linear proximal step. Logging is
        not available in this function because it rpevents implicit differentiation.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2).
        """

        # Define a helper function to compute the proximal cost.
        def proximal_cost(y, inner_x, inner_potential_params):
            potential_term = jnp.sum(potential_network.apply(inner_potential_params, y))
            return potential_term + tau * 0.5 * jnp.linalg.norm(inner_x - y) ** 2

        # Define the optimizer.
        opt = jaxopt.LBFGS(fun=proximal_cost, **self.opt_hyperparams)

        # Run the optimization loop.
        y, _ = opt.run(x, inner_x=x, inner_potential_params=potential_params)

        # Return the new omics coordinates and spatial coordinates. The spatial
        # coordinates are not updated.
        return y, space
