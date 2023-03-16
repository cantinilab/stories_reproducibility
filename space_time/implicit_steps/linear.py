from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
import optax
import wandb
from space_time import implicit_steps

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


class LinearImplicitStep(implicit_steps.ImplicitStep):
    def __init__(
        self,
        epsilon: float = 1.0,
        maxiter: int = 100,
        sinkhorn_iter: int = 100,
        implicit_diff: bool = True,
        wb: bool = False,
    ):
        """Implicit proximal step with a Sinkhorn divergence.

        Args:
            epsilon (float, optional): Entropic regularization. Defaults to 1.0.
            maxiter (int, optional): Maximum number of iterations of the optimization loop. Defaults to 100.
            sinkhorn_iter (int, optional): Number of Sinkhorn iterations. Defaults to 100.
            implicit_diff (bool, optional): Whether differentiate implicitly through the optimization loop. Defaults to True.
            wb (bool, optional): Whether to log the losses with wandb. Defaults to False.
        """
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.sinkhorn_iter = sinkhorn_iter
        self.implicit_diff = implicit_diff
        self.wb = wb

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.array, jnp.array]:
        """Implicit proximal step with the Wasserstein distance. This "inference step"
        takes a potential function as an input, and will not be differentiated through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_fun (Callable): The potential function.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.array, jnp.array]: The new omics coordinates and the new
            spatial coordinates. The spatial coordinates are not updated.
        """

        # Define the Sinkhorn solver.
        solver = Sinkhorn(max_iterations=self.sinkhorn_iter)

        def proximal_cost(y, inner_x, inner_a):
            """Helper function to compute the proximal cost."""

            # Solve the linear problem.
            geom = PointCloud(inner_x, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            sink_div = out.reg_ot_cost

            # Debias
            geom = PointCloud(y, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            sink_div -= 0.5 * out.reg_ot_cost

            # Compute the potential cost (note the marginal inner_a).
            potential_cost = jnp.sum(inner_a * potential_fun(y))

            # Return the proximal cost
            return tau * potential_cost + sink_div

        # Define an optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        @jax.jit
        def jitted_update(y, state):
            return opt.update(y, state, inner_x=x, inner_a=a)

        # Run the optimization loop, logging the proximal cost.
        y, state = x, opt.init_state(x, inner_x=x, inner_a=a)
        for _ in range(self.maxiter):
            y, state = jitted_update(y, state)
            if self.wb:
                wandb.log({"proximal_cost": state.error})
            if state.error < 1e-6:
                break

        # Return the new omics coordinates and spatial coordinates
        # (the latter do not change).
        return y, space

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.array, jnp.array]:
        """Implicit proximal step with the Wasserstein distance. This "training step"
        takes a potential network as an input, and will be differentiated through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_network (nn.Module): The potential neural network.
            potential_params (optax.Params): The potential network's parameters.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.array, jnp.array]: The new omics coordinates and the new
            spatial coordinates. The spatial coordinates are not updated.
        """

        # Define the Sinkhorn solver. Note that we disable the implicit
        # differentiation, as we will differentiate through the optimization loop.
        # min_iterations and max_iterations are equal in order to force the the
        # loop to be implemented with jax.lax.scan.
        solver = Sinkhorn(
            min_iterations=self.sinkhorn_iter,
            max_iterations=self.sinkhorn_iter,
            implicit_diff=None,
        )

        def proximal_cost(y, inner_x, inner_potential_params, inner_a):
            """Helper function to compute the proximal cost."""

            # Solve the linear problem.
            geom = PointCloud(inner_x, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            cost = out.reg_ot_cost

            # Debias
            geom = PointCloud(y, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            cost -= 0.5 * out.reg_ot_cost

            # Return the proximal cost
            fun = lambda u: potential_network.apply(inner_potential_params, u)
            return tau * jnp.sum(inner_a * fun(y)) + cost

        # Define an optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the optimization loop.
        y, state = opt.run(
            x,
            inner_x=x,
            inner_potential_params=potential_params,
            inner_a=a,
        )

        # Return the new omics coordinates and spatial coordinates
        # (the latter do not change).
        return y, space
