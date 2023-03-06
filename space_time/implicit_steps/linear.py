from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
from space_time import implicit_steps

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


class LinearImplicitStep(implicit_steps.ImplicitStep):
    """Implicit proximal step with the Wasserstein distance."""

    def __init__(
        self,
        epsilon: float = 1.0,
        maxiter: int = 100,
        sinkhorn_iter: int = 100,
        implicit_diff: bool = True,
        # stepsize: float = 1e-2,
    ):
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.sinkhorn_iter = sinkhorn_iter
        self.implicit_diff = implicit_diff
        # self.stepsize = stepsize

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        potential_fun: Callable,
        tau: float,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Implicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        def proximal_cost(y, inner_x, inner_a):

            # Compute the Sinkhorn divergence
            out = sinkhorn_divergence(
                PointCloud,
                y,
                inner_x,
                a=inner_a,
                b=inner_a,
                static_b=True,
                epsilon=self.epsilon,
            )
            cost = out.divergence

            # Return the proximal cost
            return tau * jnp.sum(potential_fun(y)) + cost

        gd = jaxopt.GradientDescent(
            fun=proximal_cost, maxiter=self.maxiter, implicit_diff=self.implicit_diff
        )
        y, _ = gd.run(x, inner_x=x, inner_a=a)
        return y

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Implicit proximal step with the Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        solver = Sinkhorn(
            min_iterations=self.sinkhorn_iter,
            max_iterations=self.sinkhorn_iter,
            implicit_diff=None,
        )

        def proximal_cost(y, inner_x, inner_potential_params, inner_a):

            # Solve the linear problem.
            geom = PointCloud(inner_x, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            cost = out.reg_ot_cost

            # Debias
            geom = PointCloud(y, y, epsilon=self.epsilon)
            out = solver(LinearProblem(geom, a=inner_a, b=inner_a))
            cost -= 0.5 * out.reg_ot_cost

            # Return the proximal cost
            return (
                tau * jnp.sum(potential_network.apply(inner_potential_params, y)) + cost
            )

        # # TODO: tolerance?
        gd = jaxopt.GradientDescent(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
            # stepsize=self.stepsize,
        )
        y, _ = gd.run(x, inner_x=x, inner_potential_params=potential_params, inner_a=a)
        return y
