from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
from space_time import implicit_steps

from ott.geometry.pointcloud import PointCloud
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein


class QuadraticImplicitStep(implicit_steps.ImplicitStep):
    """Implicit proximal step with the Gromov-Wasserstein distance."""

    def __init__(
        self,
        epsilon: float = None,
        maxiter: int = 100,
        implicit_diff: bool = True,
        sinkhorn_iter: int = 50
    ):
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.sinkhorn_iter = sinkhorn_iter

    def inference_step(
        self,
        x: jnp.array,
        potential_fun: Callable,
        tau: float,
        fused: float = 1.0,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Implicit proximal step with the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.
            n_iter (int, optional): The number of gradient descent steps. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 5e-2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        # Initialize the GW solver.
        solver = GromovWasserstein()

        def proximal_cost(y, inner_x, inner_a):

            # Solve the quadratic problem.
            geom_xx = PointCloud(inner_x, inner_x, epsilon=self.epsilon)
            geom_yy = PointCloud(y, y, epsilon=self.epsilon)
            geom_xy = PointCloud(inner_x, y, epsilon=self.epsilon)
            out = solver(QuadraticProblem(
                geom_xx,
                geom_yy,
                geom_xy,
                fused_penalty=fused,
                a=inner_a,
                b=inner_a,
            ))
            cost = out.reg_gw_cost

            # Return the proximal cost
            return tau * jnp.sum(potential_fun(y)) + cost

        gd = jaxopt.GradientDescent(fun=proximal_cost, maxiter=self.maxiter, implicit_diff=self.implicit_diff)
        y, _ = gd.run(x, inner_x=x, inner_a=a)
        return y

    def training_step(
        self,
        x: jnp.array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        fused: float = 1.0,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Implicit proximal step with the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.
            n_iter (int, optional): The number of gradient descent steps. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 5e-2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """


        # Initialize the GW solver.
        solver = GromovWasserstein(
            min_iterations=self.sinkhorn_iter,
            max_iterations=self.sinkhorn_iter,
            implicit_diff=None
        )

        def proximal_cost(
            y,
            inner_x,
            inner_potential_params,
            inner_a,
        ):

            # Solve the quadratic problem.
            geom_xx = PointCloud(inner_x, inner_x, epsilon=self.epsilon)
            geom_yy = PointCloud(y, y, epsilon=self.epsilon)
            geom_xy = PointCloud(inner_x, y, epsilon=self.epsilon)
            out = solver(QuadraticProblem(
                geom_xx,
                geom_yy,
                geom_xy,
                fused_penalty=fused,
                a=inner_a,
                b=inner_a,
            ))
            cost = out.reg_gw_cost

            # Return the proximal cost
            return tau * jnp.sum(potential_network.apply(inner_potential_params, y)) + cost

        gd = jaxopt.GradientDescent(fun=proximal_cost, maxiter=self.maxiter, implicit_diff=self.implicit_diff)
        y, _ = gd.run(x, inner_x=x, inner_potential_params=potential_params, inner_a=a)
        return y