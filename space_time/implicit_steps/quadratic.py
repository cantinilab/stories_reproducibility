from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
import optax
import wandb
from space_time import implicit_steps

from ott.geometry.pointcloud import PointCloud
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein


class QuadraticImplicitStep(implicit_steps.ImplicitStep):
    def __init__(
        self,
        epsilon: float = 1.0,
        maxiter: int = 100,
        implicit_diff: bool = True,
        sinkhorn_iter: int = 50,
        fused: float = 1.0,
        wb: bool = False,
    ):
        """Implicit proximal step with the Gromov-Wasserstein distance.

        Args:
            epsilon (float, optional): Entropic regularizaiton. Defaults to 1.0.
            maxiter (int, optional): The maximum number of iterations in the optimization loop. Defaults to 100.
            implicit_diff (bool, optional): Whether to differentiate implicitly through the optimization loop. Defaults to True.
            sinkhorn_iter (int, optional): The number of Sinkhorn iterations. Defaults to 50.
            fused (float, optional): The fused penalty. Defaults to 1.0.
            wb (bool, optional): Whether to log the losses with wandb. Defaults to False.
        """
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.sinkhorn_iter = sinkhorn_iter
        self.fused = fused
        self.wb = wb

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the Gromov-Wasserstein distance. This "inference
        step" takes a potential function as an input, and will not be differentiated
        through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_fun (Callable): The potential function.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics coordinates and the new
            spatial coordinates. As opposed to the linear case, the spatial coordinates
            are updated.
        """
        # Initialize the GW solver.
        solver = GromovWasserstein()

        def proximal_cost(y, inner_x, inner_a, dim_x=x.shape[1]):
            """Helper function to compute the proximal cost."""

            # Define the geometries of the problem.
            geom_xx = PointCloud(inner_x, inner_x, epsilon=self.epsilon)
            geom_yy = PointCloud(y, y, epsilon=self.epsilon)
            geom_xy = PointCloud(inner_x, y, epsilon=self.epsilon)

            # Define the quadratic problem.
            problem = QuadraticProblem(
                geom_xx,
                geom_yy,
                geom_xy,
                fused_penalty=self.fused,
                a=inner_a,
                b=inner_a,
            )

            # Solve the quadratic problem.
            out = solver(problem)

            # Retrieve the regularized Gromov-Wasserstein cost.
            gw_cost = out.reg_gw_cost

            # Return the proximal cost
            return tau * jnp.sum(inner_a * potential_fun(y[:, :dim_x])) + gw_cost

        # Setup a gradient descent optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the gradient descent.
        init_x = jnp.concatenate((x, space), axis=1)
        state = opt.init_state(init_x, inner_x=init_x, inner_a=a)

        @jax.jit
        def jitted_update(y, state):
            return opt.update(y, state, inner_x=init_x, inner_a=a)

        y = init_x
        for _ in range(self.maxiter):
            y, state = jitted_update(y, state)
            if self.wb:
                wandb.log({"proximal_cost": state.error})
            if state.error < 1e-6:
                break

        # Return the new omics and the new space.
        return y[:, : x.shape[1]], y[:, x.shape[1] :]

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the Gromov-Wasserstein distance. This "training
        step" takes a potential network as an input, and will be differentiated through.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_network (nn.Module): The potential network.
            potential_params (optax.Params): The potential network parameters.
            tau (float): The proximal step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics coordinates and the new
            spatial coordinates. As opposed to the linear case, the spatial coordinates
            are updated.
        """

        # Initialize the GW solver.
        solver = GromovWasserstein(
            min_iterations=self.sinkhorn_iter,
            max_iterations=self.sinkhorn_iter,
            implicit_diff=None,
        )

        def proximal_cost(
            y,
            inner_x,
            inner_potential_params,
            inner_a,
            dim_x=x.shape[1],
        ):
            """Helper function to compute the proximal cost."""

            # Define the geometries of the problem.
            geom_xx = PointCloud(inner_x, inner_x, epsilon=self.epsilon)
            geom_yy = PointCloud(y, y, epsilon=self.epsilon)
            geom_xy = PointCloud(inner_x, y, epsilon=self.epsilon)

            # Define the quadratic problem.
            problem = QuadraticProblem(
                geom_xx,
                geom_yy,
                geom_xy,
                fused_penalty=self.fused,
                a=inner_a,
                b=inner_a,
            )

            # Solve the quadratic problem.
            out = solver(problem)

            # Retrieve the regularized Gromov-Wasserstein cost.
            gw_cost = out.reg_gw_cost

            # Return the proximal cost
            potential_fun = lambda u: potential_network.apply(inner_potential_params, u)
            return tau * jnp.sum(inner_a * potential_fun(y[:, :dim_x])) + gw_cost

        # Setup a gradient descent optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the gradient descent.
        init_x = jnp.concatenate((x, space), axis=1)
        y, state = opt.run(
            init_x,
            inner_x=init_x,
            inner_potential_params=potential_params,
            inner_a=a,
        )

        # Return the new omics and the new space.
        return y[:, : x.shape[1]], y[:, x.shape[1] :]
