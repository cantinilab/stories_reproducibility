from typing import Callable
import jax.numpy as jnp
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.geometry.pointcloud import PointCloud
from jax import jit
import jax


def implicit_wasserstein_proximal_step(
    x: jnp.array,
    potential_fun: Callable,
    tau: float,
    n_iter: int = 100,
    learning_rate: float = 5e-2,
) -> jnp.array:
    """Implicit proximal step with the Wasserstein distance.

    Args:
        x (jnp.array): Input distribution, size (N, d).
        potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
        tau (float, optional): Time step.
        n_iter (int, optional): The number of gradient descent steps. Defaults to 100.
        learning_rate (float, optional): Learning rate. Defaults to 5e-2.

    Returns:
        jnp.array: The output distribution, size (N, d)
    """

    # Initialize the distribution.
    y = x.copy()

    # Initialize the Sinkhorn solver.
    solver = Sinkhorn()

    def proximal_cost(geom):

        # Solve the linear problem.
        out = solver(LinearProblem(geom))

        # Return the proximal cost and the solver output.
        return tau * potential_fun(geom.y) / y.shape[0] + out.reg_ot_cost, out

    # Create a function computing the gradient of the proximal cost.
    proximal_cost_vg = jit(jax.value_and_grad(proximal_cost, has_aux=True))

    for _ in range(n_iter):

        # Compute the gradient of the proximal cost.
        geom = PointCloud(x, y)
        (_, out), geom_g = proximal_cost_vg(geom)

        # Check that the solver converged.
        assert out.converged

        # Update the distribution.
        y = y - geom_g.y * learning_rate

    return y


def implicit_gromov_wasserstein_proximal_step(
    x: jnp.array,
    potential_fun: Callable,
    tau: float,
    n_iter: int = 200,
    learning_rate: float = 5e-2,
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

    # Initialize the distribution.
    y = x.copy()

    # Initialize the GW solver.
    solver = GromovWasserstein()
    geom_xx = PointCloud(x, x)

    def proximal_cost(geom_yy):

        # Solve the linear problem.
        out = solver(QuadraticProblem(geom_xx, geom_yy))

        # Return the proximal cost and the solver output.
        return tau * potential_fun(geom_yy.y) / y.shape[0] + out.reg_gw_cost, out

    # Create a function computing the gradient of the proximal cost.
    proximal_cost_vg = jit(jax.value_and_grad(proximal_cost, has_aux=True))

    for _ in range(n_iter):

        # Compute the gradient of the proximal cost.
        geom_yy = PointCloud(y, y)
        (_, out), geom_g = proximal_cost_vg(geom_yy)

        # Check that the solver converged.
        assert out.converged

        # Update the distribution.
        y = y - geom_g.y * learning_rate

    return y
