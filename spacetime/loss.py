from .steps.proximal_step import ProximalStep
import flax.linen as nn
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.geometry.pointcloud import PointCloud
from typing import Callable, Dict
from ott.problems.linear.linear_problem import LinearProblem
import jax.numpy as jnp
import optax
import jax


def linear_loss(
    x: jnp.ndarray,
    a: jnp.ndarray,
    y: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: float,
    cost_fn: Callable,
    balancedness: float,
    ott_solver: Callable,
    debias: bool,
) -> jnp.ndarray:
    """Compute the Sinkhorn loss. This ignores the spatial component.

    Args:
        x: A pointcloud.
        a: Histogram on x.
        y: Another pointcloud.
        b: Histogram on y.
        epsilon: Entropic regularization parameter.
        cost_fn: Cost function for optimal transport.
        balancedness: Between 0 and 1, 1 being balanced optimal transport.
        ott_solver: The OTT solver to use, i.e. an instance of Sinkhorn or LRSinkhorn.
        debias: Whether to debias the loss.

    Returns:
        The Sinkhorn loss.
    """

    # Compute the Sinkhorn loss between point clouds x and y.
    geom_xy = PointCloud(x, y, epsilon=epsilon, cost_fn=cost_fn)
    problem = LinearProblem(geom_xy, a=a, b=b, tau_a=balancedness, tau_b=balancedness)
    ot_loss = ott_solver(problem).reg_ot_cost

    # We assume x and y to have the same mass, so no need for the m(a) - m(b) term.
    if debias:
        # Debias the Sinkhorn loss with the xx term.
        geom_bias = PointCloud(x, epsilon=epsilon, cost_fn=cost_fn)
        problem = LinearProblem(
            geom_bias, a=a, b=a, tau_a=balancedness, tau_b=balancedness
        )
        ot_loss -= 0.5 * ott_solver(problem).reg_ot_cost

        # Debias the Sinkhorn loss with the yy term.
        geom_bias = PointCloud(y, epsilon=epsilon, cost_fn=cost_fn)
        problem = LinearProblem(
            geom_bias, a=b, b=b, tau_a=balancedness, tau_b=balancedness
        )
        ot_loss -= 0.5 * ott_solver(problem).reg_ot_cost

    return ot_loss


def quadratic_loss(
    x: jnp.ndarray,
    a: jnp.ndarray,
    y: jnp.ndarray,
    b: jnp.ndarray,
    space_x: jnp.ndarray,
    space_y: jnp.ndarray,
    epsilon: float,
    fused_penalty: float,
    balancedness: float,
    ott_solver: Callable,
    debias: bool,
) -> jnp.ndarray:
    """FGW loss
    The linear part of the loss operates on the gene coordinates.
    The quadratic part of the loss operates on the space coordinates.

    The issue with Fused Gromov-Wasserstein is that it is biased, ie
    FGW(x, y) != 0 when x=y. We can compute instead the following:
             FGW(x, y) - 0.5 * FGW(x, x) - 0.5 * FGW(y, y)
    As done in http://proceedings.mlr.press/v97/bunne19a/bunne19a.pdf

    TODO: specify a cost_fn

    Args:
        x: A pointcloud on the space of genes.
        a: Histogram on x.
        y: Another pointcloud on the space of genes.
        b: Histogram on y.
        space_x: A pointcloud on the space of spatial coordinates.
        space_y: Another pointcloud on the space of spatial coordinates.
        epsilon: Entropic regularization parameter.
        fused_penalty: The penalty for the fused term.
        balancedness: Between 0 and 1, 1 being balanced optimal transport.
        ott_solver: The OTT solver to use, i.e. an instance of (LR)GromovWasserstein.
        debias: Whether to debias the loss.

    Returns:
        The FGW loss.
    """

    # Define geometries on space for xx, yy, and on genes for xy.
    geom_xx = PointCloud(space_x, space_x, epsilon=epsilon, scale_cost="max_cost")
    geom_yy = PointCloud(space_y, space_y, epsilon=epsilon, scale_cost="max_cost")
    geom_xy = PointCloud(x, y, epsilon=epsilon)

    # These keyword arguments are passed to all quadratic problems.
    fused_kwds = {
        "fused_penalty": fused_penalty,
        "tau_a": balancedness,
        "tau_b": balancedness,
        "gw_unbalanced_correction": False,
    }

    # Compute the FGW loss between point clouds x and y.
    problem = QuadraticProblem(geom_xx, geom_yy, geom_xy, a=a, b=b, **fused_kwds)
    ot_loss = ott_solver(problem).reg_gw_cost

    if debias:
        # Substracting 0.5 * FGW(x, x).
        geom_bias = PointCloud(x, x, epsilon=epsilon)
        problem = QuadraticProblem(geom_xx, geom_xx, geom_bias, a=a, b=a, **fused_kwds)
        ot_loss -= 0.5 * ott_solver(problem).reg_gw_cost

        # Substracting 0.5 * FGW(y, y).
        geom_bias = PointCloud(y, y, epsilon=epsilon)
        problem = QuadraticProblem(geom_yy, geom_yy, geom_bias, a=b, b=b, **fused_kwds)
        ot_loss -= 0.5 * ott_solver(problem).reg_gw_cost

    # Rescale to make losses more comparable across fused penalty.
    return ot_loss / fused_penalty


def loss_fn(
    params: optax.Params,
    batch: Dict[str, jnp.ndarray],
    teacher_forcing: bool,
    quadratic: bool,
    proximal_step: ProximalStep,
    potential: nn.Module,
    n_steps: int,
    epsilon: float,
    cost_fn: Callable,
    balancedness: float,
    debias: bool,
    fused_penalty: float,
    ott_solver: Callable,
    tau_diff: jnp.ndarray,
) -> jnp.ndarray:
    """The loss function

    Args:
        params: The parameters of the model.
        batch: A batch of data.
        teacher_forcing: Whether to use teacher forcing.
        quadratic: Whether to use the quadratic (FGW) loss instead of the linear (W) one.
        proximal_step: The proximal step, e.g. LinearExplicitStep.
        potential: The potential function parametrized by a neural network.
        n_steps: The number of steps to take.
        epsilon: Entropic regularization parameter.
        cost_fn: Cost function for optimal transport. TODO: for now not used by FGW
        balancedness: Between 0 and 1, 1 being balanced optimal transport.
        debias: Whether to debias the loss (see linear_loss or quadratic_loss).
        fused_penalty: Parameter indicting weight of the fused term.
        ott_solver: The OTT solver to use, i.e. an instance of (LR)GromovWasserstein.
        tau_diff: The difference in time between each timepoint, e.g. [1., 1., 2.].

    Returns:
        The loss.
    """

    # Get the batch's x and space coordinates.
    batch_x, batch_space, batch_a = batch["x"], batch["space"], batch["a"]

    # This is a helper function to compute the loss for a single timepoint.
    # We will chain this function over the timepoints using lax.scan.
    def _through_time(carry, t):
        # Unpack the carry, which contains the x and space across timepoints.
        _x, _space, _a = carry

        # Predict the timepoint t+1 using the proximal step.
        pred_x = proximal_step.chained_training_steps(
            _x[t], _a[t], potential, params, tau_diff[t], n_steps
        )

        # Compute the loss between the predicted (x) and true (y) timepoints t+1.
        if quadratic:
            ot_loss = quadratic_loss(
                x=pred_x,
                a=_a[t],
                y=_x[t + 1],
                b=_a[t + 1],
                space_x=_space[t],  # We keep the current spatial coordinates ...
                space_y=_space[t + 1],  # ... and compare them to the next coordinates.
                fused_penalty=fused_penalty,
                epsilon=epsilon,
                balancedness=balancedness,
                ott_solver=ott_solver,
                debias=debias,
            )
        else:
            ot_loss = linear_loss(
                x=pred_x,
                a=_a[t],
                y=_x[t + 1],
                b=_a[t + 1],
                cost_fn=cost_fn,
                epsilon=epsilon,
                balancedness=balancedness,
                ott_solver=ott_solver,
                debias=debias,
            )

        # If no teacher-forcing, replace next observation with predicted
        replace_fn = lambda u: u.at[t + 1].set(pred_x)
        _x = jax.lax.cond(teacher_forcing, lambda u: u, replace_fn, _x)

        # Do the same thing for spatial coordinates.
        replace_fn = lambda u: u.at[t + 1].set(_space[t])
        _space = jax.lax.cond(teacher_forcing, lambda u: u, replace_fn, _space)

        # And the same thing for the histogram.
        replace_fn = lambda u: u.at[t + 1].set(_a[t])
        _a = jax.lax.cond(teacher_forcing, lambda u: u, replace_fn, _a)

        # Return the data for the next iteration and the current loss.
        # Note the scaling by tau_diff[t] to give more weight to large time gaps.
        return (_x, _space, _a), ot_loss * tau_diff[t]

    # Iterate through timepoints efficiently. ot_loss becomes a 1-D array.
    # Notice that we do not compute the loss for the last timepoint, because there
    # is no next observation to compare to.
    timepoints = jnp.arange(len(batch_x) - 1)
    _, ot_loss = jax.lax.scan(
        _through_time, (batch_x, batch_space, batch_a), timepoints
    )

    # Sum the losses over all timepoints.
    return jnp.sum(ot_loss)
