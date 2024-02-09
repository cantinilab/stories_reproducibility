# Matrix operations
from typing import Union
import numpy as np
import jax.numpy as jnp
from jax import Array
import einops

# AnnData
import anndata as ad

# OTT
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.problems.linear.linear_problem import LinearProblem
from ott.geometry.pointcloud import PointCloud
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein, LRGWOutput
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein, GWOutput

# Spacetime
from spacetime import scores


def sinkhorn(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_obs: str,
    x_obsm: str,
):
    """For each timepoint, compute the Sinkhorn distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Sinkhorn distance and divergence.
    cum_sinkhorn_dist, cum_sinkhorn_div = 0.0, 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_obs] == t
        idx &= idx_batches

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_obs] == timepoints[i + 1]
        idx_true &= idx_batches

        # Define the bias geometry on the ground-truth.
        geom_yy = PointCloud(adata.obsm[x_obsm][idx_true], epsilon=0.1, batch_size=512)

        # Define the bias geometry on the prediction.
        geom_xx = PointCloud(adata.obsm["pred"][idx], batch_size=512)
        geom_xx = geom_xx.copy_epsilon(geom_yy)

        # Define the joint geometry.
        geom_xy = PointCloud(
            adata.obsm["pred"][idx], adata.obsm[x_obsm][idx_true], batch_size=512
        ).copy_epsilon(geom_yy)

        # Compute the Sinkhorn distance.
        problem = LinearProblem(geom_xy)
        solver = Sinkhorn(inner_iterations=100, max_iterations=10_000)
        out = solver(problem)

        # If Sinkhorn converged, add the distance to the cumulative distance.
        assert out.converged
        sinkhorn_dist = float(out.reg_ot_cost)
        cum_sinkhorn_dist += t_diff[i] * sinkhorn_dist

        # Compute the Sinkhorn distance.
        problem_bias = LinearProblem(geom_xx)
        out_bias = solver(problem_bias)

        # If Sinkhorn converged, save the bias.
        assert out_bias.converged
        sinkhorn_bias = float(out_bias.reg_ot_cost)

        # Compute the Sinkhorn distance.
        problem_bias = LinearProblem(geom_yy)
        out_bias = solver(problem_bias)

        # If Sinkhorn converged, add the divergence to the cumulative divergence.
        assert out_bias.converged
        sinkhorn_bias += float(out_bias.reg_ot_cost)
        cum_sinkhorn_div += t_diff[i] * (sinkhorn_dist - 0.5 * sinkhorn_bias)

    # Save and log the Sinkhorn distance.
    stats = {
        "timepoint": t,
        score_name: cum_sinkhorn_dist / t_diff.sum(),
        f"{score_name}_div": cum_sinkhorn_div / t_diff.sum(),
    }

    # Return stats and information useful for plotting.
    return stats, (idx, idx_true, out, timepoints)


def chamfer(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_obs: str,
    x_obsm: str,
):
    """For each timepoint, compute the Chamfer distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Chamfer distance.
    cum_chamfer_dist = 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_obs] == t
        idx &= idx_batches

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_obs] == timepoints[i + 1]
        idx_true &= idx_batches

        # Compute the Chamfer distance.
        chamfer_dist = scores.chamfer_distance(
            jnp.array(adata[idx_true].obsm[x_obsm]),
            jnp.array(adata[idx].obsm["pred"]),
        )

        # Add the distance to the cumulative distance.
        cum_chamfer_dist += t_diff[i] * float(chamfer_dist)

    # Save and log the Chamfer distance.
    return {"timepoint": t, score_name: cum_chamfer_dist / t_diff.sum()}


def hausdorff(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_obs: str,
    x_obsm: str,
):
    """For each timepoint, compute the Hausdorff distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Hausdorff distance.
    cum_hausdorff_dist = 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_obs] == t
        idx &= idx_batches

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_obs] == timepoints[i + 1]
        idx_true &= idx_batches

        # Compute the Hausdorff distance.
        hausdorff_dist = scores.hausdorff_distance(
            jnp.array(adata[idx_true].obsm[x_obsm]),
            jnp.array(adata[idx].obsm["pred"]),
        )

        # Add the distance to the cumulative distance.
        cum_hausdorff_dist += t_diff[i] * float(hausdorff_dist)

    # Save and log the Hausdorff distance.
    return {"timepoint": t, score_name: cum_hausdorff_dist / t_diff.sum()}


def compute_quad(x: Array, y: Array, out: Union[GWOutput, LRGWOutput]):
    """Compute the quadratic component of FGW.

    In the following, we compute sum_ijkl |C1_ij - C2_kl|^2 P_ik P_jl.
    This is equivalent to the sum of three terms:
    (A) sum_ijkl C1_ij^2 P_ik P_jl = sum_ij C1_ij^2 / n^2
    (B) sum_ijkl C2_kl^2 P_ik P_jl = sum_kl C2_kl^2 / m^2
    (C) - 2 sum_ijkl C1_ij C2_kl P_ik P_jl."""

    # Convert the xx geometry to low rank, ie C1 = A1 @ B1.T
    # This way, we do not have to materialize the matrix C1.
    geom_xx_lr = PointCloud(x).to_LRCGeometry()
    A1, B1 = geom_xx_lr.cost_1, geom_xx_lr.cost_2
    n = A1.shape[0]  # Number of points in the prediction.

    # Convert the yy geometry to low rank, ie C2 = A2 @ B2.T
    geom_yy_lr = PointCloud(y).to_LRCGeometry()
    A2, B2 = geom_yy_lr.cost_1, geom_yy_lr.cost_2
    m = A2.shape[0]  # Number of points in the ground-truth.

    # Equivalent to term (A).
    quad_cost = einops.einsum(A1 / n, B1, A1 / n, B1, "i p, j p, i q, j q ->")

    # Equivalent to term (B).
    quad_cost += einops.einsum(A2 / m, B2, A2 / m, B2, "i p, j p, i q, j q ->")

    # Equivalent to term (C). The trick out.apply(A1.T) = A1.T @ out.matrix allows to
    # not materialize the matrix out.matrix.
    quad_cost -= 2 * einops.einsum(
        A2, B2, out.apply(A1.T), out.apply(B1.T), "k q, l q, p k, p l ->"
    )

    # Return the sum of the three terms.
    return quad_cost


def fgw(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_obs: str,
    space_obsm: str,
    x_obsm: str,
    rank: int,
):
    """For each timepoint, compute the FGW distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative FGW distance and divergence.
    cum_gw_dist, cum_fgw_dist, cum_fgw_div = 0.0, 0.0, 0.0

    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_obs] == t
        idx &= idx_batches

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_obs] == timepoints[i + 1]
        idx_true &= idx_batches

        # Define the spatial geometry on the prediction and ground-truth
        geom_s_xx = PointCloud(adata.obsm[space_obsm][idx], batch_size=512)
        geom_s_yy = PointCloud(adata.obsm[space_obsm][idx_true], batch_size=512)

        # Define the joint gene geometry on the prediction and ground-truth.
        geom_xy = PointCloud(
            adata.obsm["pred"][idx], adata.obsm[x_obsm][idx_true], batch_size=512
        )
        geom_xx = PointCloud(adata.obsm["pred"][idx], batch_size=512)
        geom_yy = PointCloud(adata.obsm[x_obsm][idx_true], batch_size=512)

        # Define keyword arguments for the solver.
        solver_kwds = {"inner_iterations": 100, "max_iterations": 500_000}
        if rank == -1:
            solver_kwds["warm_start"] = True
        else:
            solver_kwds["rank"] = rank

        # Start with computing the ground-truth bias, which will detrmine the epsilon.
        problem = QuadraticProblem(geom_s_yy, geom_s_yy, geom_yy, fused_penalty=20.0)
        if rank == -1:
            solver = GromovWasserstein(
                relative_epsilon=True, epsilon=0.01, **solver_kwds
            )
        else:
            solver = LRGromovWasserstein(**solver_kwds)
        out_yy = solver(problem)

        # If FGW converged, add the distance to the bias.
        assert out_yy.converged
        bias = float(out_yy.reg_gw_cost)

        # Then compute the prediction bias.
        problem = QuadraticProblem(geom_s_xx, geom_s_xx, geom_xx, fused_penalty=20.0)
        if rank == -1:
            solver = GromovWasserstein(epsilon=out_yy.geom.epsilon, **solver_kwds)
        else:
            solver = LRGromovWasserstein(**solver_kwds)
        out_xx = solver(problem)

        # If FGW converged, add the distance to the bias.
        assert out_xx.converged
        bias += float(out_xx.reg_gw_cost)

        # Compute the FGW distance between point clouds x and y.
        problem = QuadraticProblem(geom_s_xx, geom_s_yy, geom_xy, fused_penalty=20.0)
        if rank == -1:
            solver = GromovWasserstein(epsilon=out_yy.geom.epsilon, **solver_kwds)
        else:
            solver = LRGromovWasserstein(**solver_kwds)
        out = solver(problem)

        # If FGW converged, add the distance to the cumulative distance.
        # Also compute the divergence and the quadratic component.
        assert out.converged
        cum_fgw_dist += t_diff[i] * float(out.reg_gw_cost)
        cum_fgw_div += t_diff[i] * (float(out.reg_gw_cost) - 0.5 * bias)
        cum_gw_dist += t_diff[i] * float(compute_quad(geom_s_xx.x, geom_s_yy.y, out))

    # Save and log the FGW distance.
    stats = {
        "timepoint": t,
        score_name: cum_fgw_dist / t_diff.sum(),
        f"{score_name}_div": cum_fgw_dist / t_diff.sum(),
        f"quad_{score_name}": cum_gw_dist / t_diff.sum(),
    }

    # Return stats and information useful for plotting.
    return stats, (idx, idx_true, out, timepoints)
