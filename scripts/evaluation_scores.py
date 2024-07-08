# Matrix operations
from typing import Union
import numpy as np
import jax.numpy as jnp
from jax import Array
import jax
import einops

# AnnData
import anndata as ad

# OTT
from ott.solvers.linear.sinkhorn import Sinkhorn, SinkhornOutput
from ott.problems.linear.linear_problem import LinearProblem
from ott.geometry.pointcloud import PointCloud
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein, GWOutput

# Spacetime
from spacetime import scores


def sinkhorn(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_key: str,
    omics_key: str,
    space_key: str,
    max_cells: int = 6_000,
):
    """For each timepoint, compute the Sinkhorn distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_key].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Sinkhorn distance and divergence.
    cum_sinkhorn_dist, cum_sinkhorn_div = 0.0, 0.0
    cum_quad_sinkhorn_dist, cum_lin_sinkhorn_dist = 0.0, 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_key] == t
        idx &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx)[0],
            (min(max_cells, idx.sum()),),
            replace=False,
        )
        idx = np.full_like(idx, False)
        idx[subset] = True

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_key] == timepoints[i + 1]
        idx_true &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx_true)[0],
            (min(max_cells, idx_true.sum()),),
            replace=False,
        )
        idx_true = np.full_like(idx_true, False)
        idx_true[subset] = True

        # Define the joint geometry.
        geom_xy = PointCloud(
            adata.obsm["pred"][idx],
            adata.obsm[omics_key][idx_true],
            epsilon=0.1,
        )

        # Define the bias geometry on the ground-truth.
        geom_yy = PointCloud(
            adata.obsm[omics_key][idx_true],
            epsilon=0.1,
        )

        # Define the bias geometry on the prediction.
        geom_xx = PointCloud(
            adata.obsm["pred"][idx],
            epsilon=0.1,
        ).copy_epsilon(geom_yy)

        # Define the spatial geometry on the prediction and ground-truth
        geom_s_xx = PointCloud(adata.obsm[space_key][idx])
        geom_s_yy = PointCloud(adata.obsm[space_key][idx_true])

        # Compute the Sinkhorn distance.
        problem = LinearProblem(geom_xy)
        solver = Sinkhorn()
        out = solver(problem)

        # If Sinkhorn converged, add the distance to the cumulative distance.
        assert out.converged
        sinkhorn_dist = float(out.reg_ot_cost)
        cum_sinkhorn_dist += t_diff[i] * sinkhorn_dist
        # ...and the quadratic component...
        cum_quad_sinkhorn_dist += t_diff[i] * float(
            compute_quad(geom_s_xx.x, geom_s_yy.y, out)
        )
        # ...and the linear component.
        cum_lin_sinkhorn_dist += t_diff[i] * float(
            compute_lin(geom_xy.x, geom_xy.y, out)
        )

        # Compute the Sinkhorn distance on the bias.
        problem_bias = LinearProblem(geom_xx)
        out_bias = solver(problem_bias)
        assert out_bias.converged
        sinkhorn_bias = float(out_bias.reg_ot_cost)

        # Compute the Sinkhorn distance on the bias.
        problem_bias = LinearProblem(geom_yy)
        out_bias = solver(problem_bias)
        assert out_bias.converged
        sinkhorn_bias += float(out_bias.reg_ot_cost)

        # Add the divergence to the cumulative divergence.
        cum_sinkhorn_div += t_diff[i] * (sinkhorn_dist - 0.5 * sinkhorn_bias)

    # Save and log the Sinkhorn distance.
    stats = {
        "timepoint": t,
        score_name: cum_sinkhorn_dist / t_diff.sum(),
        f"{score_name}_div": cum_sinkhorn_div / t_diff.sum(),
        f"quad_{score_name}": cum_quad_sinkhorn_dist / t_diff.sum(),
        f"lin_{score_name}": cum_lin_sinkhorn_dist / t_diff.sum(),
    }

    # Return stats and information useful for plotting.
    return stats, (idx, idx_true, out, timepoints)


def chamfer(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_key: str,
    omics_key: str,
    max_cells: int = 6_000,
):
    """For each timepoint, compute the Chamfer distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_key].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Chamfer distance.
    cum_chamfer_dist = 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_key] == t
        idx &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx)[0],
            (min(max_cells, idx.sum()),),
            replace=False,
        )
        idx = np.full_like(idx, False)
        idx[subset] = True

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_key] == timepoints[i + 1]
        idx_true &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx_true)[0],
            (min(max_cells, idx_true.sum()),),
            replace=False,
        )
        idx_true = np.full_like(idx_true, False)
        idx_true[subset] = True

        # Compute the Chamfer distance.
        chamfer_dist = scores.chamfer_distance(
            jnp.array(adata[idx_true].obsm[omics_key]),
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
    time_key: str,
    omics_key: str,
    max_cells: int = 6_000,
):
    """For each timepoint, compute the Hausdorff distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_key].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Initialize the cumulative Hausdorff distance.
    cum_hausdorff_dist = 0.0

    # Iterate over timepoints.
    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_key] == t
        idx &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx)[0],
            (min(max_cells, idx.sum()),),
            replace=False,
        )
        idx = np.full_like(idx, False)
        idx[subset] = True

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_key] == timepoints[i + 1]
        idx_true &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx_true)[0],
            (min(max_cells, idx_true.sum()),),
            replace=False,
        )
        idx_true = np.full_like(idx_true, False)
        idx_true[subset] = True

        # Compute the Hausdorff distance.
        hausdorff_dist = scores.hausdorff_distance(
            jnp.array(adata[idx_true].obsm[omics_key]),
            jnp.array(adata[idx].obsm["pred"]),
        )

        # Add the distance to the cumulative distance.
        cum_hausdorff_dist += t_diff[i] * float(hausdorff_dist)

    # Save and log the Hausdorff distance.
    return {"timepoint": t, score_name: cum_hausdorff_dist / t_diff.sum()}


def compute_quad(x: Array, y: Array, out: Union[SinkhornOutput, GWOutput]):
    """Compute the quadratic component.

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


def compute_lin(x: Array, y: Array, out: Union[SinkhornOutput, GWOutput]):
    """Compute the linear component.

    In the following, we compute sum_ij C_ij P_ij."""

    # Convert the xy geometry to low rank, ie C = A @ B.T
    # This way, we do not have to materialize the matrix C.
    geom_xy_lr = PointCloud(x, y).to_LRCGeometry()
    A, B = geom_xy_lr.cost_1, geom_xy_lr.cost_2

    # The trick out.apply(A.T) = A.T @ out.matrix allows to
    # not materialize the matrix out.matrix.
    lin_cost = einops.einsum(B, out.apply(A.T), "j p, p j ->")

    # Return the sum of the three terms.
    return lin_cost


def fgw(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    score_name: str,
    time_key: str,
    space_key: str,
    omics_key: str,
    max_cells: int = 6_000,
):
    """For each timepoint, compute the FGW distance."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_key].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Define relative quadratic weight and epsilon
    quadratic_weight = 1e-3
    epsilon = 1e-3

    # Initialize the cumulative FGW distance and divergence.
    cum_lin_dist, cum_quad_dist, cum_fgw_dist, cum_fgw_div = 0.0, 0.0, 0.0, 0.0

    for i, t in enumerate(timepoints[:-1]):

        # Get indices for the prediction.
        idx = adata.obs[time_key] == t
        idx &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx)[0],
            (min(max_cells, idx.sum()),),
            replace=False,
        )
        idx = np.full_like(idx, False)
        idx[subset] = True

        # Get indices for the ground-truth.
        idx_true = adata.obs[time_key] == timepoints[i + 1]
        idx_true &= idx_batches
        subset = jax.random.choice(
            jax.random.PRNGKey(0),
            np.where(idx_true)[0],
            (min(max_cells, idx_true.sum()),),
            replace=False,
        )
        idx_true = np.full_like(idx_true, False)
        idx_true[subset] = True

        # Define the spatial geometry on the prediction and ground-truth
        geom_s_xx = PointCloud(
            adata.obsm[space_key][idx],
            scale_cost=(1 / np.sqrt(quadratic_weight)),
        )
        geom_s_yy = PointCloud(
            adata.obsm[space_key][idx_true],
            scale_cost=(1 / np.sqrt(quadratic_weight)),
        )

        # Define the joint gene geometry on the prediction and ground-truth.
        geom_xy = PointCloud(
            adata.obsm["pred"][idx],
            adata.obsm[omics_key][idx_true],
            scale_cost=(1 / (1 - quadratic_weight)),
        )
        geom_xx = PointCloud(
            adata.obsm["pred"][idx],
            scale_cost=(1 / (1 - quadratic_weight)),
        )
        geom_yy = PointCloud(
            adata.obsm[omics_key][idx_true],
            scale_cost=(1 / (1 - quadratic_weight)),
        )

        # Compute the FGW distance between point clouds x and y.
        problem = QuadraticProblem(geom_s_xx, geom_s_yy, geom_xy, fused_penalty=1.0)
        solver = GromovWasserstein(epsilon=epsilon)
        out = solver(problem)
        assert out.converged

        # Compute the ground-truth bias
        problem = QuadraticProblem(geom_s_yy, geom_s_yy, geom_yy, fused_penalty=1.0)
        solver = GromovWasserstein(epsilon=epsilon)
        out_yy = solver(problem)
        assert out_yy.converged
        bias = float(out_yy.reg_gw_cost)

        # Then compute the prediction bias.
        problem = QuadraticProblem(geom_s_xx, geom_s_xx, geom_xx, fused_penalty=1.0)
        solver = GromovWasserstein(epsilon=epsilon)
        out_xx = solver(problem)
        assert out_xx.converged
        bias += float(out_xx.reg_gw_cost)

        # Add the distance to the cumulative distance.
        cum_fgw_dist += t_diff[i] * float(out.reg_gw_cost)
        # Also compute the divergence...
        cum_fgw_div += t_diff[i] * (float(out.reg_gw_cost) - 0.5 * bias)
        # ...and the quadratic component...
        cum_quad_dist += t_diff[i] * float(compute_quad(geom_s_xx.x, geom_s_yy.y, out))
        # ...and the linear component.
        cum_lin_dist += t_diff[i] * float(compute_lin(geom_xy.x, geom_xy.y, out))

    # Save and log the FGW distance.
    stats = {
        "timepoint": t,
        score_name: cum_fgw_dist / t_diff.sum(),
        f"{score_name}_div": cum_fgw_div / t_diff.sum(),
        f"quad_{score_name}": cum_quad_dist / t_diff.sum(),
        f"lin_{score_name}": cum_lin_dist / t_diff.sum(),
    }

    # Return stats and information useful for plotting.
    return stats, (idx, idx_true, out, timepoints)
