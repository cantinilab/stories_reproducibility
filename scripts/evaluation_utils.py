# Typing
from typing import Sequence, Union

# Matrix operations
import numpy as np

# AnnData
import anndata as ad

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch

# OTT
from ott.solvers.linear.sinkhorn import SinkhornOutput
from ott.solvers.quadratic.gromov_wasserstein_lr import LRGWOutput
from ott.solvers.quadratic.gromov_wasserstein import GWOutput

# Neural networks
from jax.nn import gelu
import orbax.checkpoint

# Logging, for implicit steps.
import wandb

# Spacetime
import spacetime
from spacetime import potentials, steps


def pred(
    adata: ad.AnnData,
    idx_batches: np.ndarray,
    time_obs: str,
    my_model: spacetime.SpaceTime,
    x_obsm: str,
):
    """Transform the data given a subet of batches."""

    # List timepoints and time differences between them.
    timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
    t_diff = np.diff(timepoints).astype(float)

    # Iterate over timepoints and transform the data.
    for i, t in enumerate(timepoints[:-1]):
        idx = (adata.obs[time_obs] == t) & idx_batches
        adata.obsm["pred"][idx] = my_model.transform(
            adata[idx], x_obsm=x_obsm, batch_size=500, tau=t_diff[i]
        )


def plot_plan(
    adata: ad.AnnData,
    idx_last: np.ndarray,
    idx_true_last: np.ndarray,
    space_obsm: str,
    annotation_obs: str,
    timepoints_last: np.ndarray,
    out_last: Union[SinkhornOutput, GWOutput, LRGWOutput],
    random_j: np.ndarray,
):
    """Plot the plan between the last prediction and the ground-truth."""

    # Initialize the plot.
    fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

    # Plot the last prediction as a scatterplot, colored by annotation.
    dot_size = 5e3 / adata[idx_last].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx_last, 0],
        y=adata.obsm[space_obsm][idx_last, 1],
        hue=adata[idx_last].obs[annotation_obs],
        ax=axes[0],
        s=dot_size,
    )

    # Decorate the scatterplot.
    axes[0].set_title(f"Prediction from {timepoints_last[-2]}", fontsize=9)
    axes[0].set_frame_on(False)
    axes[0].get_legend().remove()

    # Plot the last ground-truth as a scatterplot, colored by annotation.
    dot_size = 5e3 / adata[idx_true_last].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx_true_last, 0],
        y=adata.obsm[space_obsm][idx_true_last, 1],
        hue=adata[idx_true_last].obs[annotation_obs],
        ax=axes[1],
        s=dot_size,
    )

    # Decorate the scatterplot.
    axes[1].set_title(f"Real timepoint {timepoints_last[-1]}", fontsize=9)
    axes[1].set_frame_on(False)
    axes[1].get_legend().remove()

    # Plot the Sinkhorn plan between the last prediction and the ground-truth.
    # For that, iterate over random points and plot the connections.
    for j in random_j:

        # Apply the transport plan to the cell j.
        indicator = np.zeros(adata[idx_last].n_obs)
        indicator[j] = 1
        push = out_last.apply(indicator)
        max_Pj = np.max(push)  # For normalization.

        # Plot the top 5 connections. Argpartition is faster than argsort.
        for k in np.argpartition(push, -5)[-5:]:

            # Coordinates in the first plot.
            xA = adata[idx_last].obsm[space_obsm][j, 0]
            yA = adata[idx_last].obsm[space_obsm][j, 1]

            # Coordinates in the second plot.
            xB = adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 0]
            yB = adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 1]

            # Create the connection.
            con = ConnectionPatch(
                xyA=(xA, yA),
                coordsA=axes[0].transData,
                xyB=(xB, yB),
                coordsB=axes[1].transData,
                alpha=0.5 * min(float(push[k] / max_Pj), 1.0),
            )

            # Add the connection to the plot.
            fig.add_artist(con)

    # Return the figure and the axes.
    return fig, axes


def load_data(
    dataset_path: str,
    x_obsm: str,
    space_obsm: str,
    n_pcs: int,
):
    """Load the data and preprocess it."""

    # Load the data.
    adata = ad.read_h5ad(dataset_path)

    # Normalize the obsm.
    adata.obsm[x_obsm] = adata.obsm[x_obsm][:, :n_pcs]
    adata.obsm[x_obsm] /= adata.obsm[x_obsm].max()

    # Center and scale each timepoint in space.
    adata.obsm[space_obsm] = adata.obsm[space_obsm].astype(float)
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b

        # Center
        mu = np.mean(adata.obsm[space_obsm][idx, :], axis=0)
        adata.obsm[space_obsm][idx, :] -= mu

        # Scale
        sigma = np.std(adata.obsm[space_obsm][idx, :], axis=0)
        adata.obsm[space_obsm][idx, :] /= sigma

    return adata


def define_model(
    step_type: str,
    implicit_diff: bool,
    max_iter: int,
    features: Sequence[int],
    teacher_forcing: bool,
    quadratic: bool,
    epsilon: float,
    fused_penalty: float,
    save_interval_steps: int,
    checkpoint_path: str,
):
    """Define the model, given a configuration."""

    # Intialize keyword arguments for the proximal step.
    step_kwargs = {}

    # If the proximal step is implicit, add the appropriate keyword arguments.
    if "implicit" in step_type:
        step_kwargs["implicit_diff"] = implicit_diff
        step_kwargs["maxiter"] = max_iter
        step_kwargs["log_callback"] = lambda x: wandb.log(x)

    # Choose the proximal step.
    if step_type == "explicit":
        step = steps.ExplicitStep()
    elif step_type == "monge_implicit":
        step = steps.MongeImplicitStep(**step_kwargs)
    elif step_type == "icnn_implicit":
        step = steps.ICNNImplicitStep(**step_kwargs)
    else:
        raise ValueError(f"Step {step_type} not recognized.")

    # Initialize the model.
    my_model = spacetime.SpaceTime(
        potential=potentials.MLPPotential(features, activation=gelu),
        proximal_step=step,
        teacher_forcing=teacher_forcing,
        quadratic=quadratic,
        epsilon=epsilon,
        log_callback=lambda x: wandb.log(x),
        fused_penalty=fused_penalty,
    )

    # Define the checkpoint manager.
    options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,
        max_to_keep=1,
        best_fn=lambda x: x["loss"],
        best_mode="min",
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        f"{checkpoint_path}/checkpoints",
        orbax.checkpoint.PyTreeCheckpointer(),
        options=options,
    )

    # Restore the model.
    best_epoch = checkpoint_manager.best_step()
    my_model.params = checkpoint_manager.restore(best_epoch)

    return my_model
