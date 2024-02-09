# Imports
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Sequence
import traceback
import sys


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    try:
        import pickle

        import jax
        import matplotlib.pyplot as plt
        import numpy as np
        import jax.numpy as jnp
        import seaborn as sns
        import wandb
        from flatten_dict import flatten

        from ott.geometry.pointcloud import PointCloud
        from ott.problems.linear.linear_problem import LinearProblem
        from ott.problems.quadratic.quadratic_problem import QuadraticProblem
        from matplotlib.patches import ConnectionPatch
        from ott.solvers.linear.sinkhorn import Sinkhorn
        from ott.solvers.quadratic.gromov_wasserstein_lr import LRGromovWasserstein
        from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
        from spacetime.scores import knn_classify, hausdorff_distance, chamfer_distance

        ################################ Get the configuration ###############################

        # Get the config file of the run to evaluate.
        eval_cfg = OmegaConf.load(f"{cfg.checkpoint_path}/config.yaml")

        # Initialize a dict with scores to save.
        scores_dict = {}

        # Initialize Weights & Biases.
        config = flatten(OmegaConf.to_container(eval_cfg, resolve=True), reducer="dot")
        wandb.init(
            project="evaluate_multi_spacetime", config=config, mode=cfg.wandb.mode
        )
        print(config, f"JAX device type: {jax.devices()[0].device_kind}")

        # Get some parameters.
        x_obsm = eval_cfg.organism.obsm
        space_obsm = eval_cfg.organism.space_obsm
        time_obs = eval_cfg.organism.time_obs
        annotation_obs = eval_cfg.organism.annotation_obs

        ############################### Load the data ####################################

        adata = _load_data(
            dataset_path=eval_cfg.organism.dataset_path,
            x_obsm=x_obsm,
            space_obsm=space_obsm,
            n_pcs=eval_cfg.n_pcs,
        )

        ############################## Define the step and model #########################

        my_model = _define_model(
            step_type=eval_cfg.step.type,
            implicit_diff=eval_cfg.step.implicit_diff,
            max_iter=eval_cfg.step.maxiter,
            features=eval_cfg.potential.features,
            teacher_forcing=eval_cfg.model.teacher_forcing,
            quadratic=eval_cfg.model.quadratic,
            epsilon=eval_cfg.model.epsilon,
            fused_penalty=eval_cfg.model.fused,
            save_interval_steps=eval_cfg.optimizer.checkpoint_interval,
            checkpoint_path=cfg.checkpoint_path,
        )

        ################################# Transform the data #############################

        # Initialize the prediction.
        adata.obsm["pred"] = adata.obsm[x_obsm].copy()
        adata.obs[time_obs] = adata.obs[time_obs].astype(float)

        # Transform the data given a subet of batches.
        def _pred(idx_batches):
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            t_diff = np.diff(timepoints).astype(float)
            for i, t in enumerate(timepoints[:-1]):
                idx = adata.obs[time_obs] == t
                idx &= idx_batches
                adata.obsm["pred"][idx] = my_model.transform(
                    adata[idx], x_obsm=x_obsm, batch_size=500, tau=t_diff[i]
                )

        # Transform the data on training batches.
        idx_train = np.isin(adata.obs["Batch"], eval_cfg.organism.train_batches)
        _pred(idx_train)

        # Transform the data on early test batches.
        idx_early_test = np.isin(
            adata.obs["Batch"], eval_cfg.organism.early_test_batches
        )
        _pred(idx_early_test)

        # Transform the data on late test batches.
        idx_late_test = np.isin(adata.obs["Batch"], eval_cfg.organism.late_test_batches)
        _pred(idx_late_test)

        ############################ Compute predicted labels ############################

        # Initialize the prediction.
        adata.obs["label_pred"] = adata.obs[annotation_obs].copy()

        # Predict the labels given a subset of batches.
        def _pred_label(idx_batches):
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            for i in range(len(timepoints) - 1):

                idx = adata.obs[time_obs] == timepoints[i + 1]
                x_real = adata.obsm[x_obsm][idx & idx_batches, :]
                labels_real = adata.obs[annotation_obs][idx & idx_batches]

                idx = adata.obs[time_obs] == timepoints[i]
                x_pred = adata.obsm["pred"][idx & idx_batches, :]

                adata.obs["label_pred"][idx & idx_batches] = knn_classify(
                    x_real, labels_real, x_pred
                )

        # Predict the labels on training batches.
        _pred_label(idx_train)

        # Predict the labels on early test batches.
        _pred_label(idx_early_test)

        # Predict the labels on late test batches.
        _pred_label(idx_late_test)

        #################### Compute sinkhorn distances for each label ###################

        def _sinkhorn(idx_batches, score_name):
            # For each timepoint, compute the Sinkhorn distance.
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            t_diff = np.diff(timepoints).astype(float)
            cum_sinkhorn_dist, cum_sinkhorn_bias = 0.0, 0.0
            for i, t in enumerate(timepoints[:-1]):
                # Get indices for the prediction and the ground-truth.
                idx = adata.obs[time_obs] == t
                idx &= idx_batches

                idx_true = adata.obs[time_obs] == timepoints[i + 1]
                idx_true &= idx_batches

                # Define the geometry.
                geom_yy = PointCloud(
                    adata.obsm[x_obsm][idx_true],
                    epsilon=0.1,
                    batch_size=512,
                )
                geom_xx = PointCloud(adata.obsm["pred"][idx], batch_size=512)
                geom_xx = geom_xx.copy_epsilon(geom_yy)
                geom_xy = PointCloud(
                    adata.obsm["pred"][idx],
                    adata.obsm[x_obsm][idx_true],
                    batch_size=512,
                ).copy_epsilon(geom_yy)

                # Compute the Sinkhorn distance.
                problem = LinearProblem(geom_xy)
                solver = Sinkhorn(inner_iterations=100, max_iterations=500_000)
                out = solver(problem)
                assert out.converged
                sinkhorn_dist = float(out.reg_ot_cost)
                cum_sinkhorn_dist += t_diff[i] * sinkhorn_dist

                # Compute the Sinkhorn distance.
                problem_bias = LinearProblem(geom_xx)
                out_bias = solver(problem_bias)
                assert out_bias.converged
                sinkhorn_bias = float(out_bias.reg_ot_cost)

                # Compute the Sinkhorn distance.
                problem_bias = LinearProblem(geom_yy)
                out_bias = solver(problem_bias)
                assert out_bias.converged
                sinkhorn_bias += float(out_bias.reg_ot_cost)

                cum_sinkhorn_bias += t_diff[i] * sinkhorn_bias

            # Save and log the Sinkhorn distance.
            stats = {
                "timepoint": t,
                score_name: cum_sinkhorn_dist / t_diff.sum(),
                f"{score_name}_div": (cum_sinkhorn_dist - 0.5 * cum_sinkhorn_bias)
                / t_diff.sum(),
            }
            wandb.log(stats)
            scores_dict[score_name] = stats

            return idx, idx_true, out, timepoints

        # Compute the Sinkhorn distance for each label.
        for label in np.unique(adata[idx_early_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_early_test].obs[time_obs].nunique() > 1:
                _sinkhorn(idx_label & idx_early_test, f"sinkhorn_early_{label}")

        # Compute the Sinkhorn distance for each label.
        for label in np.unique(adata[idx_late_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_late_test].obs[time_obs].nunique() > 1:
                _sinkhorn(idx_label & idx_late_test, f"sinkhorn_late_{label}")

        #################### Compute chamfer distances for each label ####################

        def _chamfer(idx_batches, score_name):
            # For each timepoint, compute the Chamfer distance.
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            t_diff = np.diff(timepoints).astype(float)
            cum_chamfer_dist = 0.0
            for i, t in enumerate(timepoints[:-1]):
                # Get indices for the prediction and the ground-truth.
                idx = adata.obs[time_obs] == t
                idx &= idx_batches

                idx_true = adata.obs[time_obs] == timepoints[i + 1]
                idx_true &= idx_batches

                # Compute the Chamfer distance.
                chamfer_dist = chamfer_distance(
                    jnp.array(adata[idx_true].obsm[x_obsm]),
                    jnp.array(adata[idx].obsm["pred"]),
                )
                cum_chamfer_dist += t_diff[i] * float(chamfer_dist)

            # Save and log the Chamfer distance.
            stats = {"timepoint": t, score_name: cum_chamfer_dist / t_diff.sum()}
            wandb.log(stats)
            scores_dict[score_name] = stats

        # Compute the Chamfer distance for each label.
        for label in np.unique(adata[idx_early_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_early_test].obs[time_obs].nunique() > 1:
                _chamfer(idx_label & idx_early_test, f"chamfer_early_{label}")

        # Compute the Chamfer distance for each label.
        for label in np.unique(adata[idx_late_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_late_test].obs[time_obs].nunique() > 1:
                _chamfer(idx_label & idx_late_test, f"chamfer_late_{label}")

        #################### Compute Hausdorff distances for each label ##################

        # For each label, compute the Hausdorff distance.
        def _hausdorff(idx_batches, score_name):
            # For each timepoint, compute the Hausdorff distance.
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            t_diff = np.diff(timepoints).astype(float)
            cum_hausdorff_dist = 0.0
            for i, t in enumerate(timepoints[:-1]):
                # Get indices for the prediction and the ground-truth.
                idx = adata.obs[time_obs] == t
                idx &= idx_batches

                idx_true = adata.obs[time_obs] == timepoints[i + 1]
                idx_true &= idx_batches

                # Compute the Hausdorff distance.
                hausdorff_dist = hausdorff_distance(
                    jnp.array(adata[idx_true].obsm[x_obsm]),
                    jnp.array(adata[idx].obsm["pred"]),
                )
                cum_hausdorff_dist += t_diff[i] * float(hausdorff_dist)

            # Save and log the Hausdorff distance.
            stats = {"timepoint": t, score_name: cum_hausdorff_dist / t_diff.sum()}
            wandb.log(stats)
            scores_dict[score_name] = stats

        # Compute the Hausdorff distance for each label.
        for label in np.unique(adata[idx_early_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_early_test].obs[time_obs].nunique() > 1:
                _hausdorff(idx_label & idx_early_test, f"hausdorff_early_{label}")

        # Compute the Hausdorff distance for each label.
        for label in np.unique(adata[idx_late_test].obs["label_pred"]):
            idx_label = adata.obs["label_pred"] == label
            if adata[idx_label & idx_late_test].obs[time_obs].nunique() > 1:
                _hausdorff(idx_label & idx_late_test, f"hausdorff_late_{label}")

        wandb.finish()

        # Save scores_dict as a pickle file
        with open(f"{cfg.checkpoint_path}/scores_multi.pkl", "wb") as f:
            pickle.dump(scores_dict, f)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def _load_data(
    dataset_path: str,
    x_obsm: str,
    space_obsm: str,
    n_pcs: int,
):
    import anndata as ad
    import numpy as np

    # Load the data.
    adata = ad.read_h5ad(dataset_path)

    # Normalize the obsm.
    adata.obsm[x_obsm] = adata.obsm[x_obsm][:, :n_pcs]
    adata.obsm[x_obsm] /= adata.obsm[x_obsm].max()

    # Center and scale each timepoint in space.
    adata.obsm[space_obsm] = adata.obsm[space_obsm].astype(float)
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b

        mu = np.mean(adata.obsm[space_obsm][idx, :], axis=0)
        adata.obsm[space_obsm][idx, :] -= mu

        sigma = np.std(adata.obsm[space_obsm][idx, :], axis=0)
        adata.obsm[space_obsm][idx, :] /= sigma

    return adata


def _define_model(
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
    from flax.linen.activation import gelu
    import wandb
    import orbax.checkpoint
    import spacetime
    from spacetime import potentials, steps

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


if __name__ == "__main__":
    main()
