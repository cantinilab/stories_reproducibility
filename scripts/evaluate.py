# Imports
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Sequence
import traceback
import sys

# from jax import config
# config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    try:
        import pickle

        import jax
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
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
        from spacetime import scores

        ################################ Get the configuration ###############################

        # Get the config file of the run to evaluate.
        eval_cfg = OmegaConf.load(f"{cfg.checkpoint_path}/config.yaml")

        # Initialize a dict with scores to save.
        scores_dict = {}

        # Initialize Weights & Biases.
        config = flatten(OmegaConf.to_container(eval_cfg, resolve=True), reducer="dot")
        wandb.init(project="evaluate_spacetime", config=config, mode=cfg.wandb.mode)
        print(config, f"JAX device type: {jax.devices()[0].device_kind}")

        # Get some parameters.
        x_obsm = eval_cfg.organism.obsm
        space_obsm = eval_cfg.organism.space_obsm
        time_obs = eval_cfg.organism.time_obs
        annotation_obs = eval_cfg.organism.annotation_obs

        ############################### Load the data ########################################

        adata = _load_data(
            dataset_path=eval_cfg.organism.dataset_path,
            x_obsm=x_obsm,
            space_obsm=space_obsm,
            n_pcs=eval_cfg.n_pcs,
        )

        ############################## Define the step and model #############################

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

        ################################# Transform the data #################################

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

        ############################ Compute the Sinkhorn distance ###########################

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
                solver = Sinkhorn(inner_iterations=100, max_iterations=10_000)
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

        _sinkhorn(idx_train, "sinkhorn_train")
        _sinkhorn(idx_early_test, "sinkhorn_early_test")
        idx_last, idx_true_last, out_last, timepoints_last = _sinkhorn(
            idx_late_test, "sinkhorn_late_test"
        )

        ############################# Plot the last Sinkhorn plan ############################

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

        dot_size = 5e3 / adata[idx_last].n_obs
        sns.scatterplot(
            x=adata.obsm[space_obsm][idx_last, 0],
            y=adata.obsm[space_obsm][idx_last, 1],
            hue=adata[idx_last].obs[annotation_obs],
            ax=axes[0],
            s=dot_size,
        )
        axes[0].set_title(f"Prediction from {timepoints_last[-2]}", fontsize=9)
        axes[0].set_frame_on(False)
        axes[0].get_legend().remove()

        dot_size = 5e3 / adata[idx_true_last].n_obs
        sns.scatterplot(
            x=adata.obsm[space_obsm][idx_true_last, 0],
            y=adata.obsm[space_obsm][idx_true_last, 1],
            hue=adata[idx_true_last].obs[annotation_obs],
            ax=axes[1],
            s=dot_size,
        )
        axes[1].set_title(f"Real timepoint {timepoints_last[-1]}", fontsize=9)
        axes[1].set_frame_on(False)
        axes[1].get_legend().remove()

        # Reproducible random points.
        key = jax.random.PRNGKey(0)
        random_j = jax.random.choice(
            key, adata[idx_last].n_obs, shape=(10,), replace=False
        )
        random_j = np.array(random_j)

        for j in random_j:
            indicator = np.zeros(adata[idx_last].n_obs)
            indicator[j] = 1
            push = out_last.apply(indicator)

            max_Pj = np.max(push)
            for k in np.argpartition(push, -5)[-5:]:
                con = ConnectionPatch(
                    xyA=(
                        adata[idx_last].obsm[space_obsm][j, 0],
                        adata[idx_last].obsm[space_obsm][j, 1],
                    ),
                    coordsA=axes[0].transData,
                    xyB=(
                        adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 0],
                        adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 1],
                    ),
                    coordsB=axes[1].transData,
                    alpha=0.5 * min(float(push[k] / max_Pj), 1.0),
                )

                fig.add_artist(con)

        # Log the plot.
        image = wandb.Image(plt)
        wandb.log({"Sinkhorn plan": image})
        plt.close("all")

        ########## Compute the difference between real and kNN predicted histograms. #########

        # def _l1(idx_batches, score_name):
        #     # For each timepoint, compute the L1 distance.
        #     timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
        #     t_diff = np.diff(timepoints).astype(float)
        #     cum_l1_dist = 0.0
        #     for i, t in enumerate(timepoints[:-1]):
        #         # Get indices for the prediction and the ground-truth.
        #         idx = adata.obs[time_obs] == t
        #         idx &= idx_batches

        #         idx_true = adata.obs[time_obs] == timepoints[i + 1]
        #         idx_true &= idx_batches

        #         # Compute the L1 distance.
        #         labels_real = adata[idx_true].obs[annotation_obs]
        #         l1_dist, labels_pred = scores.compare_real_and_knn_histograms(
        #             x_real=adata[idx_true].obsm[x_obsm],
        #             labels_real=labels_real,
        #             x_pred=adata[idx].obsm["pred"],
        #             k=5,
        #         )
        #         cum_l1_dist += t_diff[i] * float(l1_dist)

        #     # Save and log the L1 distance.
        #     stats = {"timepoint": t, score_name: cum_l1_dist / t_diff.sum()}
        #     wandb.log(stats)
        #     scores_dict[score_name] = stats

        #     return labels_real, labels_pred

        # labels_real, labels_pred = _l1(idx_train, "train_L1")
        # labels_real, labels_pred = _l1(idx_early_test, "early_test_L1")
        # labels_real, labels_pred = _l1(idx_late_test, "late_test_L1")

        # # plot the last histogram
        # histo_list = []
        # for label in np.unique(labels_real):
        #     histo_list.append(
        #         {
        #             "cell type": label,
        #             "prop": np.mean(labels_real == label),
        #             "type": "real",
        #         }
        #     )
        #     histo_list.append(
        #         {
        #             "cell type": label,
        #             "prop": np.mean(labels_pred == label),
        #             "type": "pred",
        #         }
        #     )

        # scores_dict["histo_list"] = histo_list

        # histo_df = pd.DataFrame(histo_list)
        # sns.barplot(data=histo_df, y="cell type", x="prop", hue="type")

        # # Log the plot.
        # image = wandb.Image(plt)
        # wandb.log({"Histograms": image})
        # plt.close("all")

        ############################ Compute the Hausdorff distance ########################

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
                hausdorff_dist = scores.hausdorff_distance(
                    jnp.array(adata[idx_true].obsm[x_obsm]),
                    jnp.array(adata[idx].obsm["pred"]),
                )
                cum_hausdorff_dist += t_diff[i] * float(hausdorff_dist)

            # Save and log the Hausdorff distance.
            stats = {"timepoint": t, score_name: cum_hausdorff_dist / t_diff.sum()}
            wandb.log(stats)
            scores_dict[score_name] = stats

        _hausdorff(idx_train, "hausdorff_train")
        _hausdorff(idx_early_test, "hausdorff_early_test")
        _hausdorff(idx_late_test, "hausdorff_late_test")

        ############################ Compute the Chamfer distance ############################

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
                chamfer_dist = scores.chamfer_distance(
                    jnp.array(adata[idx_true].obsm[x_obsm]),
                    jnp.array(adata[idx].obsm["pred"]),
                )
                cum_chamfer_dist += t_diff[i] * float(chamfer_dist)

            # Save and log the Chamfer distance.
            stats = {"timepoint": t, score_name: cum_chamfer_dist / t_diff.sum()}
            wandb.log(stats)
            scores_dict[score_name] = stats

        _chamfer(idx_train, "chamfer_train")
        _chamfer(idx_early_test, "chamfer_early_test")
        _chamfer(idx_late_test, "chamfer_late_test")

        #################### Compute the Fused Gromov-Wasserstein distance ###################

        def _fgw(idx_batches, score_name):
            # For each timepoint, compute the FGW distance.
            timepoints = np.sort(adata[idx_batches].obs[time_obs].unique())
            t_diff = np.diff(timepoints).astype(float)
            cum_fgw_dist = 0.0
            for i, t in enumerate(timepoints[:-1]):
                # Get indices for the prediction and the ground-truth.
                idx = adata.obs[time_obs] == t
                idx &= idx_batches

                idx_true = adata.obs[time_obs] == timepoints[i + 1]
                idx_true &= idx_batches

                # Define the geometries.
                geom_xx = PointCloud(adata.obsm[space_obsm][idx], batch_size=512)
                geom_yy = PointCloud(adata.obsm[space_obsm][idx_true], batch_size=512)
                geom_xy = PointCloud(
                    adata.obsm["pred"][idx],
                    adata.obsm[x_obsm][idx_true],
                    batch_size=512,
                )

                # Compute the FGW distance.
                problem = QuadraticProblem(
                    geom_xx, geom_yy, geom_xy, fused_penalty=10.0
                )
                solver = LRGromovWasserstein(
                    rank=500,
                    inner_iterations=100,
                    max_iterations=500_000,
                )
                out = solver(problem)
                print(out.converged)
                assert out.converged
                cum_fgw_dist += t_diff[i] * float(out.reg_gw_cost)

            # Save and log the FGW distance.
            stats = {"timepoint": t, score_name: cum_fgw_dist / t_diff.sum()}
            wandb.log(stats)
            scores_dict[score_name] = stats

            return idx, idx_true, out, timepoints

        _fgw(idx_train, "fgw_train")
        _fgw(idx_early_test, "fgw_early_test")
        idx_last, idx_true_last, out_last, timepoints_last = _fgw(
            idx_late_test, "fgw_late_test"
        )

        ############################# Plot the last FGW plan #################################

        fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

        dot_size = 5e3 / adata[idx_last].n_obs
        sns.scatterplot(
            x=adata.obsm[space_obsm][idx_last, 0],
            y=adata.obsm[space_obsm][idx_last, 1],
            hue=adata[idx_last].obs[annotation_obs],
            ax=axes[0],
            s=dot_size,
        )
        axes[0].set_title(f"Prediction from {timepoints_last[-2]}", fontsize=9)
        axes[0].set_frame_on(False)
        axes[0].get_legend().remove()

        dot_size = 5e3 / adata[idx_true_last].n_obs
        sns.scatterplot(
            x=adata.obsm[space_obsm][idx_true_last, 0],
            y=adata.obsm[space_obsm][idx_true_last, 1],
            hue=adata[idx_true_last].obs[annotation_obs],
            ax=axes[1],
            s=dot_size,
        )
        axes[1].set_title(f"Real timepoint {timepoints_last[-1]}", fontsize=9)
        axes[1].set_frame_on(False)
        axes[1].get_legend().remove()

        for j in random_j:
            indicator = np.zeros(adata[idx_last].n_obs)
            indicator[j] = 1
            push = out_last.apply(indicator)

            max_Pj = np.max(push)
            for k in np.argpartition(push, -5)[-5:]:
                con = ConnectionPatch(
                    xyA=(
                        adata[idx_last].obsm[space_obsm][j, 0],
                        adata[idx_last].obsm[space_obsm][j, 1],
                    ),
                    coordsA=axes[0].transData,
                    xyB=(
                        adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 0],
                        adata.obsm[space_obsm][np.where(idx_true_last)[0][k], 1],
                    ),
                    coordsB=axes[1].transData,
                    alpha=0.5 * min(float(push[k] / max_Pj), 1.0),
                )

                fig.add_artist(con)

        # Log the plot.
        image = wandb.Image(plt)
        wandb.log({"FGW plan": image})
        plt.close("all")

        wandb.finish()

        # Save scores_dict as a pickle file
        with open(f"{cfg.checkpoint_path}/scores.pkl", "wb") as f:
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
