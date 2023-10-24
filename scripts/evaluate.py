# Imports
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    import pickle

    import anndata as ad
    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    import orbax.checkpoint
    import pandas as pd
    import scanpy as sc
    import seaborn as sns
    import wandb
    from flatten_dict import flatten
    from matplotlib.patches import ConnectionPatch

    import spacetime
    from ott.geometry.pointcloud import PointCloud
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.problems.quadratic.quadratic_problem import QuadraticProblem
    from ott.solvers.linear.sinkhorn import Sinkhorn
    from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
    from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
    from spacetime import potentials, scores, steps

    # Get the config file of the run to evaluate.
    eval_cfg = OmegaConf.load(f"{cfg.checkpoint_path}/config.yaml")

    # Initialize a dict with scores to save.
    scores_dict = {}

    # Initialize Weights & Biases.
    config = flatten(OmegaConf.to_container(eval_cfg, resolve=True), reducer="dot")
    wandb.init(project="evaluate_spacetime", config=config, mode=cfg.wandb.mode)
    print(config, f"JAX device type: {jax.devices()[0].device_kind}")

    # Load the data.
    adata = ad.read_h5ad(eval_cfg.organism.dataset_path)

    # Normalize data.
    adata.obsm[eval_cfg.organism.obsm] /= adata.obsm[eval_cfg.organism.obsm].max()

    # Center the space.
    timepoints = np.sort(np.unique(adata.obs[eval_cfg.organism.time_obs]))
    for timepoint in timepoints:
        idx = adata.obs[eval_cfg.organism.time_obs] == timepoint
        mean_space = adata.obsm[eval_cfg.organism.space_obsm][idx].mean(axis=0)
        adata.obsm[eval_cfg.organism.space_obsm][idx] -= mean_space.reshape(1, 2)

    ############################ Plot the original timepoints. ###########################

    fig, axes = plt.subplots(
        1, len(timepoints), figsize=(10, 3), constrained_layout=True
    )

    for i, timepoint in enumerate(timepoints):
        idx = adata.obs[eval_cfg.organism.time_obs] == timepoint
        dot_size = 5e4 / adata[idx].n_obs
        sc.pl.embedding(
            adata[idx],
            basis=eval_cfg.organism.space_obsm,
            color=eval_cfg.organism.annotation_obs,
            ax=axes[i],
            show=False,
            frameon=False,
            s=dot_size,
            palette=sc.pl.palettes.default_28,
        )
        axes[i].get_legend().remove()
        axes[i].set_title(f"Timepoint {timepoint}", fontsize=9)

    # Log the plot.
    image = wandb.Image(plt)
    wandb.log({"original timepoints": image})
    plt.close("all")

    ############################## Define the step and model #############################

    # Intialize keyword arguments for the proximal step.
    step_kwargs = {}

    # If the proximal step is quadratic, add the appropriate keyword arguments.
    if "quadratic" in eval_cfg.step.type:
        step_kwargs["fused"] = eval_cfg.step.fused
        step_kwargs["cross"] = eval_cfg.step.cross
        step_kwargs["straight"] = eval_cfg.step.straight

    # If the proximal step is implicit, add the appropriate keyword arguments.
    if "implicit" in eval_cfg.step.type:
        step_kwargs["implicit_diff"] = eval_cfg.step.implicit_diff
        step_kwargs["maxiter"] = eval_cfg.step.maxiter
        step_kwargs["log_callback"] = lambda x: wandb.log(x)

    # Choose the proximal step.
    if eval_cfg.step.type == "linear_explicit":
        step = steps.LinearExplicitStep()
    elif eval_cfg.step.type == "quadratic_explicit":
        step = steps.QuadraticExplicitStep(**step_kwargs)
    elif eval_cfg.step.type == "monge_linear_implicit":
        step = steps.MongeLinearImplicitStep(**step_kwargs)
    elif eval_cfg.step.type == "monge_quadratic_implicit":
        step = steps.MongeQuadraticImplicitStep(**step_kwargs)
    else:
        raise ValueError(f"Step {eval_cfg.step.type} not recognized.")

    # Initialize the model.
    my_model = spacetime.SpaceTime(
        potential=potentials.MLPPotential(eval_cfg.potential.features),
        proximal_step=step,
        tau=eval_cfg.model.tau,
        quadratic=eval_cfg.model.quadratic,
        epsilon=eval_cfg.model.epsilon,
        teacher_forcing=eval_cfg.model.teacher_forcing,
        log_callback=lambda x: wandb.log(x),
        fused_penalty=eval_cfg.model.fused,
    )

    # Define the checkpoint manager.
    options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=eval_cfg.optimizer.checkpoint_interval,
        max_to_keep=1,
        best_fn=lambda x: x["loss"],
        best_mode="min",
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        f"{eval_cfg.checkpoint_path}_{eval_cfg.model.seed}/checkpoints",
        orbax.checkpoint.PyTreeCheckpointer(),
        options=options,
    )

    best_epoch = checkpoint_manager.best_step()
    my_model.params = checkpoint_manager.restore(best_epoch)

    ################################# Transform the data #################################

    # Select the one-to-last timepoint
    idx_pred = adata.obs[eval_cfg.organism.time_obs] == timepoints[-2]
    idx_true = adata.obs[eval_cfg.organism.time_obs] == timepoints[-1]
    sub_adata_pred = adata[idx_pred].copy()

    # Initialize the prediction as an empty obsm.
    sub_adata_pred.obsm["pred"] = np.zeros_like(
        sub_adata_pred.obsm[eval_cfg.organism.obsm]
    )
    sub_adata_pred.obsm["space_pred"] = np.zeros_like(
        sub_adata_pred.obsm[eval_cfg.organism.space_obsm]
    )

    # Transform the data.
    pred, space_pred = my_model.transform(
        sub_adata_pred,
        eval_cfg.organism.time_obs,
        eval_cfg.organism.obsm,
        eval_cfg.organism.space_obsm,
    )
    sub_adata_pred.obsm["pred"] = pred
    sub_adata_pred.obsm["space_pred"] = space_pred

    # Normalize the obsm to a 3-D RGB space.
    pred_color = sub_adata_pred.obsm["pred"][:, :3].copy()
    color = adata.obsm[eval_cfg.organism.obsm][:, :3].copy()

    min_color = min(pred_color.min(), color.min())
    pred_color -= min_color
    color -= min_color

    max_color = max(pred_color.max(), color.max())
    pred_color /= max_color
    color /= max_color

    ############################ Compute the Sinkhorn distance ###########################

    geom = PointCloud(
        sub_adata_pred.obsm["pred"],
        adata.obsm[eval_cfg.organism.obsm][idx_true],
        epsilon=eval_cfg.model.epsilon,
        batch_size=512,
    )
    problem = LinearProblem(geom)
    solver = LRSinkhorn(rank=500)
    out = solver(problem)
    assert out.converged

    wandb.log({"sinkhorn": out.reg_ot_cost})
    scores_dict["sinkhorn"] = float(out.reg_ot_cost)

    fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

    dot_size = 5e3 / sub_adata_pred.n_obs
    sns.scatterplot(
        x=sub_adata_pred.obsm["space_pred"][:, 0],
        y=sub_adata_pred.obsm["space_pred"][:, 1],
        c=pred_color,
        ax=axes[0],
        s=dot_size,
    )
    axes[0].set_title(f"Prediction from {timepoints[-2]}", fontsize=9)
    axes[0].set_frame_on(False)

    dot_size = 5e3 / adata[idx_true].n_obs
    sns.scatterplot(
        x=adata.obsm[eval_cfg.organism.space_obsm][idx_true, 0],
        y=adata.obsm[eval_cfg.organism.space_obsm][idx_true, 1],
        c=color[idx_true],
        ax=axes[1],
        s=dot_size,
    )
    axes[1].set_title(f"Real timepoint {timepoints[-1]}", fontsize=9)
    axes[1].set_frame_on(False)

    for j in np.random.choice(sub_adata_pred.n_obs, size=10, replace=False):
        indicator = np.zeros(sub_adata_pred.n_obs)
        indicator[j] = 1
        push = out.apply(indicator)

        max_Pj = np.max(push)
        for k in np.argpartition(push, -5)[-5:]:
            con = ConnectionPatch(
                xyA=(
                    sub_adata_pred.obsm["space_pred"][j, 0],
                    sub_adata_pred.obsm["space_pred"][j, 1],
                ),
                coordsA=axes[0].transData,
                xyB=(
                    adata.obsm[eval_cfg.organism.space_obsm][
                        np.where(idx_true)[0][k], 0
                    ],
                    adata.obsm[eval_cfg.organism.space_obsm][
                        np.where(idx_true)[0][k], 1
                    ],
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

    labels_real = adata[idx_true].obs[eval_cfg.organism.annotation_obs]
    l1_dist, labels_pred = scores.compare_real_and_knn_histograms(
        x_real=adata[idx_true].obsm[eval_cfg.organism.obsm],
        labels_real=labels_real,
        x_pred=sub_adata_pred.obsm["pred"],
        k=5,
    )

    wandb.log({"L1": l1_dist})
    scores_dict["L1"] = float(l1_dist)

    histo_list = []
    for label in np.unique(labels_real):
        histo_list.append(
            {"cell type": label, "prop": np.mean(labels_real == label), "type": "real"}
        )
        histo_list.append(
            {"cell type": label, "prop": np.mean(labels_pred == label), "type": "pred"}
        )

    scores_dict["histo_list"] = histo_list

    histo_df = pd.DataFrame(histo_list)
    sns.barplot(data=histo_df, y="cell type", x="prop", hue="type")

    # Log the plot.
    image = wandb.Image(plt)
    wandb.log({"Histograms": image})
    plt.close("all")

    #################### Compute the Fused Gromov-Wasserstein distance ###################

    # geom_xx = PointCloud(
    #     sub_adata_pred.obsm["space_pred"],
    #     scale_cost="mean",
    #     epsilon=eval_cfg.model.epsilon,
    #     batch_size=64,
    # )
    # geom_yy = PointCloud(
    #     adata.obsm[eval_cfg.organism.space_obsm][idx_true],
    #     scale_cost="mean",
    #     epsilon=eval_cfg.model.epsilon,
    #     batch_size=64,
    # )
    # geom_xy = PointCloud(
    #     sub_adata_pred.obsm["pred"],
    #     adata.obsm[eval_cfg.organism.obsm][idx_true],
    #     epsilon=eval_cfg.model.epsilon,
    #     batch_size=64,
    # )
    # problem = QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=1.0)
    # solver = GromovWasserstein(rank=20)
    # out = solver(problem)
    # assert out.converged

    # wandb.log({"FGW": out.reg_gw_cost})
    # scores_dict["FGW"] = float(out.reg_gw_cost)

    # fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

    # dot_size = 5e3 / sub_adata_pred.n_obs
    # sns.scatterplot(
    #     x=sub_adata_pred.obsm["space_pred"][:, 0],
    #     y=sub_adata_pred.obsm["space_pred"][:, 1],
    #     c=pred_color,
    #     ax=axes[0],
    #     s=dot_size,
    # )
    # axes[0].set_title(f"Prediction from {timepoints[-2]}", fontsize=9)
    # axes[0].set_frame_on(False)

    # dot_size = 5e3 / adata[idx_true].n_obs
    # sns.scatterplot(
    #     x=adata.obsm[eval_cfg.organism.space_obsm][idx_true, 0],
    #     y=adata.obsm[eval_cfg.organism.space_obsm][idx_true, 1],
    #     c=color[idx_true],
    #     ax=axes[1],
    #     s=dot_size,
    # )
    # axes[1].set_title(f"Real timepoint {timepoints[-1]}", fontsize=9)
    # axes[1].set_frame_on(False)

    # for j in np.random.choice(sub_adata_pred.n_obs, size=10, replace=False):
    #     indicator = np.zeros(sub_adata_pred.n_obs)
    #     indicator[j] = 1
    #     push = out.apply(indicator)

    #     max_Pj = np.max(push)
    #     for k in np.argpartition(push, -5)[-5:]:
    #         con = ConnectionPatch(
    #             xyA=(
    #                 sub_adata_pred.obsm["space_pred"][j, 0],
    #                 sub_adata_pred.obsm["space_pred"][j, 1],
    #             ),
    #             coordsA=axes[0].transData,
    #             xyB=(
    #                 adata.obsm[eval_cfg.organism.space_obsm][
    #                     np.where(idx_true)[0][k], 0
    #                 ],
    #                 adata.obsm[eval_cfg.organism.space_obsm][
    #                     np.where(idx_true)[0][k], 1
    #                 ],
    #             ),
    #             coordsB=axes[1].transData,
    #             alpha=0.5 * min(float(push[k] / max_Pj), 1.0),
    #         )

    #         fig.add_artist(con)

    # # Log the plot.
    # image = wandb.Image(plt)
    # wandb.log({"Gromov plan": image})
    # plt.close("all")

    wandb.finish()

    # Save scores_dict as a pickle file
    with open(f"{cfg.checkpoint_path}/scores.pkl", "wb") as f:
        pickle.dump(scores_dict, f)


if __name__ == "__main__":
    main()
