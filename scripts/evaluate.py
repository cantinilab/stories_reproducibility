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
    from flax.linen.activation import gelu
    from flatten_dict import flatten

    import spacetime
    from ott.geometry.pointcloud import PointCloud
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.problems.quadratic.quadratic_problem import QuadraticProblem
    from matplotlib.patches import ConnectionPatch
    from ott.solvers.linear.sinkhorn import Sinkhorn
    from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
    from spacetime import potentials, scores, steps

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

    # Load the data.
    adata = ad.read_h5ad(eval_cfg.organism.dataset_path)

    # Normalize the obsm.
    adata.obsm[x_obsm] = adata.obsm[x_obsm][:, : eval_cfg.n_pcs]
    adata.obsm[x_obsm] /= adata.obsm[x_obsm].max()

    # Get the timepoints.
    timepoints = np.sort(np.unique(adata.obs[time_obs].astype(int)))

    ############################ Plot the original timepoints. ###########################

    # Define as many axes as timepoints.
    _, axes = plt.subplots(1, len(timepoints), figsize=(10, 3), constrained_layout=True)

    # For each timepoint, plot the cells.
    for i, timepoint in enumerate(timepoints):
        idx = adata.obs[time_obs] == timepoint
        dot_size = 5e4 / adata[idx].n_obs
        palette = sc.pl.palettes.default_28
        kwds = {"show": False, "frameon": False, "s": dot_size, "palette": palette}
        sc.pl.embedding(
            adata[idx], basis=space_obsm, color=annotation_obs, ax=axes[i], **kwds
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

    # If the proximal step is implicit, add the appropriate keyword arguments.
    if "implicit" in eval_cfg.step.type:
        step_kwargs["implicit_diff"] = eval_cfg.step.implicit_diff
        step_kwargs["maxiter"] = eval_cfg.step.maxiter
        step_kwargs["log_callback"] = lambda x: wandb.log(x)

    # Choose the proximal step.
    if eval_cfg.step.type == "linear_explicit":
        step = steps.LinearExplicitStep()
    elif eval_cfg.step.type == "monge_linear_implicit":
        step = steps.MongeLinearImplicitStep(**step_kwargs)
    elif eval_cfg.step.type == "monge_quadratic_implicit":
        step = steps.MongeQuadraticImplicitStep(**step_kwargs)
    else:
        raise ValueError(f"Step {eval_cfg.step.type} not recognized.")

    # Initialize the model.
    my_model = spacetime.SpaceTime(
        potential=potentials.MLPPotential(eval_cfg.potential.features, activation=gelu),
        proximal_step=step,
        tau=eval_cfg.model.tau,
        tau_auto=eval_cfg.model.tau_auto,
        teacher_forcing=eval_cfg.model.teacher_forcing,
        quadratic=eval_cfg.model.quadratic,
        epsilon=eval_cfg.model.epsilon,
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

    # Restore the model.
    best_epoch = checkpoint_manager.best_step()
    my_model.params = checkpoint_manager.restore(best_epoch)

    ################################# Transform the data #################################

    # Initialize the prediction.
    adata.obsm["pred"] = adata.obsm[x_obsm].copy()
    adata.obs[time_obs] = adata.obs[time_obs].astype(int)

    # Get the difference between consecutive timepoints.
    t_diff = np.diff(np.sort(adata.obs[time_obs].unique()))
    t_diff = t_diff.astype(float)

    # For each timepoint, transform the data.
    for i, t in enumerate(timepoints[:-1]):
        idx = adata.obs[time_obs] == t
        adata.obsm["pred"][idx] = my_model.transform(
            adata[idx], x_obsm=x_obsm, batch_size=500, tau=t_diff[i]
        )

    ############################ Compute the Sinkhorn distance ###########################

    # Initialize the stats.
    scores_dict["sinkhorn"] = []

    # For each timepoint, compute the Sinkhorn distance.
    for i, t in enumerate(timepoints[:-1]):
        # Get indices for the prediction and the ground-truth.
        idx = adata.obs[time_obs] == t
        idx_true = adata.obs[time_obs] == timepoints[i + 1]

        # Define the geometry.
        geom = PointCloud(
            adata.obsm["pred"][idx],
            adata.obsm[x_obsm][idx_true],
            epsilon=0.01,
            batch_size=512,
        )

        # Compute the Sinkhorn distance.
        problem = LinearProblem(geom)
        solver = Sinkhorn()
        out = solver(problem)
        assert out.converged
        sinkhorn_dist = float(out.reg_ot_cost)

        # Define the bias geometry.
        geom_bias = PointCloud(adata.obsm["pred"][idx], epsilon=0.01, batch_size=512)

        # Compute the Sinkhorn distance.
        problem_bias = LinearProblem(geom_bias)
        out_bias = solver(problem_bias)
        assert out_bias.converged
        sinkhorn_bias = float(out_bias.reg_ot_cost)

        # Define the bias geometry.
        geom_bias = PointCloud(
            adata.obsm[x_obsm][idx_true], epsilon=0.01, batch_size=512
        )

        # Compute the Sinkhorn distance.
        problem_bias = LinearProblem(geom_bias)
        out_bias = solver(problem_bias)
        assert out_bias.converged
        sinkhorn_bias += float(out_bias.reg_ot_cost)

        # Save and log the Sinkhorn distance.
        stats = {
            "timepoint": t,
            "sinkhorn": sinkhorn_dist,
            "sinkhorn_div": sinkhorn_dist - 0.5 * sinkhorn_bias,
        }
        wandb.log(stats)
        scores_dict["sinkhorn"].append(stats)

    ############################# Plot the last Sinkhorn plan ############################

    fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

    dot_size = 5e3 / adata[idx].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx, 0],
        y=adata.obsm[space_obsm][idx, 1],
        hue=adata[idx].obs[annotation_obs],
        ax=axes[0],
        s=dot_size,
    )
    axes[0].set_title(f"Prediction from {timepoints[-2]}", fontsize=9)
    axes[0].set_frame_on(False)
    axes[0].get_legend().remove()

    dot_size = 5e3 / adata[idx_true].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx_true, 0],
        y=adata.obsm[space_obsm][idx_true, 1],
        hue=adata[idx_true].obs[annotation_obs],
        ax=axes[1],
        s=dot_size,
    )
    axes[1].set_title(f"Real timepoint {timepoints[-1]}", fontsize=9)
    axes[1].set_frame_on(False)
    axes[1].get_legend().remove()

    # Reproducible random points.
    key = jax.random.PRNGKey(0)
    random_j = jax.random.choice(key, adata[idx].n_obs, shape=(10,), replace=False)
    random_j = np.array(random_j)

    for j in random_j:
        indicator = np.zeros(adata[idx].n_obs)
        indicator[j] = 1
        push = out.apply(indicator)

        max_Pj = np.max(push)
        for k in np.argpartition(push, -5)[-5:]:
            con = ConnectionPatch(
                xyA=(
                    adata[idx].obsm[space_obsm][j, 0],
                    adata[idx].obsm[space_obsm][j, 1],
                ),
                coordsA=axes[0].transData,
                xyB=(
                    adata.obsm[space_obsm][np.where(idx_true)[0][k], 0],
                    adata.obsm[space_obsm][np.where(idx_true)[0][k], 1],
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

    # Initialize the stats.
    scores_dict["L1"] = []

    # For each timepoint, compute the L1 distance.
    for i, t in enumerate(timepoints[:-1]):
        # Get indices for the prediction and the ground-truth.
        idx = adata.obs[time_obs] == t
        idx_true = adata.obs[time_obs] == timepoints[i + 1]

        # Compute the L1 distance.
        labels_real = adata[idx_true].obs[annotation_obs]
        l1_dist, labels_pred = scores.compare_real_and_knn_histograms(
            x_real=adata[idx_true].obsm[x_obsm],
            labels_real=labels_real,
            x_pred=adata.obsm["pred"],
            k=15,
        )

        # Save and log the L1 distance.
        stats = {"timepoint": t, "L1": float(l1_dist)}
        wandb.log(stats)
        scores_dict["L1"].append(stats)

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

    # Initialize the stats.
    scores_dict["FGW"] = []

    # For each timepoint, compute the FGW distance.
    for i, t in enumerate(timepoints[:-1]):
        # Get indices for the prediction and the ground-truth.
        idx = adata.obs[time_obs] == t
        idx_true = adata.obs[time_obs] == timepoints[i + 1]

        # Define the geometries.
        geom_xx = PointCloud(
            adata.obsm[space_obsm][idx],
            scale_cost="mean",
            epsilon=0.01,
            batch_size=512,
        )
        geom_yy = PointCloud(
            adata.obsm[space_obsm][idx_true],
            scale_cost="mean",
            epsilon=0.01,
            batch_size=512,
        )
        geom_xy = PointCloud(
            adata.obsm["pred"][idx],
            adata.obsm[x_obsm][idx_true],
            epsilon=0.01,
            batch_size=512,
        )

        # Compute the FGW distance.
        problem = QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=5.0)
        solver = GromovWasserstein()
        out = solver(problem)
        print(out.converged)
        assert out.converged

        # Save and log the FGW distance.
        stats = {"timepoint": t, "FGW": float(out.reg_gw_cost)}
        wandb.log(stats)
        scores_dict["FGW"].append(stats)

    ############################# Plot the last FGW plan #################################

    fig, axes = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

    dot_size = 5e3 / adata[idx].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx, 0],
        y=adata.obsm[space_obsm][idx, 1],
        hue=adata[idx].obs[annotation_obs],
        ax=axes[0],
        s=dot_size,
    )
    axes[0].set_title(f"Prediction from {timepoints[-2]}", fontsize=9)
    axes[0].set_frame_on(False)
    axes[0].get_legend().remove()

    dot_size = 5e3 / adata[idx_true].n_obs
    sns.scatterplot(
        x=adata.obsm[space_obsm][idx_true, 0],
        y=adata.obsm[space_obsm][idx_true, 1],
        hue=adata[idx_true].obs[annotation_obs],
        ax=axes[1],
        s=dot_size,
    )
    axes[1].set_title(f"Real timepoint {timepoints[-1]}", fontsize=9)
    axes[1].set_frame_on(False)
    axes[1].get_legend().remove()

    for j in random_j:
        indicator = np.zeros(adata[idx].n_obs)
        indicator[j] = 1
        push = out.apply(indicator)

        max_Pj = np.max(push)
        for k in np.argpartition(push, -5)[-5:]:
            con = ConnectionPatch(
                xyA=(
                    adata[idx].obsm[space_obsm][j, 0],
                    adata[idx].obsm[space_obsm][j, 1],
                ),
                coordsA=axes[0].transData,
                xyB=(
                    adata.obsm[space_obsm][np.where(idx_true)[0][k], 0],
                    adata.obsm[space_obsm][np.where(idx_true)[0][k], 1],
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


if __name__ == "__main__":
    main()
