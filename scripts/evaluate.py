import hydra
from omegaconf import DictConfig, OmegaConf
import traceback


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    try:
        import pickle
        import jax
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb
        from flatten_dict import flatten

        # Add scripts to the path.
        import sys

        sys.path.append(
            "/pasteur/zeus/projets/p02/ml4ig_hot/Users/ghuizing/space-time/scripts"
        )

        from evaluation_utils import pred, load_data, define_model, plot_plan
        from evaluation_scores import sinkhorn, chamfer, hausdorff, fgw

        ################################ Get the configuration ###########################

        # Get the config file of the run to evaluate.
        eval_cfg = OmegaConf.load(f"{cfg.checkpoint_path}/config.yaml")

        # Initialize a dict with scores to save.
        scores_dict = {}

        # Initialize Weights & Biases.
        config = flatten(OmegaConf.to_container(eval_cfg, resolve=True), reducer="dot")
        wandb.init(project="evaluate_spacetime", config=config, mode=cfg.wandb.mode)
        print(config, f"JAX device type: {jax.devices()[0].device_kind}")

        # Get some parameters.
        omics_key = eval_cfg.organism.omics_key
        space_key = eval_cfg.organism.space_key
        time_key = eval_cfg.organism.time_key
        annotation_key = eval_cfg.organism.annotation_key
        train_batches = eval_cfg.organism.train_batches
        early_test_batches = eval_cfg.organism.early_test_batches
        late_test_batches = eval_cfg.organism.late_test_batches

        ############################### Load the data ####################################

        # Load the data.
        adata = load_data(
            dataset_path=eval_cfg.organism.dataset_path,
            omics_key=omics_key,
            space_key=space_key,
            n_pcs=eval_cfg.n_pcs,
        )

        # Get the indices for the training, early test, and late test batches.
        idx_train = np.isin(adata.obs["Batch"], train_batches)
        idx_early_test = np.isin(adata.obs["Batch"], early_test_batches)
        idx_late_test = np.isin(adata.obs["Batch"], late_test_batches)

        ############################## Define the step and model #########################

        my_model = define_model(
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
        adata.obsm["pred"] = adata.obsm[omics_key].copy()
        adata.obs[time_key] = adata.obs[time_key].astype(float)

        # Transform the data on training batches, early and late test batches.
        pred(adata, idx_train, time_key, my_model, omics_key)
        pred(adata, idx_early_test, time_key, my_model, omics_key)
        pred(adata, idx_late_test, time_key, my_model, omics_key)

        ############################ Compute the Sinkhorn distance #######################

        # Compute the Sinkhorn distance for each timepoint on the training set.
        score_name = "sinkhorn_train"
        stats, _ = sinkhorn(adata, idx_train, score_name, time_key, omics_key)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the Sinkhorn distance for each timepoint on the early test set.
        score_name = "sinkhorn_early_test"
        stats, _ = sinkhorn(adata, idx_early_test, score_name, time_key, omics_key)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the Sinkhorn distance for each timepoint on the late test set.
        score_name = "sinkhorn_late_test"
        stats, res = sinkhorn(adata, idx_late_test, score_name, time_key, omics_key)
        idx_last, idx_true_last, out_last, timepoints_last = res  # Keep for plot.
        wandb.log(stats)
        scores_dict[score_name] = stats

        ############################# Plot the last Sinkhorn plan ########################

        # Reproducible random points.
        key, n_cells = jax.random.PRNGKey(0), adata[idx_last].n_obs
        random_j = np.array(jax.random.choice(key, n_cells, shape=(10,), replace=False))

        # Keyword arguments for the plot.
        plot_kwds = {
            "adata": adata,
            "space_key": space_key,
            "annotation_key": annotation_key,
            "random_j": random_j,
        }

        # Plot the last Sinkhorn plan.
        fig, axes = plot_plan(
            idx_last=idx_last,
            idx_true_last=idx_true_last,
            timepoints_last=timepoints_last,
            out_last=out_last,
            **plot_kwds,
        )

        # Log the plot.
        image = wandb.Image(plt)
        wandb.log({"Sinkhorn plan": image})
        plt.close("all")

        ############################ Compute the Hausdorff distance ######################

        # Compute the Hausdorff distance for each timepoint on the training set.
        score_name = "hausdorff_train"
        stats = hausdorff(adata, idx_train, score_name, time_key, omics_key)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the Hausdorff distance for each timepoint on the early test set.
        score_name = "hausdorff_early_test"
        stats = hausdorff(adata, idx_early_test, score_name, time_key, omics_key)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the Hausdorff distance for each timepoint on the late test set.
        score_name = "hausdorff_late_test"
        stats = hausdorff(adata, idx_late_test, score_name, time_key, omics_key)
        wandb.log(stats)
        scores_dict[score_name] = stats

        ############################ Compute the Chamfer distance ########################

        # Compute the Chamfer distance for each timepoint on the training set.
        stats = chamfer(adata, idx_train, "chamfer_train", time_key, omics_key)
        wandb.log(stats)
        scores_dict["chamfer_train"] = stats

        # Compute the Chamfer distance for each timepoint on the early test set.
        stats = chamfer(
            adata, idx_early_test, "chamfer_early_test", time_key, omics_key
        )
        wandb.log(stats)
        scores_dict["chamfer_early_test"] = stats

        # Compute the Chamfer distance for each timepoint on the late test set.
        stats = chamfer(adata, idx_late_test, "chamfer_late_test", time_key, omics_key)
        wandb.log(stats)
        scores_dict["chamfer_late_test"] = stats

        #################### Compute the Fused Gromov-Wasserstein distance ###############

        # Keyword arguments for the FGW function.
        fgw_kwargs = {
            "adata": adata,
            "time_key": time_key,
            "space_key": space_key,
            "omics_key": omics_key,
        }

        # Compute the FGW distance for each timepoint on the training set.
        score_name = "fgw_train"
        stats, _ = fgw(idx_batches=idx_train, score_name=score_name, **fgw_kwargs)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the FGW distance for each timepoint on the early test set.
        score_name = "fgw_early_test"
        stats, _ = fgw(idx_batches=idx_early_test, score_name=score_name, **fgw_kwargs)
        wandb.log(stats)
        scores_dict[score_name] = stats

        # Compute the FGW distance for each timepoint on the late test set.
        score_name = "fgw_late_test"
        stats, res = fgw(idx_batches=idx_late_test, score_name=score_name, **fgw_kwargs)
        idx_last, idx_true_last, out_last, timepoints_last = res  # Keep for plot.
        wandb.log(stats)
        scores_dict[score_name] = stats

        ############################# Plot the last FGW plan #############################

        # Plot the last FGW plan.
        fig, axes = plot_plan(
            idx_last=idx_last,
            idx_true_last=idx_true_last,
            timepoints_last=timepoints_last,
            out_last=out_last,
            **plot_kwds,
        )

        # Log the plot.
        image = wandb.Image(plt)
        wandb.log({"FGW plan": image})
        plt.close("all")

        ################################# Finish the run #################################

        wandb.finish()

        # Save scores_dict as a pickle file
        with open(f"{cfg.checkpoint_path}/scores.pkl", "wb") as f:
            pickle.dump(scores_dict, f)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
