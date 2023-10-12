# Imports
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="plot")
def main(cfg: DictConfig) -> None:
    import pickle

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sinkhorn_scores = []
    FGW_scores = []
    L1_scores = []
    histo_list = []

    for checkpoint_path in cfg.checkpoint_paths:
        # Get the config files of the run to evaluate.
        eval_cfg = OmegaConf.load(f"{checkpoint_path}/config.yaml")

        # Load the scores.
        scores_path = f"{checkpoint_path}/scores.pkl"
        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        sinkhorn_scores.append(
            {
                "sinkhorn": scores["sinkhorn"],
                "seed": eval_cfg.model.seed,
                "checkpoint_path": eval_cfg.checkpoint_path,
            }
        )
        FGW_scores.append(
            {
                "FGW": scores["FGW"],
                "seed": eval_cfg.model.seed,
                "checkpoint_path": eval_cfg.checkpoint_path,
            }
        )
        L1_scores.append(
            {
                "L1": scores["L1"],
                "seed": eval_cfg.model.seed,
                "checkpoint_path": eval_cfg.checkpoint_path,
            }
        )
        for x in scores["histo_list"]:
            histo_list.append(
                {
                    **x,
                    "seed": eval_cfg.model.seed,
                    "checkpoint_path": eval_cfg.checkpoint_path,
                }
            )

    df = pd.DataFrame(sinkhorn_scores)
    df["sinkhorn"] = df["sinkhorn"].astype("float")
    sns.barplot(data=df, x="checkpoint_path", y="sinkhorn")
    plt.savefig("plots/sinkhorn_scores.png")
    plt.close()

    df = pd.DataFrame(FGW_scores)
    df["FGW"] = df["FGW"].astype("float")
    sns.barplot(data=df, x="checkpoint_path", y="FGW")
    plt.savefig("plots/FGW_scores.png")
    plt.close()

    df = pd.DataFrame(L1_scores)
    df["L1"] = df["L1"].astype("float")
    sns.barplot(data=df, x="checkpoint_path", y="L1")
    plt.savefig("plots/L1_scores.png")
    plt.close()

    df = pd.DataFrame(histo_list)
    df["prop"] = df["prop"].astype("float")
    sns.catplot(
        data=df,
        y="cell type",
        col="checkpoint_path",
        x="prop",
        hue="type",
        kind="bar",
    )
    plt.savefig("plots/histo.png")
    plt.close()


if __name__ == "__main__":
    main()
