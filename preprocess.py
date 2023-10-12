import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    # Load the dataset.
    adata = sc.read_h5ad(cfg.data_path)
    print("Loaded dataset: ", adata)

    # If we have "spatial_x" and "spatial_y" obs, combine them into a "spatial" obsm.
    if "spatial_x" in adata.obs and "spatial_y" in adata.obs:
        adata.obsm["spatial"] = adata.obs[["spatial_x", "spatial_y"]].values

    # Optionally flip the y-axis.
    if cfg.flip_y:
        adata.obsm["spatial"][:, 1] = -adata.obsm["spatial"][:, 1]

    # Replace the preprocessed counts with the raw counts.
    adata.X = adata.layers[cfg.count_layer]

    # Filter the counts.
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print("Filtered counts")

    # Compute Pearson Residuals.
    sc.experimental.pp.highly_variable_genes(
        adata,
        n_top_genes=cfg.n_top_genes,
        batch_key=cfg.batch_key,
        subset=True,
        chunksize=500,
    )
    sc.experimental.pp.normalize_pearson_residuals(adata)
    print("Computed Pearson Residuals")

    # Apply PCA.
    sc.tl.pca(adata)
    print("Computed PCA")

    # Integrate the batches.
    sc.external.pp.harmony_integrate(adata, key=cfg.batch_key)
    print("Integrated batches")

    # Compute the UMAP.
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    print("Computed neighbors")
    sc.tl.umap(adata)
    print("Computed UMAP")

    if cfg.time_recipe == "zebrafish":
        adata.obs["time"] = (
            adata.obs["time"].str.replace("hpf", "").astype(float).astype(int)
        )
    if cfg.time_recipe == "axolotl":
        idx = adata.obs["Batch"] != "Injury_control_FP200000239BL_E3"
        adata = adata[idx]
        adata.obs["time"] = (
            adata.obs["Batch"].str.split("_").str[1].str.replace("DPI", "").astype(int)
        )
    if cfg.time_recipe == "mouse":
        adata.obs["time"] = (
            adata.obs["timepoint"].str.replace("E", "").astype(float).astype(int)
        )
    elif cfg.time_recipe == "midbrain":
        adata.obs["time"] = (
            adata.obs["Time point"].str.replace("E", "").astype(float).astype(int)
        )

        idx = adata.obs["Batch"] == "FP200000600TR_E3"
        idx |= adata.obs["Batch"] == "SS200000108BR_A3A4"
        idx |= adata.obs["Batch"] == "SS200000108BR_B1B2"
        adata = adata[idx]

    # Save the data.
    adata.write_h5ad(cfg.output_path)
    print("Saved data")

    # Plot the UMAP.
    sc.pl.umap(adata, color=[cfg.color_key, cfg.batch_key])
    plt.savefig(cfg.output_path + ".png")


if __name__ == "__main__":
    main()
