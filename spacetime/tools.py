from typing import Dict

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey


class DataLoader:
    """DataLoader feeds data from an AnnData object to the model as JAX arrays.

    Args:
        adata (ad.AnnData): The input AnnData object.
        time_obs (str): The obs field with the integer time observations
        x_obsm (str): The obsm field with the omics coordinates.
        space_obsm (str): The obsm field with the spatial coordinates.
        batch_size (int): The batch size.
        train_val_split (float, optional): The proportion of train in the split.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        time_obs: str,
        x_obsm: str,
        space_obsm: str,
        batch_size: int,
        train_val_split: float,
    ):

        # Check that we have a valid time observation.
        assert adata.obs[time_obs].dtype == int, "Time observations must be integers."
        self.timepoints = np.sort(np.unique(adata.obs[time_obs]))

        # Fill in some fields.
        self.adata = adata
        self.time_obs = time_obs
        self.x_obsm = x_obsm
        self.space_obsm = space_obsm
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.stats = []

        # Get the number of features, spatial dimensions, and timepoints.
        self.n_features = adata.obsm[x_obsm].shape[1]
        self.n_space = adata.obsm[space_obsm].shape[1]
        self.n_timepoints = len(self.timepoints)

        # Get the indices for each timepoint.
        self.idx = [np.where(adata.obs[time_obs] == t)[0] for t in self.timepoints]

    def make_train_val_split(self, key: PRNGKey) -> None:
        """Make a train/validation split. Must be called before training.

        Args:
            key (PRNGKey): The random number generator key for permutations.
        """

        # Iterate over timepoints.
        self.idx_train, self.idx_val = [], []
        for idx_t in self.idx:

            # Permute the indices.
            key, key_permutation = jax.random.split(key)
            permuted_idx = jax.random.permutation(key_permutation, idx_t)

            # Split the indices between train and validation.
            split = int(self.train_val_split * len(idx_t))
            self.idx_train.append(permuted_idx[:split])
            self.idx_val.append(permuted_idx[split:])

        # Get the number of training and validation batches for each timepoint.
        s = self.batch_size
        self.n_batches_train = [len(idx_train_t) // s for idx_train_t in self.idx_train]
        self.n_batches_val = [len(idx_val_t) // s for idx_val_t in self.idx_val]

        # Reset the batch counters.
        self.current_batch_train = 0
        self.current_batch_val = 0

    def shuffle_train_val(self, key: PRNGKey) -> None:
        """Shuffle the train and validation sets. To be called at the start of an epoch.

        Args:
            key (PRNGKey): The random number generator key for permutations.
        """

        # Iterate over timepoints.
        for i, idx_t in enumerate(self.idx_train):

            # Permute the indices.
            key, key_permutation = jax.random.split(key)
            self.idx_train[i] = jax.random.permutation(key_permutation, idx_t)

        # Iterate over timepoints.
        for i, idx_t in enumerate(self.idx_val):

            # Permute the indices.
            key, key_permutation = jax.random.split(key)
            self.idx_val[i] = jax.random.permutation(key_permutation, idx_t)

        # Reset the batch counters.
        self.current_batch_train = 0
        self.current_batch_val = 0

    def next_train(self) -> Dict[str, jnp.ndarray]:
        """Get the next training batch.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary of JAX arrays."""

        x, space = [], []

        # Iterate over timepoints.
        for idx_t, n_batches_t in zip(self.idx_train, self.n_batches_train):

            # Get the start and stop indices for the batch.
            start = (self.current_batch_train % n_batches_t) * self.batch_size
            stop = start + self.batch_size

            # Get the omics and spatial coordinates for the batch.
            x.append(self.adata.obsm[self.x_obsm][idx_t[start:stop]])
            space.append(self.adata.obsm[self.space_obsm][idx_t[start:stop]])

        # Increment the batch counter.
        self.current_batch_train += 1

        # Return a dictionary of JAX arrays.
        return {"x": jnp.array(np.stack(x)), "space": jnp.array(np.stack(space))}

    def next_val(self) -> Dict[str, jnp.ndarray]:
        """Get the next validation batch.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary of JAX arrays."""
        x, space = [], []

        # Iterate over timepoints.
        for idx_t, n_batches_t in zip(self.idx_val, self.n_batches_val):

            # Get the start and stop indices for the batch.
            start = (self.current_batch_val % n_batches_t) * self.batch_size
            stop = start + self.batch_size

            # Get the omics and spatial coordinates for the batch.
            x.append(self.adata.obsm[self.x_obsm][idx_t[start:stop]])
            space.append(self.adata.obsm[self.space_obsm][idx_t[start:stop]])

        # Increment the batch counter.
        self.current_batch_val += 1

        # Return a dictionary of JAX arrays.
        return {"x": jnp.array(np.stack(x)), "space": jnp.array(np.stack(space))}
