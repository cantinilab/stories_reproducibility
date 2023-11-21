from typing import Dict

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey, KeyArray


class DataLoader:
    """DataLoader feeds data from an AnnData object to the model as JAX arrays. It
    samples with replacement for a given batch size.

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

        # If time is valid, then we can get hold of the timepoints and their indices.
        self.timepoints = np.sort(np.unique(adata.obs[time_obs]))
        self.idx = [np.where(adata.obs[time_obs] == t)[0] for t in self.timepoints]

        # Fill in some fields.
        self.adata = adata
        self.time_obs = time_obs
        self.x_obsm = x_obsm
        self.space_obsm = space_obsm
        self.batch_size = batch_size
        self.train_val_split = train_val_split

        # Get the number of features, spatial dimensions, and timepoints.
        self.n_features = adata.obsm[x_obsm].shape[1]
        self.n_space = adata.obsm[space_obsm].shape[1]
        self.n_timepoints = len(self.timepoints)

    def make_train_val_split(self, key: KeyArray) -> None:
        """Make a train/validation split. Must be called before training.

        Args:
            key (PRNGKey): The random number generator key for permutations.
        """

        # Initialize the train and validation indices, from which we will sample batches.
        self.idx_train, self.idx_val = [], []

        # Iterate over timepoints.
        for idx_t in self.idx:
            # Permute the indices in order to make the split random.
            key, key_permutation = jax.random.split(key)
            permuted_idx = jax.random.permutation(key_permutation, idx_t)

            # Split the indices between train and validation.
            split = int(self.train_val_split * len(idx_t))
            self.idx_train.append(permuted_idx[:split])
            self.idx_val.append(permuted_idx[split:])

        # Print some stats about the split.
        print("Train (# cells): ", [len(idx) for idx in self.idx_train])
        print("Val (# cells): ", [len(idx) for idx in self.idx_val])

    def next_train(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        """Get the next training batch.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary of JAX arrays."""

        # Initialize the lists of omics and spatial coordinates over timepoints.
        x, space, a = [], [], []

        # Iterate over timepoints.
        for idx_t in self.idx_train:
            # if the batch size is smaller or equal to the number of cells n, then we
            # want to sample a minibatch without replacement.
            if self.batch_size <= len(idx_t):
                key, key_choice = jax.random.split(key)
                batch_idx = jax.random.choice(
                    key_choice, idx_t, shape=(self.batch_size,), replace=False
                )
                batch_a = np.ones(self.batch_size) / self.batch_size
            # if the batch size is greater than the number of cells n, then we want
            # to pad the cells with random cells and pad a with zeroes.
            else:
                key, key_choice = jax.random.split(key)
                batch_idx = jax.random.choice(
                    key_choice, idx_t, shape=(self.batch_size - len(idx_t),)
                )
                batch_idx = np.concatenate((idx_t, batch_idx))
                batch_a = np.concatenate(
                    (np.ones(len(idx_t)), np.zeros(self.batch_size - len(idx_t)))
                )
                batch_a /= np.sum(batch_a)

            # Get the omics and spatial coordinates for the batch.
            x.append(self.adata.obsm[self.x_obsm][batch_idx])
            space.append(self.adata.obsm[self.space_obsm][batch_idx])
            a.append(batch_a)

        # Return a dictionary of JAX arrays, the first axis being time.
        return {
            "x": jnp.array(np.stack(x)),
            "space": jnp.array(np.stack(space)),
            "a": jnp.array(np.stack(a)),
        }

    def next_val(self, key: KeyArray) -> Dict[str, jnp.ndarray]:
        """Get the next validation batch.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary of JAX arrays."""
        x, space, a = [], [], []

        # Iterate over timepoints.
        for idx_t in self.idx_val:
            # if the batch size is smaller or equal to the number of cells n, then we
            # want to sample a minibatch without replacement.
            if self.batch_size <= len(idx_t):
                key, key_choice = jax.random.split(key)
                batch_idx = jax.random.choice(
                    key_choice, idx_t, shape=(self.batch_size,), replace=False
                )
                batch_a = np.ones(self.batch_size) / self.batch_size
            # if the batch size is greater than the number of cells n, then we want
            # to pad the cells with random cells and pad a with zeroes.
            else:
                key, key_choice = jax.random.split(key)
                batch_idx = jax.random.choice(
                    key_choice, idx_t, shape=(self.batch_size - len(idx_t),)
                )
                batch_idx = np.concatenate((idx_t, batch_idx))
                batch_a = np.concatenate(
                    (np.ones(len(idx_t)), np.zeros(self.batch_size - len(idx_t)))
                )
                batch_a /= np.sum(batch_a)

            # Get the omics and spatial coordinates for the batch.
            x.append(self.adata.obsm[self.x_obsm][batch_idx])
            space.append(self.adata.obsm[self.space_obsm][batch_idx])
            a.append(batch_a)

        # Return a dictionary of JAX arrays, the first axis being time.
        return {
            "x": jnp.array(np.stack(x)),
            "space": jnp.array(np.stack(space)),
            "a": jnp.array(np.stack(a)),
        }
