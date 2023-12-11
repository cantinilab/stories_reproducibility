from typing import Callable

from dataclasses import dataclass
from anndata import AnnData
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
from flax.training.early_stopping import EarlyStopping
from jax.random import KeyArray, PRNGKey
from optax import GradientTransformation
from orbax.checkpoint import CheckpointManager
from tqdm import tqdm

from .steps.proximal_step import ProximalStep
from .steps.explicit import ExplicitStep
from .potentials import MLPPotential
from .tools import DataLoader
from .loss import loss_fn


@dataclass
class SpaceTime:
    """Multi-modal Wasserstein gradient flow model for spatio-temporal omics data."""

    potential: nn.Module = MLPPotential()
    proximal_step: ProximalStep = ExplicitStep()
    n_steps: int = 1
    teacher_forcing: bool = True
    quadratic: bool = False
    debias: bool = True
    epsilon: float = 0.05
    balancedness: float = 1.0
    log_callback: Callable = None
    fused_penalty: float = 5.0

    def fit(
        self,
        adata: AnnData,
        time_obs: str,
        x_obsm: str,
        space_obsm: str,
        optimizer: GradientTransformation = optax.adamw(1e-2),
        max_iter: int = 10_000,
        batch_size: int = 250,
        train_val_split: float = 0.8,
        min_delta: float = 1e-4,
        patience: float = 25,
        checkpoint_manager: CheckpointManager = None,
        key: KeyArray = PRNGKey(0),
    ) -> None:
        """Fit the model.

        Args:
            adata (AnnData): The AnnData object.
            time_obs (str): The name of the time observation.
            x_obsm (str): The name of the obsm field containing cell coordinates.
            space_obsm (str): The name of the obsm field containing space coordinates.
            optimizer (GradientTransformation, optional): The optimizer.
            max_iter (int, optional): The max number of iterations. Defaults to 10_000.
            batch_size (int, optional): The batch size. Defaults to 250.
            train_val_split (float, optional): The proportion of train in the split.
            min_delta (float, optional): The minimum delta for early stopping.
            patience (float, optional): The patience for early stopping.
            checkpoint_manager (CheckpointManager, optional): The checkpoint manager.
            key (KeyArray, optional): The random key. Defaults to PRNGKey(0).
        """

        # Initialize the statistics for logging.
        self.train_it, self.train_losses = [], []
        self.val_it, self.val_losses = [], []

        # Create a data loader for the AnnData object.
        dataloader = DataLoader(
            adata,
            time_obs=time_obs,
            x_obsm=x_obsm,
            space_obsm=space_obsm,
            batch_size=batch_size,
            train_val_split=train_val_split,
        )

        # Split the cells ids into train and validation.
        split_key, key = jax.random.split(key)
        dataloader.make_train_val_split(split_key)

        # Compute tau from the timepoints.
        tau_diff = jnp.array(np.diff(dataloader.timepoints).astype(float))

        # Define some arguments shared by the train and validation loss.
        lkwargs = {"teacher_forcing": self.teacher_forcing, "potential": self.potential}
        lkwargs = {**lkwargs, "balancedness": self.balancedness, "tau_diff": tau_diff}
        lkwargs = {**lkwargs, "quadratic": self.quadratic, "debias": self.debias}
        lkwargs = {**lkwargs, "n_steps": self.n_steps, "epsilon": self.epsilon}
        lkwargs = {**lkwargs, "fused_penalty": self.fused_penalty}
        lkwargs = {**lkwargs, "proximal_step": self.proximal_step}

        @jax.jit
        def jitted_update(params, opt_state, batch):
            """A jitted update function."""
            v, g = jax.value_and_grad(loss_fn)(params, batch, **lkwargs)
            updates, opt_state = optimizer.update(g, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, v

        @jax.jit
        def jitted_validation(params, batch):
            """A jitted validation function."""
            return loss_fn(params, batch, **lkwargs)

        # Initialize the parameters of the potential function and of the optimizer.
        init_key, key = jax.random.split(key)
        dummy_x = jnp.ones((batch_size, dataloader.n_features))  # Used to infer sizes.
        self.params = self.potential.init(init_key, dummy_x)
        opt_state = optimizer.init(self.params)

        # Define the early stopping criterion and checkpointing parameters.
        early_stop = EarlyStopping(min_delta=min_delta, patience=patience)
        save_kwargs = {"save_args": orbax_utils.save_args_from_target(self.params)}

        # The training loop.
        pbar = tqdm(range(1, max_iter + 1))
        for it in pbar:
            # Randomly choose whether to train or validate, weighted by train-val split.
            is_train_key, batch_key, key = jax.random.split(key, num=3)
            is_train = dataloader.train_or_val(is_train_key)

            # If train, update the parameters. If val, just compute the loss.
            if is_train:
                self.params, opt_state, train_loss = jitted_update(
                    self.params, opt_state, dataloader.next(batch_key, "train")
                )
                self.train_losses.append(train_loss)
                self.train_it.append(it)
            else:
                next_batch = dataloader.next(batch_key, "val")
                val_loss = jitted_validation(self.params, next_batch)
                self.val_losses.append(val_loss)
                self.val_it.append(it)

            # Recap the statistics for the current iteration.
            len_train, len_val = len(self.train_losses), len(self.val_losses)
            iteration_stats = {
                "iteration": it,
                "train_loss": np.inf if len_train == 0 else train_loss,
                "val_loss": np.inf if len_val == 0 else val_loss,
            }

            # Update the progress bar and log.
            pbar.set_postfix(iteration_stats)
            if self.log_callback:
                self.log_callback(iteration_stats)

            # Should we try early stopping and try to save the parameters?
            # If we have validation set, this is done every validation batch.
            # If we don't have validation set, this is done every train batch.
            update_train = train_val_split == 1.0 and is_train
            update_val = train_val_split < 1.0 and not is_train
            if update_train or update_val:
                # Check early stopping.
                last_l = self.train_losses[-1] if update_train else self.val_losses[-1]
                _, early_stop = early_stop.update(last_l)
                if early_stop.should_stop:
                    print("Met early stopping criteria, breaking...")
                    break

                # If we have a checkpoint manager, try to save the parameters.
                if checkpoint_manager:
                    metrics = {"loss": np.float64(last_l)}
                    checkpoint_manager.save(it, self.params, save_kwargs, metrics)

    def transform(
        self,
        adata: AnnData,
        x_obsm: str,
        tau: float,
        batch_size: int = 250,
        key: KeyArray = PRNGKey(0),
    ) -> np.ndarray:
        """Transform an AnnData object.

        Args:
            adata (AnnData): The AnnData object to transform.
            x_obsm (str): The obsm field containing the data to transform.
            tau (float, optional): The time step.
            batch_size (int, optional): The batch size. Defaults to 250.
            key (KeyArray, optional): The random key. Defaults to PRNGKey(0).

        Returns:
            np.ndarray: The predictions.
        """

        # Define the potential function.
        potential_fun = lambda u: self.potential.apply(self.params, u)

        # Helper function to transform a batch.
        def _transform_batch(idx_batch):
            x = jnp.array(adata.obsm[x_obsm][idx_batch])
            a = jnp.ones(len(idx_batch)) / len(idx_batch)
            return self.proximal_step.chained_inference_steps(
                x, a, potential_fun, tau, self.n_steps
            )

        # Initizalize the prediction.
        x_pred = np.zeros(adata.obsm[x_obsm].shape)

        # Iterate over batches and store the predictions.
        idx = np.array(jax.random.permutation(key, jnp.arange(x_pred.shape[0])))
        for idx_batch in np.array_split(idx, x_pred.shape[0] // batch_size):
            x_pred[idx_batch] = np.array(_transform_batch(idx_batch))

        # Return the predictions.
        return x_pred
