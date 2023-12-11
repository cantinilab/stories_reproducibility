from typing import Callable, Tuple, Dict

import anndata as ad
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

from ott.solvers.linear.implicit_differentiation import ImplicitDiff
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
from ott.geometry.costs import SqEuclidean

from .steps.proximal_step import ProximalStep
from .steps.linear_explicit import LinearExplicitStep
from .potentials import MLPPotential
from .tools import DataLoader
from .loss import loss_fn


class SpaceTime:
    """Multi-modal Fused Gromov-Wasserstein gradient
    flow model for spatio-temporal omics data.

    Args:
        potential (nn.Module): The scalar potential neural network.
        proximal_step (ProximalStep): The proximal step to use.
        tau (float, optional): The time step. Defaults to 1.0.
        teacher_forcing (bool, optional): Whether to use teacher forcing.
        quadratic (bool, optional): Whether to use a Gromov-Wasserstein loss.
        debias (bool, optional): Whether to debias the Sinkhorn/FGW loss.
        epsilon (float, optional): The entropy parameter for the loss.
        balancedness (float, optional): The balancedness parameter for the loss.
        log_callback (Callable, optional): A callback function to log optimization.
        fused_penalty (float, optional): The penalty for the fused term.
    """

    def __init__(
        self,
        potential: nn.Module = MLPPotential(),
        proximal_step: ProximalStep = LinearExplicitStep(),
        tau: float = 1.0,
        tau_auto: bool = False,
        n_steps: int = 1,
        teacher_forcing: bool = True,
        quadratic: bool = False,
        debias: bool = True,
        cost_fn: Callable = SqEuclidean(),
        epsilon: float = 0.05,
        balancedness: float = 1.0,
        log_callback: Callable = None,
        fused_penalty: float = 5.0,
    ):
        # Fill in some fields.
        self.potential = potential
        self.tau = tau
        self.tau_auto = tau_auto
        self.n_steps = n_steps
        self.quadratic = quadratic
        self.debias = debias
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.balancedness = balancedness
        self.teacher_forcing = teacher_forcing
        self.log_callback = log_callback
        self.proximal_step = proximal_step
        self.fused_penalty = fused_penalty

        # Check that is debias is True, cost_fn is SqEuclidean.
        if self.debias:
            error_msg = "cost_fn must be SqEuclidean() when debias is True."
            assert isinstance(self.cost_fn, SqEuclidean), error_msg

        # Initialize the OTT solver (quadratic or linear).
        impl_diff = ImplicitDiff(symmetric=True)
        if quadratic:
            self.ott_solver = GromovWasserstein(
                threshold=1e-3,
                implicit_diff=impl_diff,
                epsilon=self.epsilon,
                relative_epsilon=True,
            )
        else:
            self.ott_solver = Sinkhorn(threshold=1e-3, implicit_diff=impl_diff)

    def fit(
        self,
        adata: ad.AnnData,
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
            adata (ad.AnnData): The AnnData object.
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
        self.grad_norms = []

        # Create a data loader for the AnnData object.
        dataloader = DataLoader(
            adata,
            time_obs=time_obs,
            x_obsm=x_obsm,
            space_obsm=space_obsm,
            batch_size=batch_size,
            train_val_split=train_val_split,
        )
        print("Created data loader.")

        # If tau_auto, compute tau from the timepoints.
        if self.tau_auto:
            self.tau_diff = np.diff(np.sort(adata.obs[time_obs].unique())).astype(float)
            self.tau_diff = jnp.array(self.tau_diff)
        else:
            self.tau_diff = jnp.ones(len(dataloader.timepoints) - 1)

        # TODO: define a jitted validation function.

        @jax.jit
        def jitted_update(params, opt_state, batch):
            """A jitted update function."""

            # A bit of a mouthful.
            v, g = jax.value_and_grad(loss_fn)(
                params,
                batch,
                teacher_forcing=self.teacher_forcing,
                quadratic=self.quadratic,
                proximal_step=self.proximal_step,
                potential=self.potential,
                n_steps=self.n_steps,
                epsilon=self.epsilon,
                cost_fn=self.cost_fn,
                balancedness=self.balancedness,
                debias=self.debias,
                fused_penalty=self.fused_penalty,
                ott_solver=self.ott_solver,
                tau_diff=self.tau_diff,
            )

            # Update the parameters and the optimizer state.
            updates, opt_state = optimizer.update(g, opt_state, params)

            # Update the parameters.
            params = optax.apply_updates(params, updates)

            # Return the updated parameters, optimizer state, loss, and gradient.
            return params, opt_state, v, g

        @jax.jit
        def jitted_validation(params, batch):
            """A jitted validation function."""

            # A bit of a mouthful.
            return loss_fn(
                params,
                batch,
                teacher_forcing=self.teacher_forcing,
                quadratic=self.quadratic,
                proximal_step=self.proximal_step,
                potential=self.potential,
                n_steps=self.n_steps,
                epsilon=self.epsilon,
                cost_fn=self.cost_fn,
                balancedness=self.balancedness,
                debias=self.debias,
                fused_penalty=self.fused_penalty,
                ott_solver=self.ott_solver,
                tau_diff=self.tau_diff,
            )

        # Initialize the parameters of the potential function and of the optimizer.
        init_key, key = jax.random.split(key)
        dummy_x = jnp.ones((batch_size, dataloader.n_features))  # Used to infer sizes.
        self.params = self.potential.init(init_key, dummy_x)
        opt_state = optimizer.init(self.params)

        # Define the progress bar.
        pbar = tqdm(range(1, max_iter + 1))

        # Define the early stopping criterion.
        early_stop = EarlyStopping(min_delta=min_delta, patience=patience)

        # Split the cells ids into train and validation.
        split_key, key = jax.random.split(key)
        dataloader.make_train_val_split(split_key)

        # Define the structure of the parameters for checkpointing.
        if checkpoint_manager:
            save_kwargs = {"save_args": orbax_utils.save_args_from_target(self.params)}

        # The training loop.
        for it in pbar:
            # Randomly choose whether to train or validate, weighted by train-val split.
            is_train_key, key = jax.random.split(key)
            p = jnp.array([train_val_split, 1 - train_val_split])
            is_train = jax.random.choice(is_train_key, jnp.array([True, False]), p=p)

            # Sample a batch, do updates, and track statistics.
            batch_key, key = jax.random.split(key)
            if is_train:
                # If it is a train batch, update the parameters.
                self.params, opt_state, train_loss, g = jitted_update(
                    self.params,
                    opt_state=opt_state,
                    batch=dataloader.next_train(batch_key),
                )

                # Then save statistics.
                self.train_losses.append(train_loss)
                self.train_it.append(it)
                self.grad_norms.append(optax.global_norm(g))
            else:
                # If it is a val batch, just compute the loss.
                val_loss = jitted_validation(
                    self.params,
                    batch=dataloader.next_val(batch_key),
                )

                # Then save statistics.
                self.val_losses.append(val_loss)
                self.val_it.append(it)

            # Recap the statistics for the current iteration.
            len_train, len_val = len(self.train_losses), len(self.val_losses)
            iteration_stats = {
                "iteration": it,
                "train_loss": np.inf if len_train == 0 else self.train_losses[-1],
                "val_loss": np.inf if len_val == 0 else self.val_losses[-1],
                "grad_norm": np.inf if len_train == 0 else self.grad_norms[-1],
            }

            # Update the progress bar.
            pbar.set_postfix(iteration_stats)

            # If we have a log callback, then log the iteration statistics.
            if self.log_callback:
                self.log_callback(iteration_stats)

            # Should we try early stopping and try to save the parameters?
            # If we have validation set, this is done every validation batch.
            # If we don't have validation set, this is done every train batch.
            update_val = train_val_split < 1.0 and not is_train
            update_train = train_val_split == 1.0 and is_train

            # If so, check early stopping and potentially save the parameters.
            if update_train or update_val:
                # Check for early stopping.
                if update_train:
                    early_stop_loss = self.train_losses[-1]
                elif update_val:
                    early_stop_loss = self.val_losses[-1]

                _, early_stop = early_stop.update(early_stop_loss)

                # If we have a checkpoint manager, try to save the parameters.
                if checkpoint_manager:
                    checkpoint_manager.save(
                        step=it,
                        items=self.params,
                        save_kwargs=save_kwargs,
                        metrics={"loss": np.float64(early_stop_loss)},
                    )

                # If we should stop, break the loop.
                if early_stop.should_stop:
                    print("Met early stopping criteria, breaking...")
                    break

    def transform(
        self,
        adata: ad.AnnData,
        x_obsm: str,
        tau: float = None,
        batch_size: int = 250,
        key: KeyArray = PRNGKey(0),
    ) -> np.ndarray:
        """Transform an AnnData object.

        Args:
            adata (ad.AnnData): The AnnData object to transform.
            x_obsm (str): The obsm field containing the data to transform.
            batch_size (int, optional): The batch size. Defaults to 250.

        Returns:
            np.ndarray: The predictions.
        """

        # If the user did not specify a time step, use the default.
        tau = self.tau if tau is None else tau

        # Initizalize the prediction.
        x_pred = np.zeros(adata.obsm[x_obsm].shape)

        # Get a shuffled list of indices and separate them into batches.
        idx = np.array(jax.random.permutation(key, jnp.arange(x_pred.shape[0])))
        batches = np.array_split(idx, x_pred.shape[0] // batch_size)

        # Iterate over batches and store the predictions.
        for idx_batch in batches:
            x_pred[idx_batch] = np.array(
                self.proximal_step.chained_inference_steps(
                    jnp.array(adata.obsm[x_obsm][idx_batch]),
                    jnp.ones(len(idx_batch)) / len(idx_batch),
                    lambda u: self.potential.apply(self.params, u),
                    tau=tau,
                    n_steps=self.n_steps,
                )
            )

        # Return the predictions.
        return x_pred
