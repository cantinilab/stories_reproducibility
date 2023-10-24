from typing import Callable, Dict, Tuple

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

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.implicit_differentiation import ImplicitDiff
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein

from .steps.proximal_step import ProximalStep
from .tools import DataLoader


class SpaceTime:
    """Multi-modal Fused Gromov-Wasserstein gradient
    flow model for spatio-temporal omics data.

    Args:
        potential (nn.Module): The scalar potential neural network.
        proximal_step (ProximalStep): The proximal step to use.
        tau (float, optional): The time step. Defaults to 1.0.
        teacher_forcing (bool, optional): Whether to use teacher forcing.
        quadratic (bool, optional): Whether to use a Gromov-Wasserstein loss.
        epsilon (float, optional): The entropy parameter for the loss.
        log_callback (Callable, optional): A callback function to log optimization.
    """

    def __init__(
        self,
        potential: nn.Module,
        proximal_step: ProximalStep,
        tau: float = 1.0,
        teacher_forcing: bool = True,
        quadratic: bool = False,
        epsilon: float = 0.05,
        log_callback: Callable = None,
        fused_penalty: float = 2.0,
    ):
        # Fill in some fields.
        self.potential = potential
        self.tau = tau
        self.quadratic = quadratic
        self.epsilon = epsilon
        self.teacher_forcing = teacher_forcing
        self.log_callback = log_callback
        self.proximal_step = proximal_step
        self.fused_penalty = fused_penalty

        # Initialize the OTT solver (quadratic or linear).
        # TODO: check if it should be symmetric or not.
        kwds = {
            "threshold": 1e-2,
            "implicit_diff": ImplicitDiff(symmetric=True, precondition_fun=lambda x: x),
        }
        self.ott_solver = GromovWasserstein(**kwds) if quadratic else Sinkhorn(**kwds)

    def fit(
        self,
        adata: ad.AnnData,
        time_obs: str,
        x_obsm: str,
        space_obsm: str,
        optimizer: GradientTransformation = optax.adam(1e-2),
        max_epochs: int = 500,
        batch_size: int = 128,
        train_val_split: float = 0.8,
        min_delta: float = 1e-2,
        patience: float = 3,
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
            max_epochs (int, optional): The maximum number of epochs. Defaults to 500.
            batch_size (int, optional): The batch size. Defaults to 128.
            train_val_split (float, optional): The proportion of train in the split.
            min_delta (float, optional): The minimum delta for early stopping.
            patience (float, optional): The patience for early stopping.
            checkpoint_manager (CheckpointManager, optional): The
                checkpoint manager.
            key (KeyArray, optional): The random key. Defaults to PRNGKey(0).
        """

        # Initialize the statistics for logging.
        self.stats = []

        # Create a data loader for the AnnData object.
        dataloader = DataLoader(
            adata,
            time_obs=time_obs,
            x_obsm=x_obsm,
            space_obsm=space_obsm,
            batch_size=batch_size,
            train_val_split=train_val_split,
        )

        @jax.jit
        def loss_fn(params: optax.Params, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            """The loss function"""

            # Get the batch's x and space coordinates.
            batch_x, batch_space = batch["x"], batch["space"]

            # This is a helper function to compute the loss for a single timepoint.
            # We will chain this function over the timepoints using lax.scan.
            def _through_time(carry, t):
                # Unpack the carry, which contains the x and space across timepoints.
                _x, _space = carry

                # Predict the timepoint t+1 using the proximal step.
                pred_x = self.proximal_step.training_step(
                    _x[t], self.potential, params, self.tau
                )

                if not self.quadratic:
                    # Compute the Sinkhorn loss. This ignores the spatial component.
                    geom_xy = PointCloud(pred_x, _x[t + 1], epsilon=self.epsilon)
                    ot_loss = self.ott_solver(LinearProblem(geom_xy)).reg_ot_cost

                    # Debias the Sinkhorn loss with the xx term.
                    geom_bias = PointCloud(pred_x, pred_x, epsilon=self.epsilon)
                    ot_loss -= (
                        0.5 * self.ott_solver(LinearProblem(geom_bias)).reg_ot_cost
                    )

                    # Debias the Sinkhorn loss with the yy term.
                    geom_bias = PointCloud(_x[t + 1], _x[t + 1], epsilon=self.epsilon)
                    ot_loss -= (
                        0.5 * self.ott_solver(LinearProblem(geom_bias)).reg_ot_cost
                    )

                else:
                    # In case we want a Fused Gromov-Wasserstein loss, we should define
                    # geometries on space for xx, yy, and on genes for xy.
                    pointcloud_kwds = {
                        "epsilon": self.epsilon,
                        "scale_cost": "max_cost",
                    }
                    geom_xx = PointCloud(_space[t], _space[t], **pointcloud_kwds)
                    geom_yy = PointCloud(
                        _space[t + 1], _space[t + 1], **pointcloud_kwds
                    )
                    geom_xy = PointCloud(pred_x, _x[t + 1], epsilon=self.epsilon)

                    # The linear part of the loss operates on the gene coordinates.
                    # The quadratic part of the loss operates on the space coordinates.
                    fused_kwds = {"fused_penalty": self.fused_penalty}
                    problem = QuadraticProblem(geom_xx, geom_yy, geom_xy, **fused_kwds)
                    ot_loss = self.ott_solver(problem).reg_gw_cost

                    # The issue with Fused Gromov-Wasserstein is that it is biased, ie
                    # FGW(x, y) != 0 when x=y. We can compute instead the following:
                    #          FGW(x, y) - 0.5 * FGW(x, x) - 0.5 * FGW(y, y)
                    # As done in http://proceedings.mlr.press/v97/bunne19a/bunne19a.pdf

                    # Substracting 0.5 * FGW(x, x).
                    # The bias geometry operates on the gene coordinates.
                    geom_bias = PointCloud(pred_x, pred_x, epsilon=self.epsilon)
                    problem = QuadraticProblem(
                        geom_xx, geom_xx, geom_bias, **fused_kwds
                    )
                    ot_loss -= 0.5 * self.ott_solver(problem).reg_gw_cost

                    # Substracting 0.5 * FGW(y, y).
                    # The bias geometry operates on the gene coordinates.
                    geom_bias = PointCloud(_x[t + 1], _x[t + 1], epsilon=self.epsilon)
                    problem = QuadraticProblem(
                        geom_yy, geom_yy, geom_bias, **fused_kwds
                    )
                    ot_loss -= 0.5 * self.ott_solver(problem).reg_gw_cost

                # If no teacher-forcing, replace next observation with predicted
                _x = jax.lax.cond(
                    self.teacher_forcing,
                    lambda u: u,
                    lambda u: u.at[t + 1].set(pred_x),
                    _x,
                )
                _space = jax.lax.cond(
                    self.teacher_forcing,
                    lambda u: u,
                    lambda u: u.at[t + 1].set(_space[t]),
                    _space,
                )

                # Return the data for the next iteration and the current loss.
                return (_x, _space), ot_loss

            # Iterate through timepoints efficiently. ot_loss becomes a 1-D array.
            # Notice that we do not compute the loss for the last timepoint, because there
            # is no next observation to compare to.
            timepoints = jnp.arange(len(batch_x) - 1)
            _, ot_loss = jax.lax.scan(_through_time, (batch_x, batch_space), timepoints)

            # Sum the losses over all timepoints.
            return jnp.sum(ot_loss)

        # Define a jitted update function.
        @jax.jit
        def jitted_update(params, opt_state, batch):
            v, g = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = optimizer.update(g, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, v

        # Initialize the parameters of the potential function and of the optimizer.
        init_key, key = jax.random.split(key)
        dummy_x = jnp.ones((batch_size, dataloader.n_features))  # Used to infer sizes.
        self.params = self.potential.init(init_key, dummy_x)
        opt_state = optimizer.init(self.params)

        pbar = tqdm(range(1, max_epochs + 1))  # Progress bar.

        # Define the early stopping criterion.
        early_stop = EarlyStopping(min_delta=min_delta, patience=patience)

        # Split the cells ids into train and validation.
        split_key, key = jax.random.split(key)
        dataloader.make_train_val_split(split_key)

        # Get the number of iterations per epoch.
        # The number of batches is different for each timepoint, so take the max.
        iter_per_epoch_train = int(np.max(dataloader.n_batches_train))
        iter_per_epoch_val = int(np.max(dataloader.n_batches_val))

        # The training loop, iterating over epochs.
        for epoch in pbar:
            # Shuffle the data at the start of very epoch.
            shuffle_key, key = jax.random.split(key)
            dataloader.shuffle_train_val(shuffle_key)

            # Get a random list saying if the next batch should be train or val.
            # This is used to alternate between train and val batches.
            train_or_val_key, key = jax.random.split(key)
            train_or_val = jnp.concatenate(
                [jnp.ones(iter_per_epoch_train), jnp.zeros(iter_per_epoch_val)]
            )
            train_or_val = jax.random.permutation(train_or_val_key, train_or_val)

            # Iterate over the batches.
            train_losses, val_losses = [], []
            for is_train in train_or_val:
                if is_train == 1:
                    # If it is a train batch, update the parameters.
                    self.params, opt_state, train_loss = jitted_update(
                        self.params,
                        opt_state=opt_state,
                        batch=dataloader.next_train(),
                    )
                    # The save the loss.
                    train_losses.append(train_loss)
                else:
                    print("test")
                    # If it is a val batch, just save the loss.
                    val_losses.append(loss_fn(self.params, dataloader.next_val()))

            print("val losses", np.mean(val_losses))
            # Compute epoch statistics.
            epoch_stats = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "val_loss": 0 if iter_per_epoch_val == 0 else np.mean(val_losses),
            }
            self.stats.append(epoch_stats)

            # Update the progress bar.
            pbar.set_postfix(epoch_stats)

            # Log.
            if self.log_callback:
                # Log the epoch statistics.
                self.log_callback(epoch_stats)

                # Log the training losses for each minibatch.
                n_iters = len(train_losses) * (epoch - 1)
                for i, loss in enumerate(train_losses):
                    self.log_callback({"iter": n_iters + i, "iter_train_loss": loss})

                # Log the validation losses for each minibatch.
                n_iters = len(val_losses) * (epoch - 1)
                for i, loss in enumerate(val_losses):
                    self.log_callback({"iter": n_iters + i, "iter_val_loss": loss})

            # Save the parameters.
            update_metric = "train_loss" if iter_per_epoch_val == 0 else "val_loss"
            save_kwargs = {"save_args": orbax_utils.save_args_from_target(self.params)}
            if checkpoint_manager:
                checkpoint_manager.save(
                    step=epoch,
                    items=self.params,
                    save_kwargs=save_kwargs,
                    metrics={"loss": np.float64(epoch_stats[update_metric])},
                )

            # Check for early stopping.
            _, early_stop = early_stop.update(epoch_stats[update_metric])
            if early_stop.should_stop:
                print("Met early stopping criteria, breaking...")
                break

    def transform(
        self,
        adata: ad.AnnData,
        time_obs: str,
        x_obsm: str,
        batch_size: int = 128,
        key: PRNGKey = PRNGKey(0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Transform the latest timepoint of an AnnData object.

        Args:
            adata (ad.AnnData): The AnnData object to transform.
            time_obs (str): The name of the timepoint observation.
            x_obsm (str): The obsm field containing the data to transform.
            batch_size (int, optional): The batch size. Defaults to 128.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The x and spatial predictions.
        """

        # Get a list of timepoints, so that we can select the last one.
        assert adata.obs[time_obs].dtype == int, "Timepoints must be integer."
        timepoints = np.sort(np.unique(adata.obs[time_obs]))

        # Get the x and spatial data for the last timepoint.
        idx = adata.obs[time_obs] == timepoints[-1]
        x = adata[idx].obsm[x_obsm].copy()

        # Initialize the predictions.
        x_pred = np.zeros(x.shape)

        # Get a shuffled list of indices.
        idx = jnp.arange(x.shape[0])
        idx = jax.random.permutation(key, idx)
        idx = np.array(idx)

        # Iterate over the batches.
        for idx_batch in np.array_split(idx, x.shape[0] // batch_size):
            # Predict the batch's next state.
            x_pred_batch = self.proximal_step.inference_step(
                jnp.array(x[idx_batch]),
                lambda u: self.potential.apply(self.params, u),
                tau=self.tau,
            )

            # Put the batch's predictions in the obsm field.
            x_pred[idx_batch] = np.array(x_pred_batch)

        # Return the predictions.
        return x_pred
