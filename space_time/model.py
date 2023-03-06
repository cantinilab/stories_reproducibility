from typing import Callable, Dict, Iterable, List, Union

import anndata as ad
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import PRNGKey
from optax import GradientTransformation
from space_time.explicit_steps import ExplicitStep
from space_time.implicit_steps import ImplicitStep
from space_time.potentials import MLPPotential
from tqdm import tqdm

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


class SpaceTime:
    def __init__(
        self,
        potential: nn.Module = MLPPotential(),
        proximal_step: Callable = Union[ExplicitStep, ImplicitStep],
        tau: float = 1e-2,
        debias: bool = True,
        epsilon: float = 0.05,
    ):
        """Multi-modal Fused Gromov-Wasserstein gradient flow model for spatio-temporal omics data.

        Args:
            potential (nn.Module, optional): The neural network representing the potential function. Defaults to MLPPotential().
            proximal_step (Callable, optional): The proximal step function to use. Defaults to explicit_wasserstein_proximal_training_step.
            tau (float, optional): The timestep. Defaults to 1e-2.
            debias (bool, optional): If True, use the Sinkhorn divergence instead of the entropic OT cost as a training loss. Defaults to True.
            epsilon (float, optional): Entropic regularization. Defaults to 0.05.
        """
        self.potential = potential
        self.tau = tau
        self.debias = debias
        self.epsilon = epsilon
        self.proximal_step = proximal_step

    def fit(
        self,
        input_distributions: List[Dict],  # (time, cells, dims)
        optimizer: GradientTransformation = optax.chain(
            optax.zero_nans(), optax.adam(1e-2)
        ),
        n_iter: int = 100,
        batch_iter: int = 100,
        batch_size: int = None,
        key: PRNGKey = PRNGKey(0),
    ):
        """Fit the model to some input data.

        Args:
            input_distributions (jnp.ndarray): The training data. Size (time, cells, dims).
            n_iter (int, optional): The number of iterations in the training loop. Defaults to 1_000.
            dump_every (int, optional): Print every few iterations. Defaults to 100.
            key (PRNGKey, optional): Used to initilize the model. Defaults to PRNGKey(0).
        """
        # TODO:  implement batching.
        # TODO:  implement tqdm.

        def loss(
            params: optax.Params,
            batch_x: jnp.ndarray,
            batch_space: jnp.ndarray,
            batch_marginals: jnp.ndarray,
        ) -> jnp.ndarray:
            # TODO: Make teacher forcing optional.
            # TODO: move epsilon and debias here.

            # We initialize the Sinkhorn solver outside of the loop.
            sinkhorn_solver = sinkhorn.Sinkhorn()

            def _through_time(carry, t):
                """Helper function to compute the loss at a given timepoint."""

                _batch_x, _batch_space, _batch_marginals = carry

                # Predict the timepoint t+1 using the proximal step.
                pred = self.proximal_step.training_step(
                    _batch_x[t],  # Batch at time t
                    _batch_space[t],  # Batch at time t
                    potential_network=self.potential,
                    potential_params=params,
                    tau=self.tau,  # Time step
                    a=_batch_marginals[t],  # Marginal at time t
                )

                # Compute the loss between the predicted and the true next time step.
                geom_xy = PointCloud(pred, _batch_x[t + 1], epsilon=self.epsilon)
                problem = linear_problem.LinearProblem(
                    geom_xy,
                    a=_batch_marginals[t],
                    b=_batch_marginals[t + 1],
                )
                sink_loss = sinkhorn_solver(problem).reg_ot_cost

                # If debiasing is enabled, compute the terms of the Sinkhorn divergence.
                # We only need the term xx, because the term yy is a constant.
                if self.debias:
                    geom_xx = PointCloud(pred, pred, epsilon=self.epsilon)
                    problem = linear_problem.LinearProblem(
                        geom_xx,
                        a=_batch_marginals[t],
                        b=_batch_marginals[t],
                    )
                    sink_loss -= 0.5 * sinkhorn_solver(problem).reg_ot_cost

                # Return the data for the next iteration and the current loss.
                # To remove teacher-forcing, this will have to be changed.
                return (_batch_x, _batch_space, _batch_marginals), sink_loss

            # Iterate through timepoints efficiently. sink_loss becomes a 1-D array.
            _, sink_loss = jax.lax.scan(
                _through_time,
                (batch_x, batch_space, batch_marginals),
                jnp.arange(len(batch_x) - 1),  # All timepoints except the last one
            )

            # Sum the losses over all timepoints.
            return jnp.sum(sink_loss)

        @jax.jit
        def step(
            params: optax.Params,
            opt_state: optax.OptState,
            batch_x: jnp.ndarray,
            batch_space: jnp.ndarray,
            batch_marginals: jnp.ndarray,
        ):
            """Jitted helper function to perform a single optimization step."""

            # Given a batch, compute the value and grad of the loss for current parameters.
            loss_value, grads = jax.value_and_grad(loss)(
                params, batch_x, batch_space, batch_marginals
            )

            # Using the computed gradients, update the optimizer.
            updates, opt_state = optimizer.update(grads, opt_state, params)

            # Perform an optimization step and update the parameters.
            params = optax.apply_updates(params, updates)

            # Return the updated parameters, optimizer state and loss.
            return params, opt_state, loss_value

        # Define the batch_size.
        if batch_size is None:
            batch_size = min([len(dist["x"]) for dist in input_distributions])

        # Pad the input distributions with zeros if they are smaller than batch_size.
        padded_x_distributions = []
        padded_space_distributions = []
        padded_marginals = []
        for timepoint in input_distributions:
            x, space = timepoint["x"], timepoint["space"]
            if len(x) < batch_size:

                # Pad the timepoint with zeros.
                padded_x_distributions.append(
                    jnp.pad(
                        x,
                        ((0, batch_size - len(x)), (0, 0)),
                        mode="mean",
                    )
                )

                padded_space_distributions.append(
                    jnp.pad(
                        space,
                        ((0, batch_size - len(space)), (0, 0)),
                        mode="mean",
                    )
                )

                # Pad the marginals with zeros.
                a = jnp.pad(
                    jnp.ones(x.shape[0]),
                    (0, batch_size - len(x)),
                    mode="constant",
                    constant_values=1e-6,
                )

                padded_marginals.append(a / a.sum())
            else:
                padded_x_distributions.append(x)
                padded_space_distributions.append(space)
                padded_marginals.append(jnp.ones(x.shape[0]) / x.shape[0])

        # Initialize the parameters of the potential function.
        init_key, batch_key = jax.random.split(key)
        self.params = self.potential.init(init_key, padded_x_distributions[0])

        # Initialize the optimizer.
        opt_state = optimizer.init(self.params)

        # Iterate through the training loop.
        pbar = tqdm(range(n_iter))
        for outer_it in pbar:

            # Sample a batch of cells from each timepoint.
            idx_list = []
            for timepoint in padded_x_distributions:
                timepoint_key, batch_key = jax.random.split(batch_key)
                idx_timepoint = jax.random.choice(
                    timepoint_key,
                    len(timepoint),
                    shape=(batch_size,),
                    replace=False,
                )
                idx_list.append(idx_timepoint)

            batch_x = jnp.stack(
                [
                    timepoint[idx]
                    for timepoint, idx in zip(padded_x_distributions, idx_list)
                ]
            )
            batch_space = jnp.stack(
                [
                    timepoint[idx]
                    for timepoint, idx in zip(padded_space_distributions, idx_list)
                ]
            )
            batch_marginals = jnp.stack(
                [
                    timepoint[idx] / timepoint[idx].sum()
                    for timepoint, idx in zip(padded_marginals, idx_list)
                ]
            )

            for batch_it in range(batch_iter):

                # Perform an optimization step.
                self.params, opt_state, loss_value = step(
                    self.params,
                    opt_state=opt_state,
                    batch_x=jnp.stack(batch_x),
                    batch_space=jnp.stack(batch_space),
                    batch_marginals=jnp.stack(batch_marginals),
                )

                pbar.set_postfix({"loss": loss_value})

    def fit_adata(
        self,
        adata: ad.AnnData,
        time_obs: str,
        obsm: str = "X_pca",
        space_obsm: str = "X_space",
        **kwargs,
    ) -> None:
        """Fit the model to an AnnData object."""

        # Check that we have a valid time observation.
        assert adata.obs[time_obs].dtype == int, "Time observations must be integers."
        timepoints = np.sort(np.unique(adata.obs[time_obs]))

        # Iterate through the timepoints and add the data to the list.
        input_distributions = []
        for t in timepoints:
            x = jnp.array(adata[adata.obs[time_obs] == t].obsm[obsm])
            space = jnp.array(adata[adata.obs[time_obs] == t].obsm[space_obsm])
            input_distributions.append(
                {
                    "x": x,
                    "space": space,
                }
            )

        # Fit the model to the input distributions.
        self.fit(
            input_distributions=input_distributions,
            **kwargs,
        )

    def transform_adata(self, adata: ad.AnnData) -> ad.AnnData:
        """Transform the latest timepoint of an AnnData object."""
        pass

    def fit_transform_adata(
        self,
        adata: ad.AnnData,
        time_obs: str,
        obsm: str = "X_pca",
        space_obsm: str = "X_space",
        optimizer: GradientTransformation = optax.adam(learning_rate=1e-3),
        n_iter: int = 1_000,
        key: PRNGKey = PRNGKey(0),
    ) -> ad.AnnData:
        """Fit the model to an AnnData object and transform the latest timepoint."""

        self.fit_adata(
            adata=adata,
            time_obs=time_obs,
            obsm=obsm,
            space_obsm=space_obsm,
            optimizer=optimizer,
            n_iter=n_iter,
            key=key,
        )

        return self.transform_adata(adata)

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transform a distribution using the learned potential.

        Args:
            x (jnp.ndarray): The distribution to transform. Size (cells, dims).

        Returns:
            jnp.ndarray: The transformed batch. Size (cells, dims).
        """
        return self.proximal_step.inference_step(
            x,
            lambda x: self.potential.apply(self.params, x),  # Potential function
            self.tau,  # Time step
        )

    def fit_transform(
        self,
        input_distributions: Iterable[jnp.ndarray],  # (time, cells, dims)
        optimizer: GradientTransformation = optax.adam(learning_rate=1e-3),
        n_iter: int = 1_000,
        dump_every: int = 100,
        key: PRNGKey = PRNGKey(0),
        **kwargs,
    ):
        """Fit the model to some input data with t timepoints and predict the t+1 timepoint.

        Args:
            input_distributions (jnp.ndarray): The training data. Size (time, cells, dims).
            n_iter (int, optional): The number of iterations in the training loop. Defaults to 1_000.
            dump_every (int, optional): Print every few iterations. Defaults to 100.
            key (PRNGKey, optional): Used to initilize the model. Defaults to PRNGKey(0).
        """

        # TODO:  implement batching.

        # First, fit the model.
        self.fit(input_distributions, optimizer, n_iter, dump_every, key, **kwargs)

        # Then, predict the next timepoint.
        return self.transform(input_distributions[-1])
