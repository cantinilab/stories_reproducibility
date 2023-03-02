from typing import Callable, Iterable, List, Union

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
        input_distributions: List[jnp.ndarray],  # (time, cells, dims)
        optimizer: GradientTransformation = optax.chain(optax.zero_nans(), optax.adam(1e-2)),
        n_iter: int = 1_000,
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

        def loss(params: optax.Params, batch: jnp.ndarray, batch_marginals: jnp.ndarray) -> jnp.ndarray:
            # TODO: Make teacher forcing optional.
            # TODO: move epsilon and debias here.

            # We initialize the Sinkhorn solver outside of the loop.
            sinkhorn_solver = sinkhorn.Sinkhorn()

            def _through_time(carry, t):
                """Helper function to compute the loss at a given timepoint."""

                _batch, _batch_marginals = carry

                # Predict the timepoint t+1 using the proximal step.
                pred = self.proximal_step.training_step(
                    _batch[t],  # Batch at time t
                    potential_network = self.potential,
                    potential_params = params,
                    tau = self.tau,  # Time step
                    a = _batch_marginals[t],  # Marginal at time t
                )

                # Compute the loss between the predicted and the true next time step.
                geom_xy = PointCloud(pred, _batch[t + 1], epsilon=self.epsilon)
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
                return (_batch, _batch_marginals), sink_loss

            # Iterate through timepoints efficiently. sink_loss becomes a 1-D array.
            _, sink_loss = jax.lax.scan(
                _through_time,
                (batch, batch_marginals),
                jnp.arange(len(batch) - 1),  # All timepoints except the last one
            )

            # Sum the losses over all timepoints.
            return jnp.sum(sink_loss)

        @jax.jit
        def step(
            params: optax.Params,
            opt_state: optax.OptState,
            batch: jnp.ndarray,
            batch_marginals: jnp.ndarray,
        ):
            """Jitted helper function to perform a single optimization step."""

            # Given a batch, compute the value and grad of the loss for current parameters.
            loss_value, grads = jax.value_and_grad(loss)(params, batch, batch_marginals)

            # Using the computed gradients, update the optimizer.
            updates, opt_state = optimizer.update(grads, opt_state, params)

            # Perform an optimization step and update the parameters.
            params = optax.apply_updates(params, updates)

            # Return the updated parameters, optimizer state and loss.
            return params, opt_state, loss_value

        # Define the batch_size.
        if batch_size is None:
            batch_size = min([len(batch) for batch in input_distributions])
        
        # Pad the input distributions with zeros if they are smaller than batch_size.
        padded_distributions = []
        padded_marginals = []
        for timepoint in input_distributions:
            if len(timepoint) < batch_size:

                # Pad the timepoint with zeros.
                padded_distributions.append(jnp.pad(
                    timepoint,
                    ((0, batch_size - len(timepoint)), (0, 0)),
                    mode="mean",
                ))

                # Pad the marginals with zeros.
                padded_marginals.append(jnp.pad(
                    jnp.ones(timepoint.shape[0])/timepoint.shape[0],
                    (0, batch_size - len(timepoint)),
                    mode="constant",
                ))
            else:
                padded_distributions.append(timepoint[:batch_size])
                padded_marginals.append(jnp.ones(batch_size)/batch_size)

        print(padded_distributions)
        print(padded_marginals)

        # Initialize the parameters of the potential function.
        init_key, batch_key = jax.random.split(key)
        self.params = self.potential.init(init_key, padded_distributions[0])

        # Initialize the optimizer.
        opt_state = optimizer.init(self.params)

        # Sample a batch of cells from each timepoint.
        # idx_list = []
        # for timepoint in input_distributions:
        #     timepoint_key, batch_key = jax.random.split(batch_key)
        #     idx_timepoint = jax.random.choice(
        #         timepoint_key,
        #         len(timepoint),
        #         shape=(batch_size,),
        #         replace=False,
        #     )
        #     idx_list.append(idx_timepoint)
        
        # batch = jnp.stack([timepoint[idx] for timepoint, idx in zip(input_distributions, idx_list)])
        # batch_marginals = jnp.stack([timepoint[idx]/timepoint[idx].sum() for timepoint, idx in zip(input_marginals, idx_list)])

        # Iterate through the training loop.
        pbar = tqdm(range(n_iter))
        for _ in pbar:

            # Perform an optimization step.
            self.params, opt_state, loss_value = step(
                self.params,
                opt_state=opt_state,
                batch=jnp.stack(padded_distributions),
                batch_marginals=jnp.stack(padded_marginals),
            )

            pbar.set_postfix({"loss": loss_value})
    
    def fit_adata(
        self,
        adata: ad.AnnData,
        time_obs: str,
        obsm: str = "pca",
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
            input_distributions.append(x)

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
        obsm: str = "pca",
        optimizer: GradientTransformation = optax.adam(learning_rate=1e-3),
        n_iter: int = 1_000,
        dump_every: int = 100,
        key: PRNGKey = PRNGKey(0),
    ) -> ad.AnnData:
        """Fit the model to an AnnData object and transform the latest timepoint."""
        
        self.fit_adata(
            adata=adata,
            time_obs=time_obs,
            obsm=obsm,
            optimizer=optimizer,
            n_iter=n_iter,
            dump_every=dump_every,
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
