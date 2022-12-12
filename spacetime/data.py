# Typing imports.
from typing import Iterable

# JAX imports.
import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Matplotlib imports.
import matplotlib.pyplot as plt

import numpy as np
import anndata as ad
import pandas as pd

# Define the Styblinksi potential.
def styblinski_potential(u: jnp.ndarray) -> callable:
    return jnp.sum(u**4 - 16 * u**2 + 5 * u) / 2


# Define the quadratic potential.
def quadratic_potential(u: jnp.ndarray) -> callable:
    return jnp.sum(u**2) / 2


# Get the drift function for some potential.
def get_drift(potential: callable):
    return vmap(grad(potential))


# Generate a sequence of populations using the Euler-Maruyama method.
def euler_maruyama(
    potential: callable,
    u0: jnp.ndarray,
    dt: float,
    sd: float,
    n_steps: int,
    key: jnp.ndarray = random.PRNGKey(0),
) -> Iterable:

    # Get the drift function from the potential.
    drift = get_drift(potential)

    # Set the initial population.
    u = u0

    # Iterate over the number of steps.
    for _ in range(n_steps):

        # Perform the Euler-Maruyama step.
        u = u - drift(u) * dt + jnp.sqrt(dt) * sd * random.normal(key, u.shape)

        yield u  # Yield the new population.

# Generate an AnnData on a 2D grid, following a Styblinski potential.
def generate_adata():
    
    # Define the hyperparameters.
    key = random.PRNGKey(42)
    n, d_space, d_embedding = 25, 2, 2
    dt, n_steps = 0.06, 6
    sd = 0.5
    u0 = random.normal(key, (n, d_embedding), dtype=jnp.float32)

    # Generate an iterator for the populations.
    populations = euler_maruyama(
        potential=styblinski_potential,
        u0=u0,
        dt=dt,
        sd=sd,
        n_steps=n_steps,
        key=key,
    )

    # Save the populations to a list. 
    populations = jnp.stack(list(populations))

    # Get the first and second coordinates.
    aa = np.array(populations[-1, :, 0])
    bb = np.array(populations[-1, :, 1])

    # Separate cells into clusters.
    clusters = np.zeros(len(aa), dtype=int)
    clusters[(aa < 0) & (bb < 0)] = 0
    clusters[(aa < 0) & (bb > 0)] = 1
    clusters[(aa > 0) & (bb < 0)] = 2
    clusters[(aa > 0) & (bb > 0)] = 3

    # Sort the cell by cluster.
    idx = np.argsort(clusters)

    # Initialize the space, time and embedding.
    space = np.zeros((n_steps, n, d_space))
    embedding = np.zeros((n_steps, n, d_embedding), dtype=np.float32)
    time = np.zeros((n_steps, n), dtype=int)
    xx, yy = jnp.meshgrid(np.arange(5), np.arange(5))

    # Iterate over the number of steps to define space, time and embedding.
    for i in range(n_steps):
        space[i] = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
        embedding[i] = populations[i, idx, :]
        time[i] = np.ones(n)*i

    # Create the AnnData object.
    adata = ad.AnnData(embedding.reshape(-1, d_embedding))
    adata.uns["spatial"] = {"spatial_key": "spatial"}
    adata.obsm["spatial"] = space.reshape(-1, d_space)
    adata.obs["cluster"] = pd.Categorical(np.concatenate([clusters[idx]]*n_steps))
    adata.obs["time"] = pd.Categorical(time.reshape(-1))

    # Return the AnnData object.
    return adata
