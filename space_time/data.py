# Typing imports.
from typing import Iterable

# JAX imports.
import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Matplotlib imports.
import matplotlib.pyplot as plt

# Define the Styblinksi potential.
def styblinski_potential(u: jnp.ndarray) -> callable:
    return jnp.sum(u**4 - 16 * u**2 + 5 * u) / 2


# Define the quadratic potential.
def quadratic_potential(u: jnp.ndarray) -> callable:
    return jnp.sum(u**2) / 2


# Get the drift function for some potential.
def get_drift(potential: callable):
    return -vmap(grad(potential))


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
        u = u + drift(u) * dt + jnp.sqrt(dt) * sd * random.normal(key, u.shape)

        yield u  # Yield the new population.
