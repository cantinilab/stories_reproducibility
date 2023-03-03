from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


def styblinski_potential(u: jnp.ndarray) -> jnp.ndarray:
    """The Styblinski potential has 4 minima.

    Args:
        u (jnp.ndarray): A 2D point or a pointcloud of size (N, 2).

    Returns:
        jnp.ndarray: _description_
    """
    return jnp.sum(u**4 - 16 * u**2 + 5 * u) / 2


class MLPPotential(nn.Module):
    act_fn: callable = nn.softplus
    features: Sequence[int] = (32, 32)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features[0])(x)
        x = self.act_fn(x)
        x = nn.Dense(features=self.features[1])(x)
        x = self.act_fn(x)
        x = nn.Dense(features=1)(x)
        return x.sum(-1)
