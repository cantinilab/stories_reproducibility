from typing import Sequence

import flax.linen as nn
from flax.linen.activation import softplus
from flax.linen.initializers import he_uniform, zeros_init, constant
import jax.numpy as jnp


class MLPPotential(nn.Module):
    """
    This class defines a simple multi-layer perceptron (MLP) potential which takes a
    batch of cells and returns a batch of scalars. The activation function is a softplus
    function, which is a smooth approximation to the rectified linear unit (ReLU). This
    makes the gradient of the potential twice differentiable wrt input. We recommend
    trying gelu as an activation function as well.

    Args:
        features: A sequence of integers specifying the number of hidden units in each
        layer of the MLP. The length of this sequence determines the number of hidden
        layers in the MLP.
        activation: Activation function to use in the hidden layers.

    """

    features: Sequence[int] = (32, 32)  # Default to two hidden layers
    activation: callable = softplus  # Default to softplus activation

    @nn.compact
    def __call__(self, x):
        # Iterate over hidden layers
        for feature in self.features:
            x = nn.Dense(kernel_init=he_uniform(), features=feature)(x)
            x = self.activation(x)

        # Define the output layer. Since we care about the gradient of the potential,
        # we do not include a bias term.
        x = nn.Dense(kernel_init=zeros_init(), features=1, use_bias=False)(x)

        # Squeeze the output to remove the last dimension
        return x.squeeze()