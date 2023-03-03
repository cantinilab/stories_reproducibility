from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import grad, vmap
from space_time import explicit_steps


class QuadraticExplicitStep(explicit_steps.ExplicitStep):
    """Explicit proximal step with the Gromov Wasserstein distance."""

    def grad_phi(self, x: jnp.array, phi: Callable) -> jnp.array:
        """Given a function $\phi : \mathbb R^d \to \mathbb R$,
        return $\nabla \phi(x_i - x_j)$ as a jnp.array of size (N, N, d).

        Args:
            x (jnp.array): The input data, size (N, d).
            phi (Callable): The function $\phi$, typically a squared euclidean norm.

        Returns:
            jnp.array:  $\nabla \phi(x_i - x_j)$, a jnp.array of size (N, N, d).
        """
        n, d = x.shape
        return grad(phi)(x.reshape(n, 1, d) - x.reshape(1, n, d))

    def kernel(self, x: jnp.array, phi: Callable) -> jnp.array:
        """Given a function $\phi : \mathbb R^d \to \mathbb R$, return the kernel
        $\Phi(x_i, x_j) = \nabla \phi(x_i - x_j)^T \nabla \phi(x_i - x_j)$ as a
        jnp.array of size (N, N, d, d).

        Args:
            x (jnp.array): The input data, size (N, d).
            phi (Callable): The function $\phi$, typically a squared euclidean norm.

        Returns:
            jnp.array: The kernel $\Phi(x_i, x_j)$, a jnp.array of size (N, N, d, d).
        """
        g = self.grad_phi(x, phi)
        n, _, d = g.shape
        return g.reshape(n, n, d, 1) @ g.reshape(n, n, 1, d)

    def inverse_partial_Q(
        self,
        x: jnp.array,
        potential_fun: Callable,
        phi: Callable,
        fused: float,
    ) -> jnp.array:
        """Given a function $\phi : \mathbb R^d \to \mathbb R$ and the potential functions,
        return the velocity vector field v as a jnp.array of size (N, d).

        Args:
            x (jnp.array): The input data, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            phi (Callable): The function $\phi$, typically a squared euclidean norm.
            fused (float): Scaling parameter for the linear term (fused GW).

        Returns:
            jnp.array: The velocity vector field v as a jnp.array of size (N, d).
        """

        n, d = x.shape
        Phi = self.kernel(x, phi)

        A = jnp.einsum("ij,ipkl->ijkl", jnp.eye(n), Phi) - Phi
        A += fused * jnp.einsum("ij,kl->ijkl", jnp.eye(n), jnp.eye(d))
        b = -vmap(grad(potential_fun))(x)

        return jnp.linalg.tensorsolve(A, b, axes=(1, 3))

    def inference_step(
        self,
        x: jnp.array,
        potential_fun: Callable,
        tau: float,
        fused: float = 1,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Explicit proximal step using the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): The input data, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float): Time step.
            fused (float, optional): Scaling parameter for the linear term (fused GW).
            phi (Callable, optional): The function $\phi$, typically a squared euclidean norm. Defaults to lambda u:jnp.linalg.norm(u)**2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """
        # Compute the velocity vector field v.
        v = self.inverse_partial_Q(x, potential_fun, phi, fused)

        # Return the next timepoint.
        return x + tau * v

    def training_step(
        self,
        x: jnp.array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        fused: float = 1,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
        a: jnp.ndarray = None,
    ) -> jnp.array:
        """Explicit proximal step using the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): The input data, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float): Time step.
            fused (float, optional): Scaling parameter for the linear term (fused GW).
            phi (Callable, optional): The function $\phi$, typically a squared euclidean norm. Defaults to lambda u:jnp.linalg.norm(u)**2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """
        # Compute the velocity vector field v.
        potential_fun = lambda u: potential_network.apply(potential_params, u)
        v = self.inverse_partial_Q(x, potential_fun, phi, fused)

        # Return the next timepoint.
        return x + tau * v
