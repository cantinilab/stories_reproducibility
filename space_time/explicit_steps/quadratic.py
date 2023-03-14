from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import grad, vmap
from space_time import explicit_steps


class QuadraticExplicitStep(explicit_steps.ExplicitStep):
    """Explicit proximal step with the Gromov Wasserstein distance."""

    def __init__(
        self,
        fused: float = 1.0,
        cross: float = 1.0,
        straight: float = 1.0,
    ):
        self.fused = fused
        self.cross = cross
        self.straight = straight

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

    def kernel(
        self,
        x: jnp.array,
        space: jnp.array,
        phi: Callable,
    ) -> jnp.array:
        """Given a function $\phi : \mathbb R^d \to \mathbb R$, return the kernel
        $\Phi(x_i, x_j) = \nabla \phi(x_i - x_j)^T \nabla \phi(x_i - x_j)$ as a
        jnp.array of size (N, N, d, d).

        Args:
            x (jnp.array): The input data, size (N, d).
            phi (Callable): The function $\phi$, typically a squared euclidean norm.

        Returns:
            jnp.array: The kernel $\Phi(x_i, x_j)$, a jnp.array of size (N, N, d, d).
        """

        g_xx = self.grad_phi(x, phi)
        g_ss = self.grad_phi(space, phi)
        grad_phi = jnp.concatenate((g_xx, g_ss), axis=2)

        n, _, d = grad_phi.shape
        k = grad_phi.reshape(n, n, d, 1) @ grad_phi.reshape(n, n, 1, d)

        d1, d2 = x.shape[1], space.shape[1]
        u_xx = self.straight * jnp.ones((d1, d1))
        u_xs = self.cross * jnp.ones((d1, d2))
        u_ss = self.straight * jnp.ones((d2, d2))
        u_sx = self.cross * jnp.ones((d2, d1))

        u = jnp.block([[u_xx, u_xs], [u_sx, u_ss]])

        return jnp.einsum("ijkl,kl->ijkl", k, u)

    def inverse_partial_Q(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
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

        n, dx = x.shape
        n, ds = space.shape

        # Compute the outer product of gradients.
        Phi = self.kernel(x, space, phi)

        # A*v is the LHS of the linear system.
        A = jnp.einsum("p,ij,ipkl->ijkl", a, jnp.eye(n), Phi)
        A -= jnp.einsum("j,ijkl->ijkl", a, Phi)
        A += fused * jnp.einsum("ij,kl->ijkl", jnp.eye(n), jnp.eye(dx + ds))

        # b is RHS of the linear system.
        b = -vmap(grad(potential_fun))(x)
        b = jnp.concatenate((b, jnp.zeros((n, ds))), axis=1)

        return jnp.linalg.tensorsolve(A, b, axes=(1, 3))

    def inference_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
    ) -> jnp.ndarray:
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
        v = self.inverse_partial_Q(x, space, a, potential_fun, phi, self.fused)

        # Return the next timepoint.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
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
        v = self.inverse_partial_Q(x, space, a, potential_fun, phi, self.fused)

        # Return the next timepoint.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]
