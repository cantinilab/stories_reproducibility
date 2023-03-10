from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
from jax import grad
from space_time import implicit_steps


class MongeQuadraticImplicitStep(implicit_steps.ImplicitStep):
    """Implicit proximal step with the Gromov-Wasserstein distance, learning the velocity v."""

    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
    ):
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff

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

        return grad_phi.reshape(n, n, d, 1) @ grad_phi.reshape(n, n, 1, d)

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
        fused: float = 1.0,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
    ) -> jnp.array:
        """Implicit proximal step with the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.
            n_iter (int, optional): The number of gradient descent steps. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 5e-2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        def proximal_cost(v, inner_x, inner_phi, inner_a):

            # Compute the quadratic cost.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)
            cost = tau**2 * jnp.einsum(
                "i,j,ijk,ijkl,ijl->",
                inner_a,
                inner_a,
                v_diff,
                inner_phi,
                v_diff,
            )
            cost += fused * tau**2 * jnp.sum(inner_a * v**2)

            # Return the proximal cost
            y = inner_x + tau * v[:, : inner_x.shape[1]]
            return tau * jnp.sum(potential_fun(y)) + cost

        Phi = self.kernel(x, space, phi)

        gd = jaxopt.GradientDescent(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )
        v, _ = gd.run(
            jnp.zeros_like((x.shape[0], x.shape[1] + space.shape[1])),
            inner_x=x,
            inner_phi=Phi,
            inner_a=a,
        )
        y = x + tau * v[:, : x.shape[1]]
        return y, space

    def training_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        fused: float = 1.0,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
    ) -> jnp.array:
        """Implicit proximal step with the Gromov-Wasserstein distance.

        Args:
            x (jnp.array): Input distribution, size (N, d).
            potential_fun (Callable): A potential function, taking a pointcloud of size (N, d) as input.
            tau (float, optional): Time step.
            n_iter (int, optional): The number of gradient descent steps. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 5e-2.

        Returns:
            jnp.array: The output distribution, size (N, d)
        """

        def proximal_cost(v, inner_x, inner_potential_params, inner_phi, inner_a):

            # Compute the quadratic cost.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)
            cost = tau**2 * jnp.einsum(
                "i,j,ijk,ijkl,ijl->",
                inner_a,
                inner_a,
                v_diff,
                inner_phi,
                v_diff,
            )
            cost += fused * tau**2 * jnp.sum(inner_a * v**2)

            # Return the proximal cost
            y = inner_x + tau * v[:, : inner_x.shape[0]]
            potential_fun = lambda u: potential_network.apply(inner_potential_params, u)
            return tau * jnp.sum(potential_fun(y)) + cost

        Phi = self.kernel(x, space, phi)

        gd = jaxopt.GradientDescent(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )
        v, _ = gd.run(
            jnp.zeros_like((x.shape[0], x.shape[1] + space.shape[1])),
            inner_x=x,
            inner_potential_params=potential_params,
            inner_phi=Phi,
            inner_a=a,
        )
        y = x + tau * v[:, : x.shape[0]]
        return y, space
