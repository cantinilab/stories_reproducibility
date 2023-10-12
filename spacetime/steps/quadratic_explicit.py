from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import grad, vmap
from .proximal_step import ProximalStep


class QuadraticExplicitStep(ProximalStep):
    """This class implements the explicit proximal step associated with the Gromov
    Wasserstein distance, i.e. the velocity field :math:`v` is obtained by solving a
    linear system which depends on the potential :math:`\\Phi`.

    Args:
        fused (float, optional): The weight of the fused term. Defaults to 1.0. The fused
            term corresponds to performing Fused Gromov-Wasserstein, i.e. adding a linear
            term in the optimal transport problem.
        cross (float, optional): The weight of the cross term. Defaults to 1.0. The cross
            term compares the x component and the spatial component of cells.
        straight (float, optional): The weight of the straight term. Defaults to 1.0. The
            straight term compares the x component of cells to itself, and the spatial
            component of cells to iself.
        dist_fun (Callable, optional): A distance function. Defaults to the squared
            Euclidean distance divided by two, such that grad(dist_fun)(x) = x.
    """

    def __init__(
        self,
        fused: float = 1.0,
        cross: float = 1.0,
        straight: float = 1.0,
        dist_fun: Callable = lambda u: 0.5 * jnp.linalg.norm(u) ** 2,
    ):
        self.fused = fused
        self.cross = cross
        self.straight = straight
        self.dist_fun = dist_fun

    def inference_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray]:
        """Performs a quadratic explicit step on the input distribution and returns the
        updated distribution, given a potential function. Unlike in a linear step, the
        spatial coordinates may change in a quadratic proximal step.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2).
        """

        # Compute the velocity vector field v, which involves solving a linear system.
        # v is the concatenation of the velocity in the x and space components.
        v = self.inverse_partial_Q(x, space, potential_fun)

        # Return the next timepoint, splitting v into it's x and space components.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]

    def training_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray]:
        """Performs a quadratic explicit step on the input distribution and returns the
        updated distribution, given a potential network. Unlike in a linear step, the
        spatial coordinates may change in a quadratic proximal step. This function is
        different from inference_step because it takes a potential network as input.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            space (jnp.ndarray): The space variable of size (batch_size, 2)
            potential_network (nn.Module): A potential network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Tuple[jnp.ndarray]: The updated distribution of size (batch_size, n_dims) and
            its spatial coordinates of size (batch_size, 2).
        """

        # Turn the potential network into a function.
        potential_fun = lambda u: potential_network.apply(potential_params, u)

        # Then simply apply the inference step since it's differentiable.
        return self.inference_step(x, space, potential_fun, tau)

    def grad_dist_fun(self, u: jnp.ndarray) -> jnp.ndarray:
        """Applies the gradient of the distance function to u_i - u_j.

        Args:
            u (jnp.ndarray): The input array of shape (n, d).

        Returns:
            jnp.ndarray: The output array of shape (n, n, d)."""

        n, d = u.shape
        return grad(self.dist_fun)(u.reshape(n, 1, d) - u.reshape(1, n, d))

    def kernel(self, x: jnp.ndarray, space: jnp.ndarray) -> jnp.ndarray:
        """Computes the kernel matrix associated with the quadratic proximal step.

        Args:
            x (jnp.ndarray): The input distribution of size (n, n_dims)
            space (jnp.ndarray): The space variable of size (n, 2)

        Returns:
            jnp.ndarray: The kernel matrix of size (n, n_dims + 2, n, n_dims + 2)."""

        # Compute the gradient of dist_fun applied to x_i - x_j and space_i - space_j.
        g_xx = self.grad_dist_fun(x)
        g_ss = self.grad_dist_fun(space)
        grad_dist_fun = jnp.concatenate((g_xx, g_ss), axis=2)

        # Compute the outer product of the gradient of dist_fun.
        K = jnp.einsum("ijk,ijl->ijkl", grad_dist_fun, grad_dist_fun)

        # Define a mask to apply to the outer product, based on the straight and cross
        # values. This is a block matrix of shape (d1 + d2, d1 + d2), indexed by k,l.
        d1, d2 = x.shape[1], space.shape[1]
        u_xx = self.straight * jnp.ones((d1, d1))
        u_xs = self.cross * jnp.ones((d1, d2))
        u_ss = self.straight * jnp.ones((d2, d2))
        u_sx = self.cross * jnp.ones((d2, d1))
        u = jnp.block([[u_xx, u_xs], [u_sx, u_ss]])

        return jnp.einsum("ijkl,kl->ijkl", K, u)

    def inverse_partial_Q(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_fun: Callable,
    ) -> jnp.ndarray:
        """Computes the velocity vector field v, which involves solving a linear system.
        v is the concatenation of the velocity in the x and space components.

        Args:
            x (jnp.ndarray): The input distribution of size (n, n_dims)
            space (jnp.ndarray): The space variable of size (n, 2)
            potential_fun (Callable): A scalar potential function.

        Returns:
            jnp.ndarray: The velocity vector field of size (n, n_dims + 2)."""

        # To compute the velocity vector field v, we will need to solve a linear system.
        # This system involves many indices, so we will use einsum to clarify things.

        # First, let use define a shorthand for the number of cells and dimensions.
        n, dx = x.shape
        n, ds = space.shape

        # We want to solve a linear system of the form A*v = b.
        # The definition of A involves an outer product of the gradient of dist_fun.
        # Let us recall that dist_fun defines a distance between cells.
        # Let us call K this tensor of shape (n, n, dx + ds, dx + ds)
        K = self.kernel(x, space)

        # Let us denote A*v the LHS of the linear system.
        # A is indexed by i,j,k,l and v is indexed by j,l.

        # Notice the index p, a change a variable needed to sum over the index i of v.
        # The formula for A thus mirrors the v_il - v_jl term in the definition of the
        # operator \partial Q_mu (v) that we are trying to invert. The 1/n is because
        # we are using the uniform distribution weights for the cells.
        A = jnp.einsum("ij,ipkl->ijkl", jnp.eye(n), K) / n - K / n

        # Adding a fused linear term to Gromov-Wasserstein corresponds to adding the
        # identity function to A, which corresponds to the following tensor.
        A += self.fused * jnp.einsum("ij,kl->ijkl", jnp.eye(n), jnp.eye(dx + ds))

        # b is the RHS of the linear system. b is indexed by i,k.
        # It has an x component which depends on a potential, and a zero space component.
        b = -vmap(grad(potential_fun))(x)
        b = jnp.concatenate((b, jnp.zeros((n, ds))), axis=1)

        # Solve the linear system for v indexed by j,l.
        return jnp.linalg.tensorsolve(A, b, axes=(1, 3))
