from typing import Callable, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jaxopt
import optax
import wandb
from jax import grad
from space_time import implicit_steps


class MongeQuadraticImplicitStep(implicit_steps.ImplicitStep):
    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
        wb: bool = False,
        fused: float = 1.0,
        cross: float = 1.0,
        straight: float = 1.0,
    ):
        """Implicit proximal step with the squared and debiased Fused Gromov-Wasserstein
        distance. We assume the transport plan is the identity (each cell is
        mapped to itself).

        Args:
            maxiter (int, optional): The maximum number of iterations for the optimization loop. Defaults to 100.
            implicit_diff (bool, optional): Whether to differentiate implicitly through the optimization loop. Defaults to True.
            wb (bool, optional): Whether to log the proximal cost using wandb. Defaults to False.
            fused (float, optional): The linear term. Defaults to 1.0.
            cross (float, optional): The cross-modalities term. Defaults to 1.0.
            straight (float, optional): The intra-modalty term. Defaults to 1.0.
        """
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.wb = wb
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

    def kernel(self, x: jnp.array, space: jnp.array, phi: Callable) -> jnp.array:
        """Given a function $\phi : \mathbb R^d \to \mathbb R$, return the kernel
        $\Phi(u_i, u_j) = \nabla \phi(u_i - u_j)^T \nabla \phi(u_i - u_j)$ as a
        jnp.array of size (N, N, d1 + d2, d1 + d2).

        Args:
            x (jnp.array): The omics coordiantes, size (N, d1).
            space (jnp.array): The spatial coordinates, size (N, d2).
            phi (Callable): The function $\phi$, typically a squared euclidean norm.

        Returns:
            jnp.array: The kernel $\Phi(u_i, u_j)$, a jnp.array of size (N, N, d1 + d2, d1 + d2).
        """

        n, d1 = x.shape
        n, d2 = space.shape
        d = d1 + d2

        # Compute the gradient of phi with respect to x and space, for the differences
        # between each point in x and space, and then concatenate them.
        g_xx = self.grad_phi(x, phi)
        g_ss = self.grad_phi(space, phi)
        grad_phi = jnp.concatenate((g_xx, g_ss), axis=2)

        # Compute the kernel Phi which is an outer product of these gradients.
        k = grad_phi.reshape(n, n, d, 1) @ grad_phi.reshape(n, n, 1, d)

        # Compute a mask for the kernel with scalings for the cross and straight terms.
        u_xx = self.straight * jnp.ones((d1, d1))
        u_xs = self.cross * jnp.ones((d1, d2))
        u_ss = self.straight * jnp.ones((d2, d2))
        u_sx = self.cross * jnp.ones((d2, d1))
        u = jnp.block([[u_xx, u_xs], [u_sx, u_ss]])

        # Apply the mask to the kernel and return the result.
        return jnp.einsum("ijkl,kl->ijkl", k, u)

    def inference_step(
        self,
        x: jnp.array,
        space: jnp.array,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
        phi: Callable = lambda u: jnp.linalg.norm(u) ** 2,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the squared Gromov-Wasserstein distance. This
        "inference step" takes the potential function as an argument, and returns
        the new omics and spatial coordinates. As opposed to a linear step, the space
        is updated.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_fun (Callable): The potential function.
            tau (float): The proximal step size.
            phi (_type_, optional): A function characterizing the distance used in the
            omics and physical spaces. Defaults to a square norm.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics and spatial coordinates.
        """

        def proximal_cost(v, inner_x, inner_phi, inner_a):
            """A helper function to compute the proximal cost."""

            # Compute the velocity difference between each point in x and space.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)

            # Compute the quadratic term.
            gw_term = jnp.einsum(
                "i,j,ijk,ijkl,ijl->",
                inner_a,
                inner_a,
                v_diff,
                inner_phi,
                v_diff,
            )

            # Add the fused term.
            gw_term += self.fused * jnp.sum(inner_a.reshape(-1, 1) * v**2)

            # Compute the potential term, which excluded the spatial component.
            y = inner_x + tau * v[:, : inner_x.shape[1]]
            potential_term = jnp.sum(inner_a * potential_fun(y))

            # Return the total proximal cost
            return potential_term + tau * gw_term

        # Compute the kernel.
        Phi = self.kernel(x, space, phi)

        # If we're logging things, log the rank of the linear operator \partial Q.
        # If fused is greater than 0, then it should be full rank.
        if self.wb:

            # First, compute the oprator \partial Q.
            n, _, d, _ = Phi.shape
            partial_Q = jnp.einsum("ij,ipkl->ijkl", jnp.eye(n), Phi) - Phi

            # Then, reshape it to be a matrix.
            new_shape = n * d, n * d
            reshaped_partial_Q = jnp.transpose(partial_Q, (0, 2, 1, 3))
            reshaped_partial_Q = reshaped_partial_Q.reshape(new_shape)

            # Finally, log the rank of the matrix.
            wandb.log({"rank_partial_Q": jnp.linalg.matrix_rank(reshaped_partial_Q)})

        # Define the optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the gradient descent, logging the proximal cost if we're using wandb.
        init_v = jnp.zeros((x.shape[0], x.shape[1] + space.shape[1]))
        v, state = init_v, opt.init_state(init_v, inner_x=x, inner_phi=Phi, inner_a=a)
        for _ in range(self.maxiter):
            v, state = opt.update(v, state, inner_x=x, inner_phi=Phi, inner_a=a)
            if self.wb:
                wandb.log({"proximal_cost": state.error})

        # Compute the new omics and spatial coordinates.
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implicit proximal step with the squared Gromov-Wasserstein distance. This
        "training step" takes the potential network and parameters as arguments, and
        returns the new omics and spatial coordinates. As opposed to a linear step,
        the space is updated.

        Args:
            x (jnp.array): The omics coordinates.
            space (jnp.array): The spatial coordinates.
            a (jnp.ndarray): The marginal weights.
            potential_network (nn.Module): The potential network.
            potential_params (optax.Params): The potential network's parameters.
            tau (float): The proximal step size.
            phi (Callable, optional): A function characterizing the distance used in the
            omics and physical spaces. Defaults to a square norm.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The new omics and spatial coordinates.
        """

        def proximal_cost(v, inner_x, inner_potential_params, inner_phi, inner_a):
            """A helper function to compute the proximal cost."""

            # Compute the velocity difference between each point in x and space.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)

            # Compute the quadratic term.
            gw_term = jnp.einsum(
                "i,j,ijk,ijkl,ijl->",
                inner_a,
                inner_a,
                v_diff,
                inner_phi,
                v_diff,
            )

            # Add the fused term.
            gw_term += self.fused * jnp.sum(inner_a.reshape(-1, 1) * v**2)

            # Compute the potential term, which excludes the spatial component.
            y = inner_x + tau * v[:, : inner_x.shape[1]]
            fun = lambda u: potential_network.apply(inner_potential_params, u)
            potential_term = jnp.sum(inner_a * fun(y))

            # Return the proximal cost
            return potential_term + tau * gw_term

        # Compute the kernel.
        Phi = self.kernel(x, space, phi)

        # Define the optimizer.
        opt = jaxopt.LBFGS(
            fun=proximal_cost,
            maxiter=self.maxiter,
            implicit_diff=self.implicit_diff,
        )

        # Run the optimization loop.
        init_v = jnp.zeros((x.shape[0], x.shape[1] + space.shape[1]))
        v, state = opt.run(
            init_v,
            inner_x=x,
            inner_phi=Phi,
            inner_potential_params=potential_params,
            inner_a=a,
        )

        # Compute the new omics and spatial coordinates.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]
