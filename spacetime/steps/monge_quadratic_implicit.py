from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
import optax
from jax import grad
from .proximal_step import ProximalStep


class MongeQuadraticImplicitStep(ProximalStep):
    """This class implements the implicit proximal step associated with the Gromov
    Wasserstein distance, i.e. the velocity field :math:`v` is obtained by solving a
    linear system which depends on the potential :math:`\\Phi`.

    Args:
        maxiter (int, optional): The maximum number of iterations for the optimization
            problem. Defaults to 100.
        implicit_diff (bool, optional): Whether to use implicit differentiation. Defaults
            to True.
        tol (float, optional): The tolerance for the optimization problem. Defaults to
            1e-8.
        log_callback (Callable, optional): A callback function to log the optimization
            problem. Defaults to None.
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
        maxiter: int = 100,
        implicit_diff: bool = True,
        tol: float = 1e-8,
        log_callback: Callable = None,
        fused: float = 1.0,
        cross: float = 1.0,
        straight: float = 1.0,
        dist_fun: Callable = lambda u: 0.5 * jnp.linalg.norm(u) ** 2,
    ):
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.log_callback = log_callback
        self.fused = fused
        self.cross = cross
        self.straight = straight
        self.dist_fun = dist_fun

        self.opt_hyperparams = {
            "maxiter": maxiter,
            "implicit_diff": implicit_diff,
            "tol": tol,
        }

    def inference_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """This function computes the velocity field :math:`v` associated with the
        potential :math:`\\Phi` using the implicit proximal step and returns the
        updated x and spatial coordiantes.

        Args:
            x (jnp.ndarray): The x coordinates of the cells.
            space (jnp.ndarray): The spatial coordinates of the cells.
            potential_fun (Callable): A scalar potential function
            tau (float): The step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The updated x and spatial coordinates.
        """

        # Define a mask to apply to the outer product, based on the straight and cross
        # values. This is a block matrix of shape (d1 + d2, d1 + d2), indexed by k,l.
        d1, d2 = x.shape[1], space.shape[1]
        mask_xx = self.straight * jnp.ones((d1, d1))
        mask_xs = self.cross * jnp.ones((d1, d2))
        mask_ss = self.straight * jnp.ones((d2, d2))
        mask_sx = self.cross * jnp.ones((d2, d1))
        mask = jnp.block([[mask_xx, mask_xs], [mask_sx, mask_ss]])

        # Compute the gradient of the distance function applied to u_i - u_j.
        grad_dist_x = self.grad_dist_fun(x)
        grad_dist_space = self.grad_dist_fun(space)
        grad_dist = jnp.concatenate([grad_dist_x, grad_dist_space], axis=1)

        # Let us define a helper function to compute the proximal cost.
        def proximal_cost(v, inner_x, inner_grad_dist, inner_mask):

            # Compute the velocity difference between each point in x and space.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)

            # Compute the quadratic term.
            gw_term = jnp.einsum(
                "ijk,kl,ijl,ijl,ik->",
                inner_grad_dist,
                inner_mask,
                inner_grad_dist,
                v_diff,
                v,
            )
            gw_term /= n

            # Add the fused term.
            gw_term += self.fused * jnp.linalg.norm(v) ** 2

            # Compute the potential term, which excludes the spatial component.
            y = inner_x + tau * v[:, : inner_x.shape[1]]
            potential_term = jnp.sum(potential_fun(y))

            # Return the total proximal cost
            return potential_term + tau * gw_term

        # Define the optimizer.
        opt = jaxopt.LBFGS(fun=proximal_cost, **self.opt_hyperparams)

        @jax.jit
        def jitted_update(v, state):
            return opt.update(
                v,
                state,
                inner_x=x,
                inner_grad_dist=grad_dist,
                inner_mask=mask,
            )

        # Initialize the velocity and optimizer state.
        init_v = jnp.zeros((x.shape[0], x.shape[1] + space.shape[1]))
        v, state = init_v, opt.init_state(
            init_v,
            inner_x=x,
            inner_grad_dist=grad_dist,
            inner_mask=mask,
        )

        # Run the optimization, logging the proximal cost if needed.
        for _ in range(self.maxiter):
            v, state = jitted_update(v, state)
            if self.log_callback:
                self.log_callback({"proximal_cost": state.error})
            if state.error < self.tol:
                break

        # Compute the new omics and spatial coordinates.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]

    def training_step(
        self,
        x: jnp.ndarray,
        space: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """This function computes the velocity field :math:`v` associated with the
        potential :math:`\\Phi` using the implicit proximal step and returns the
        updated x and spatial coordiantes. This function is used during training
        when the potential is parameterized by a neural network.

        Args:
            x (jnp.ndarray): The x coordinates of the cells.
            space (jnp.ndarray): The spatial coordinates of the cells.
            potential_network (nn.Module): A neural network that outputs a scalar.
            potential_params (optax.Params): The parameters of the neural network.
            tau (float): The step size.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The updated x and spatial coordinates.
        """

        # Define a mask to apply to the outer product, based on the straight and cross
        # values. This is a block matrix of shape (d1 + d2, d1 + d2), indexed by k,l.
        d1, d2 = x.shape[1], space.shape[1]
        mask_xx = self.straight * jnp.ones((d1, d1))
        mask_xs = self.cross * jnp.ones((d1, d2))
        mask_ss = self.straight * jnp.ones((d2, d2))
        mask_sx = self.cross * jnp.ones((d2, d1))
        mask = jnp.block([[mask_xx, mask_xs], [mask_sx, mask_ss]])

        # Compute the gradient of the distance function applied to u_i - u_j.
        grad_dist_x = self.grad_dist_fun(x)
        grad_dist_space = self.grad_dist_fun(space)
        grad_dist = jnp.concatenate([grad_dist_x, grad_dist_space], axis=1)

        def proximal_cost(
            v,
            inner_potential_params,
            inner_x,
            inner_grad_dist,
            inner_mask,
            # Let us define a helper function to compute the proximal cost.
        ):

            # Compute the velocity difference between each point in x and space.
            n, d = v.shape
            v_diff = v.reshape(n, 1, d) - v.reshape(1, n, d)

            # Compute the quadratic term.
            gw_term = jnp.einsum(
                "ijk,kl,ijl,ijl,ik->",
                inner_grad_dist,
                inner_mask,
                inner_grad_dist,
                v_diff,
                v,
            )
            gw_term /= n

            # Add the fused term.
            gw_term += self.fused * jnp.linalg.norm(v) ** 2

            # Compute the potential term, which excludes the spatial component.
            y = inner_x + tau * v[:, : inner_x.shape[1]]
            potential_term = jnp.sum(potential_network.apply(inner_potential_params, y))

            # Return the total proximal cost
            return potential_term + tau * gw_term

        # Define the optimizer.
        opt = jaxopt.LBFGS(fun=proximal_cost, **self.opt_hyperparams)

        # Initialize the velocity and optimizer state.
        init_v = jnp.zeros((x.shape[0], x.shape[1] + space.shape[1]))
        v, _ = opt.run(
            init_v,
            inner_potential_params=potential_params,
            inner_x=x,
            inner_grad_dist=grad_dist,
            inner_mask=mask,
        )

        # Compute the new omics and spatial coordinates.
        return x + tau * v[:, : x.shape[1]], space + tau * v[:, x.shape[1] :]

    def grad_dist_fun(self, u: jnp.ndarray) -> jnp.ndarray:
        """Applies the gradient of the distance function to u_i - u_j.

        Args:
            u (jnp.ndarray): The input array of shape (n, d).

        Returns:
            jnp.ndarray: The output array of shape (n, n, d)."""

        n, d = u.shape
        return grad(self.dist_fun)(u.reshape(n, 1, d) - u.reshape(1, n, d))
