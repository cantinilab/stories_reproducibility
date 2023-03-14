from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp

from ott.initializers.linear.initializers import SortingInitializer


def _vectorized_update(f: jnp.ndarray, modified_cost: jnp.ndarray) -> jnp.ndarray:
    """Inner loop DualSort Update.

    Args:
      f: potential f, array of size n.
      modified_cost: cost matrix minus diagonal column-wise.

    Returns:
      updated potential vector, f.
    """
    return jnp.min(modified_cost + f[None, :], axis=1)


def _coordinate_update(f: jnp.ndarray, modified_cost: jnp.ndarray) -> jnp.ndarray:
    """Coordinate-wise updates within inner loop.

    Args:
      f: potential f, array of size n.
      modified_cost: cost matrix minus diagonal column-wise.

    Returns:
      updated potential vector, f.
    """

    def body_fn(i: int, f: jnp.ndarray) -> jnp.ndarray:
        new_f = jnp.min(modified_cost[i, :] + f)
        return f.at[i].set(new_f)

    return jax.lax.fori_loop(0, len(f), body_fn, f)


@jax.tree_util.register_pytree_node_class
class ScanSortingInitializer(SortingInitializer):
    """Sorting initializer :cite:`thornton2022rethinking:22`.

    Solve non-regularized OT problem via sorting, then compute potential through
    iterated minimum on C-transform and use this potential to initialize
    regularized potential.

    Args:
      vectorized_update: Whether to use vectorized loop.
      max_iter: Number of DualSort steps.
    """

    def __init__(
        self,
        vectorized_update: bool = True,
        tolerance: float = 0.01,
        force_scan: bool = False,
        max_iter: int = 100,
    ):
        super().__init__(vectorized_update, tolerance, max_iter)
        self.force_scan = force_scan

    def _init_sorting_dual(
        self, modified_cost: jnp.ndarray, init_f: jnp.ndarray
    ) -> jnp.ndarray:
        """Run DualSort algorithm.

        Args:
          modified_cost: cost matrix minus diagonal column-wise.
          init_f: potential f, array of size n. This is the starting potential,
            which is then updated to make the init potential, so an init of an init.

        Returns:
          potential f, array of size n.
        """

        def body_fn(carry, t):
            new_f = fn(carry, modified_cost)
            return new_f, t

        if self.force_scan:
            fn = _vectorized_update if self.vectorized_update else _coordinate_update
            f_potential, _ = jax.lax.scan(body_fn, init_f, jnp.arange(self.max_iter))
            return f_potential

        else:
            return super()._init_sorting_dual(modified_cost, init_f)

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
        return (
            [],
            {
                "max_iter": self.max_iter,
                "force_scan": self.force_scan,
                "vectorized_update": self.vectorized_update,
            },
        )
