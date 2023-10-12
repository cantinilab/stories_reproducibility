from typing import Iterable, Tuple

import jax.numpy as jnp
import numpy as np
from jax.random import KeyArray, PRNGKey
from sklearn.neighbors import KNeighborsClassifier

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein


def sinkhorn_distance(
    x_real: jnp.ndarray,
    x_pred: jnp.ndarray,
    rank: int = -1,
    key: KeyArray = PRNGKey(0),
) -> float:
    """Compute the Sinkhorn distance between two distributions.

    Args:
        x_real (jnp.ndarray): The real distribution.
        x_pred (jnp.ndarray): The predicted distribution.
        rank (int, optional): If different to -1, use low-rank Sinkhorn. Defaults to -1.
        key (KeyArray, optional): The random key. Defaults to PRNGKey(0).

    Returns:
        float: The Sinkhorn distance between distributions.
    """
    # TODO specify epsilon

    # Initialize the Optimal Transport problem.
    geom = PointCloud(x_pred, x_real)
    problem = LinearProblem(geom)

    # Define the solver, either full rank or low-rank.
    solver = Sinkhorn() if rank == -1 else LRSinkhorn(rank=rank)

    # Solve the problem and check covnergence.
    out = solver(problem, rng=key)
    assert out.converged

    # Return the Sinkhorn distance.
    return out.reg_ot_cost


def gromov_wasserstein_distance(
    x_real: jnp.ndarray,
    space_real: jnp.ndarray,
    x_pred: jnp.ndarray,
    space_pred: jnp.ndarray,
    rank: int = -1,
    fused: float = 1.0,
    key: KeyArray = PRNGKey(0),
) -> float:
    """Compute the entropy-regularized Gromov-Wasserstein distance
    between two distributions.

    Args:
        x_real (jnp.ndarray): The real distribution.
        space_real (jnp.ndarray): The real spatial coordinates.
        x_pred (jnp.ndarray): The predicted distribution.
        space_pred (jnp.ndarray): The predicted spatial coordinates.
        rank (int, optional): If different to -1, use low-rank GW. Defaults to -1.
        fused (float, optional): The fused penalty. Defaults to 1.0.
        key (KeyArray, optional): The random key. Defaults to PRNGKey(0).

    Returns:
        float: The Gromov-Wasserstein distance between distributions.
    """
    # TODO specify epsilon

    # Initialize the Optimal Transport problem.
    geom_xy = PointCloud(x_pred, x_real)
    geom_xx = PointCloud(space_pred, space_pred)
    geom_yy = PointCloud(space_real, space_real)
    problem = QuadraticProblem(geom_xx, geom_yy, geom_xy=geom_xy, fused_penalty=fused)

    # Define the solver, either full rank or low-rank.
    linear_ot_solver = Sinkhorn if rank == -1 else LRSinkhorn
    solver = GromovWasserstein(rank=rank, linear_ot_solver=linear_ot_solver)

    # Solve the problem and check covnergence.
    out = solver(problem, rng=key)
    assert out.converged

    # Return the Sinkhorn distance.
    return out.reg_gw_cost


def knn_classify(
    x_real: jnp.ndarray,
    labels_real: Iterable,
    x_pred: jnp.ndarray,
    k: int = 5,
) -> Iterable:
    """Compute the accuracy of a k-nearest neighbors classifier.

    Args:
        x_real (jnp.ndarray): The real distribution.
        labels_real (Iterable): The labels of the real distribution.
        x_pred (jnp.ndarray): The predicted distribution.
        k (int, optional): The number of neighbors. Defaults to 5.

    Returns:
        Iterable: The predicted labels.
    """

    # Define the classifier.
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier.
    classifier.fit(x_real, labels_real)

    # Predict the labels.
    return classifier.predict(x_pred)


def compare_real_and_knn_histograms(
    x_real: jnp.ndarray,
    labels_real: Iterable,
    x_pred: jnp.ndarray,
    k: int = 5,
) -> float:
    """First, compute the predicted labels using a k-nearest neighbors classifier.
    Then compare the histogram of the real labels and the histogram of the predicted
    labels.

    Args:
        x_real (jnp.ndarray): The real distribution.
        labels_real (Iterable): The labels of the real distribution.
        x_pred (jnp.ndarray): The predicted distribution.
        k (int, optional): The number of neighbors. Defaults to 5.

    Returns:
        float: The L1 distance between the two histograms."""

    # Compute the predicted labels.
    labels_pred = knn_classify(x_real, labels_real, x_pred, k)

    # Compute the L1 distance between the two histograms.
    score = 0
    for label in np.unique(labels_real):
        score += np.abs(np.mean(labels_real == label) - np.mean(labels_pred == label))
    return score, labels_pred


def score_transitions(
    x_real: jnp.ndarray,
    labels_previous: Iterable,
    unique_labels_previous: Iterable,
    labels_real: Iterable,
    unique_labels_real: Iterable,
    transition_mask: jnp.ndarray,
    x_pred: jnp.ndarray,
    k: int = 5,
) -> Tuple[jnp.ndarray, float]:
    """First, compute the predicted labels using a k-nearest neighbors classifier.
    Then, compute the transition between previous labels and predicted labels.
    Finally, use the transition mask to compute the proportion of valid transitions.

    Args:
        x_real (jnp.ndarray): The real distribution.
        labels_previous (Iterable): The labels of the previous distribution.
        unique_labels_previous (Iterable): The unique labels of the previous distribution.
        labels_real (Iterable): The labels of the real distribution.
        unique_labels_real (Iterable): The unique labels of the real distribution.
        transition_mask (jnp.ndarray): The transition mask, indexed by the unique
        labels defined previously.
        x_pred (jnp.ndarray): The predicted distribution.
        k (int, optional): The number of neighbors. Defaults to 5.

    Returns:
        Tuple[jnp.ndarray, float]: The transition matrix and the proportion
        of valid transitions."""

    # Compute the predicted labels.
    labels_pred = knn_classify(x_real, labels_real, x_pred, k)

    # Initialize the transition matrix to zero.
    n_previous = len(unique_labels_previous)
    n_real = len(unique_labels_real)
    transition_matrix = np.zeros((n_previous, n_real))

    # Compute the transition matrix.
    for i, l_previous in enumerate(unique_labels_previous):
        for j, l_new in enumerate(unique_labels_real):
            transition_matrix[i, j] = np.sum(
                (labels_previous == l_previous) & (labels_pred == l_new)
            )

    # Score the transition matrix.
    score = np.sum(transition_matrix * transition_mask) / np.sum(transition_matrix)

    return transition_matrix, score
