from typing import Iterable, Tuple

import jax.numpy as jnp
import jax
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


@jax.jit
def chamfer_distance(
    x_real: jnp.ndarray,
    x_pred: jnp.ndarray,
) -> float:
    """Compute the Chamfer distance between two point clouds.

    Args:
        x_real (jnp.ndarray): The real distribution.
        x_pred (jnp.ndarray): The predicted distribution.

    Returns:
        float: The Chamfer distance.
    """

    # Compute the distance matrix.
    dist_matrix = jnp.sum((x_real[:, None, :] - x_pred[None, :, :]) ** 2, axis=-1)

    # Compute the Chamfer distance.
    return jnp.mean(jnp.min(dist_matrix, axis=0)) + jnp.mean(
        jnp.min(dist_matrix, axis=1)
    )


@jax.jit
def hausdorff_distance(
    x_real: jnp.ndarray,
    x_pred: jnp.ndarray,
) -> float:
    """Compute the Hausdorff distance between two point clouds.

    Args:
        x_real (jnp.ndarray): The real distribution.
        x_pred (jnp.ndarray): The predicted distribution.

    Returns:
        float: The Hausdorff distance.
    """

    # Compute the distance matrix.
    dist_matrix = jnp.sum((x_real[:, None, :] - x_pred[None, :, :]) ** 2, axis=-1)

    # Compute the Hausdorff distance.
    return jnp.max(jnp.min(dist_matrix, axis=0)) + jnp.max(jnp.min(dist_matrix, axis=1))


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
