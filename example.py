"""Example of utilization to find clusters a correpted list of clusterings."""
import numpy as np

from src.ipvc import iterative_probabilistic_voting_consensus
from src.ivc import iterative_voting_consensus


def check_cluster(groundtruth: np.ndarray, found: np.ndarray) -> float:
    """Evaluate the percentage of errors between the groundtruth clustering and the found clustering.

    Parameters
    ----------
    groundtruth : np.ndarray
        Groundtruth clustering
    found : np.ndarray
        Estimated clustering

    Returns
    -------
    float
        Percentage of error
    """
    values = np.unique(groundtruth)
    errors = 0
    for value in values:
        mask = groundtruth == value
        counts = np.bincount(found[mask])
        errors += counts.sum() - counts.max()
    return errors / groundtruth.size


if __name__ == "__main__":
    # Parameters
    number_of_clusters = 10
    number_of_clusterings = 5

    # Generate the true clusterings
    clusters_true = np.random.randint(number_of_clusters, size=(1000))

    # Simulate various corrupted clusterings
    clusters_false = np.repeat(
        clusters_true[:, None], repeats=number_of_clusterings, axis=1
    )
    replacements = np.random.randint(number_of_clusters, size=clusters_false.shape)
    mask = np.random.choice([0, 1], size=clusters_false.shape, p=[0.9, 0.1]).astype(
        np.bool_
    )
    clusters_false[mask] = replacements[mask]

    # Use the algorithms to find the true clusterings
    pi_star = iterative_voting_consensus(clusters_false, max_value=number_of_clusters)
    print(
        f"Percentage of error with the IVC algorithm {check_cluster(clusters_true, pi_star)}"
    )

    pi_star = iterative_probabilistic_voting_consensus(
        clusters_false, max_value=number_of_clusters
    )
    print(
        f"Percentage of error with the IPVC algorithm {check_cluster(clusters_true, pi_star)}"
    )
