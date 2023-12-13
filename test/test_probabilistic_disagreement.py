import numpy as np

from ivc.utils import compute_probabilistic_disagreement


def test_disagreement():
    clusters = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    pi_star = np.array([1, 0, 1, 0])
    weights = np.ones_like(clusters.shape[1])
    disagreement = compute_probabilistic_disagreement(clusters, 2, pi_star, weights)

    assert (disagreement == np.array([[2.5, 0], [0.5, 2], [2.5, 0], [0.5, 3]])).all()


def test_disagreement_with_weights():
    clusters = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    pi_star = np.array([1, 0, 1, 0])
    weights = np.array([0, 1, 1])
    disagreement = compute_probabilistic_disagreement(clusters, 2, pi_star, weights)

    assert (disagreement == np.array([[1.5, 0], [0.5, 1], [1.5, 0], [0.5, 2]])).all()
