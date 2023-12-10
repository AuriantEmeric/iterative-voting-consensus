import numpy as np

from src.ivc import compute_disagreement, get_majority


def test_get_majority():
    matrix = np.array([[1, 2, 1], [2, 3, 3]])
    assert (get_majority(matrix) == [1, 3]).all()

    matrix = np.array([[1, 2, 1, 2], [2, 3, 3, 3]])
    assert (get_majority(matrix) == [1, 3]).all() or (
        get_majority(matrix) == [2, 3]
    ).all()


def test_compute_disagreement():
    clusters = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    pi_star = np.array([1, 0, 1, 0])
    weights = np.ones_like(clusters.shape[1])
    disagreement = compute_disagreement(clusters, 2, pi_star, weights)

    assert (disagreement == np.array([[2, 0], [0, 2], [2, 0], [1, 3]])).all()


def test_disagreement_with_weights():
    clusters = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    pi_star = np.array([1, 0, 1, 0])
    weights = np.array([0, 1, 1])
    disagreement = compute_disagreement(clusters, 2, pi_star, weights)
    print(disagreement)
    assert (disagreement == np.array([[1, 0], [0, 1], [1, 0], [1, 2]])).all()
