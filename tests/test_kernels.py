"""
Tests kernels between network representations.
"""
import pytest
import numpy as np
from sklearn.utils.validation import check_random_state
from netrep.multiset import cnd_kernel
from netrep.barycenter import barycenter, alignment
TOL = 1e-6


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [4, 11])
@pytest.mark.parametrize('n', [2, 5, 10])
@pytest.mark.parametrize('N', [5, 20])
@pytest.mark.parametrize('lam', [1e-1, 1e0, 1e1])
@pytest.mark.parametrize('group', ["orth", "perm"])
@pytest.mark.parametrize('reference', ["barycenter", "random"])
@pytest.mark.parametrize('ground_metric', ["euclidean", "angular"])
def test_laplacian_kernel_posdef(seed, m, n, N, lam, reference, group, ground_metric):

    # Set random seed, sample random datasets
    rs = check_random_state(seed)

    # Random datasets.
    Xs = [rs.randn(m, n) for _ in range(N)]

    # Compute reference point.
    if reference == "barycenter":
        # TODO: switch ground metric when manopt is incorporated
        Xbar = barycenter(Xs, group=group, ground_metric="euclidean")
    elif reference == "random":
        Xbar = rs.randn(m, n)

    # Compute conditionally negative definite kernel.
    D = cnd_kernel(Xs, Xbar, group=group, ground_metric=ground_metric)

    # Compute kernel matrix.
    K = np.exp(-lam * D)
    
    # Assert positive definite.    
    assert np.linalg.eigvalsh(K).min() >= 0
