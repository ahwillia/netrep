"""
Tests deterministic metrics.
"""
import pytest
import numpy as np
from netrep.metrics import LinearMetric, PermutationMetric #, KernelizedMetric
from netrep.utils import angular_distance, rand_orth
from netrep.multiset import pairwise_distances
from numpy.testing import assert_array_almost_equal
from sklearn.utils.validation import check_random_state

TOL = 1e-6


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_uncentered_procrustes(seed, m, n):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)

    # Create a pair of randomly rotated matrices.
    X = rs.randn(m, n)
    Y = X @ Q

    # Fit model, assert distance == 0.
    metric = LinearMetric(alpha=1.0, center_columns=False)
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_centered_procrustes(seed, m, n):

    # Set random seed, draw random rotation, offset, and isotropic scaling.
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)
    v = rs.randn(1, n)
    c = rs.exponential()

    # Create a pair of randomly rotated matrices.
    X = rs.randn(m, n)
    Y = c * X @ Q + v

    # Fit model, assert distance == 0.
    metric = LinearMetric(alpha=1.0, center_columns=True)
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_uncentered_cca(seed, m, n):

    # Set random seed, draw random linear alignment.
    rs = check_random_state(seed)
    W = rs.randn(n, n)

    # Create pair of matrices related by a linear transformation.
    X = rs.randn(m, n)
    Y = X @ W

    # Fit CCA, assert distance == 0.
    metric = LinearMetric(alpha=0.0, center_columns=False)
    metric.fit(X, Y)
    assert metric.score(X, Y) < TOL

    # Fit Procrustes, assert distance is nonzero.
    metric = LinearMetric(alpha=1.0, center_columns=False)
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) > TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_centered_cca(seed, m, n):

    # Set random seed, draw random linear alignment and offset.
    rs = check_random_state(seed)
    W = rs.randn(n, n)
    v = rs.randn(1, n)

    # Create a pair of matrices related by a linear transformation.
    X = rs.randn(m, n)
    Y = X @ W + v

    # Fit model, assert distance is zero.
    metric = LinearMetric(alpha=0.0, center_columns=True)
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL

    # Fit Procrustes, assert distance is nonzero.
    metric = LinearMetric(alpha=1.0, center_columns=True)
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) > TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_principal_angles(seed, m, n):

    # Set random seed, draw random linear alignment.
    rs = check_random_state(seed)
    W = rs.randn(n, n)

    # Create pair of matrices related by a linear transformation.
    X = rand_orth(m, n)
    Y = rand_orth(m, n)

    # Compute metric based on principal angles.
    cos_thetas = np.linalg.svd(X.T @ Y, compute_uv=False)
    dist_1 = np.arccos(np.mean(cos_thetas))

    # Fit model, assert two approaches match.
    metric = LinearMetric(alpha=1.0, center_columns=False).fit(X, Y)
    assert abs(dist_1 - metric.score(X, Y)) < TOL


# @pytest.mark.parametrize('seed', [1, 2, 3])
# @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
# @pytest.mark.parametrize('m', [100])
# @pytest.mark.parametrize('n', [10])
# def test_linear_kernelized_procrustes(seed, alpha, m, n):
    
#     rs = check_random_state(seed)
#     X = rs.randn(m, n)
#     Y = rs.randn(m, n)

#     metric_1 = LinearMetric(alpha=alpha, center_columns=True)
#     metric_2 = KernelizedMetric(alpha=alpha, kernel="linear")

#     tX1, tY1 = metric_1.fit_transform(X, Y)
#     tX2, tY2 = metric_2.fit_transform(X, Y)

#     dist_1 = angular_distance(*metric_1.fit_transform(X, Y))
#     dist_2 = angular_distance(*metric_2.fit_transform(X, Y))
#     assert abs(dist_1 - dist_2) < TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('m', [31])
@pytest.mark.parametrize('n', [30])
def test_triangle_inequality_linear(seed, alpha, m, n):
    
    rs = check_random_state(seed)
    X = rs.randn(m, n)
    Y = rs.randn(m, n)
    M = rs.randn(m, n)

    metric = LinearMetric(alpha=alpha, center_columns=True)

    dXY = metric.fit(X, Y).score(X, Y)
    dXM = metric.fit(X, M).score(X, M)
    dMY = metric.fit(M, Y).score(M, Y)

    assert dXY <= (dXM + dMY + TOL)


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('center_columns', [True, False])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_permutation(seed, center_columns, m, n):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    
    # Create a pair of randomly rotated matrices.
    X = rs.randn(m, n)
    Y = np.copy(X)[:, rs.permutation(n)]

    # Fit model, assert distance == 0.
    metric = PermutationMetric(center_columns=center_columns)
    assert abs(metric.fit(X, Y).score(X, Y)) < TOL



@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [31])
@pytest.mark.parametrize('n', [30])
@pytest.mark.parametrize('N', [20])
@pytest.mark.parametrize('b', np.linspace(0.1, 1, 4))
@pytest.mark.parametrize('lam', [1e-2, 1e-1, 1e0, 1e1])
def test_laplacian_kernel_posdef(seed, m, n, N, b, lam):

    # Set random seed, sample random datasets
    rs = check_random_state(seed)
    X0 = rs.randn(m, n)
    Xs = [b * X0 + (1 - b) * rs.randn(m, n) for _ in range(N)]


    # Compute pairwise distances.
    metric = LinearMetric(alpha=1.0, center_columns=True)
    D = pairwise_distances(metric, Xs, verbose=False)

    # Compute kernel matrix.
    K = np.exp(-lam * D)
    
    # Assert positive definite.    
    assert np.linalg.eigvalsh(K).min() > -TOL
