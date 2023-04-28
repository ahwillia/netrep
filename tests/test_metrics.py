"""
Tests metrics between network representations.
"""
import pytest
import numpy as np
from netrep.metrics import LinearMetric, PermutationMetric
from netrep.utils import angular_distance, rand_orth
from sklearn.utils.validation import check_random_state

TOL = 1e-6


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
@pytest.mark.parametrize('score_method', ['euclidean', 'angular'])
def test_permutation(seed, center_columns, m, n, score_method):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    
    # Create a pair of randomly rotated matrices.
    X = rs.randn(m, n)
    Y = np.copy(X)[:, rs.permutation(n)]

    # Fit model, assert distance == 0.
    metric = PermutationMetric(
        center_columns=center_columns, score_method=score_method
    )
    assert abs(metric.fit(X, Y).score(X, Y)) < TOL


