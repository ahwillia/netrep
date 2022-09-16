"""
Tests utility functions.
"""
import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from netrep.utils import (
    rand_orth,
    centered_kernel,
    fwht,
    rand_struc_orth,
    struc_orth_matvec,
    whiten,
    sq_bures_metric,
    sq_bures_metric_slow
)

from sklearn.utils.validation import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels 

import scipy.linalg

ATOL = 1e-6
RTOL = 1e-6

@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [50, 100])
@pytest.mark.parametrize('n', [10, 20])
def test_whiten(seed, m, n):
    # For this test, we assume m > n.
    rs = check_random_state(seed)
    X = rs.randn(m, n)
    XZ, Z = whiten(X, 0.0, preserve_variance=False)
    assert_array_almost_equal(XZ.T @ XZ, np.eye(n))


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [50, 100])
@pytest.mark.parametrize('n', [10, 20])
def test_whiten_preserve_variance(seed, m, n):
    # For this test, we assume m > n.
    rs = check_random_state(seed)
    X = rs.randn(m, n)
    XZ, Z = whiten(X, 0.0, preserve_variance=True)
    gram = (XZ.T @ XZ)
    d = np.full(n, np.sum(gram) / n)
    assert_array_almost_equal(gram, np.diag(d))


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('m', [50, 100])
@pytest.mark.parametrize('n', [10, 20])
def test_partial_whiten_preserve_variance(seed, alpha, m, n):
    # For this test, we assume m > n.
    rs = check_random_state(seed)
    X = rs.randn(m, n)
    XZ, Z = whiten(X, alpha, preserve_variance=True)
    assert_allclose(np.trace(X.T @ X), np.trace(XZ.T @ XZ))


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [10, 20])
@pytest.mark.parametrize('n', [10, 20])
def test_rand_orth(seed, m, n):
    Q = rand_orth(m, n, random_state=seed)

    if m == n:
        assert_array_almost_equal(Q.T @ Q, np.eye(n))
        assert_array_almost_equal(Q @ Q.T, np.eye(n))
    elif m > n:
        assert_array_almost_equal(Q.T @ Q, np.eye(n))
    else:
        assert_array_almost_equal(Q @ Q.T, np.eye(m))


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
def test_centered_kernel(seed, m, n):

    rs = check_random_state(seed)
    
    # Check linear kernel is centered.
    X = rs.randn(m, n)
    K = pairwise_kernels(X - np.mean(X, axis=0), metric="linear")
    K2 = centered_kernel(X, metric="linear")
    assert_array_almost_equal(K, K2)
    assert_array_almost_equal(centered_kernel(X), centered_kernel(X, X))


# @pytest.mark.parametrize('seed', [1, 2, 3])
# @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
# def test_fast_hadamard_transform(seed, n):

#     # Form Hadamard matrix explicitly
#     H = scipy.linalg.hadamard(2 ** n)
    
#     # Draw random vector.
#     rs = check_random_state(seed)
#     x = rs.randn(2 ** n)

#     # Perform explicit computation
#     expected = H @ x

#     # Check that Fast-Walsh_Hadamard transform matches.
#     fwht(x)  # updates x in-place.
#     assert_array_almost_equal(expected, x)


# @pytest.mark.parametrize('seed', [1, 2, 3])
# @pytest.mark.parametrize('n', [1, 2, 14])
# @pytest.mark.parametrize('n_transforms', [1, 3, 6])
# def test_structured_orth(seed, n, n_transforms):

#     # Draw random vectors.
#     rs = check_random_state(seed)
#     x = rs.randn(2 ** n)
#     y = rs.randn(2 ** n)

#     # Compute inner product.
#     original_inner_prod = np.dot(x, y)

#     # Draw sign flips.
#     Ds = rand_struc_orth(2 ** n, n_transforms=n_transforms, random_state=rs)

#     # Apply structured orthogonal transformation. If this is
#     # indeed orthogonal, the inner product should be preserved.
#     struc_orth_matvec(Ds, x)
#     struc_orth_matvec(Ds, y)

#     # Check that the inner products match.
#     assert_allclose(
#         np.dot(x, y), original_inner_prod, atol=ATOL, rtol=RTOL)


# @pytest.mark.parametrize('seed', [1, 2, 3])
# @pytest.mark.parametrize('n', [1, 2, 14])
# @pytest.mark.parametrize('n_transforms', [1, 3, 6])
# def test_structured_orth_inverse(seed, n, n_transforms):

#     # Draw random vectors.
#     rs = check_random_state(seed)
#     x = rs.randn(2 ** n)
#     y = x.copy()

#     # Draw sign flips.
#     Ds = rand_struc_orth(2 ** n, n_transforms=n_transforms, random_state=rs)

#     # Apply structured orthogonal transformation, and then apply
#     # the inverse transformation.
#     struc_orth_matvec(Ds, x)
#     struc_orth_matvec(Ds, x, transpose=True)

#     # Check that we recover our original vector
#     assert_allclose(x, y, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('n', [1, 2, 14])
def test_bures(seed, n):

    # Draw covariances.
    rs = check_random_state(seed)
    X = rs.randn(n, n)
    Y = rs.randn(n, n)
    Sx = X @ X.T
    Sy = Y @ Y.T

    # Check that we recover our original vector
    assert_allclose(
        sq_bures_metric(Sx, Sy),
        sq_bures_metric_slow(Sx, Sy),
        atol=ATOL, rtol=RTOL
    )
