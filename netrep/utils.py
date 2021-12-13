"""
Miscellaneous helper functions.
"""
import numba
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_random_state
from netrep.validation import check_equal_shapes
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import ortho_group
from scipy.spatial.distance import squareform


def centered_kernel(*args, **kwargs):
    """
    Lightly wraps `sklearn.metrics.pairwise.pairwise_kernels`
    to compute the centered kernel matrix.
    """
    K = pairwise_kernels(*args, **kwargs)
    sc = np.sum(K, axis=0, keepdims=True)
    sr = np.sum(K, axis=1, keepdims=True)
    ss = np.sum(sc)
    return K - (sc / sr.size) - (sr / sc.size) + (ss / K.size)


def angular_distance(X, Y):
    """
    Computes angular distance based on Frobenius inner product
    between two matrices.

    Parameters
    ----------
    X : m x n matrix
    Y : m x n matrix

    Returns
    -------
    distance : float between zero and pi.
    """
    normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
    corr = np.dot(X.ravel(), Y.ravel()) / normalizer
    # numerical precision issues require us to clip inputs to arccos
    return np.arccos(np.clip(corr, -1.0, 1.0))


def trunc_svd(X, r):
    """
    Singular value decomposition, keeping top r components.
    """
    return randomized_svd(X, n_components=r, n_iter=5)


def econ_svd(X):
    """
    Economic Singular Value Decomposition (SVD).
    """
    return np.linalg.svd(X, full_matrices=False)


def rand_orth(m, n=None, random_state=None):
    """
    Creates a random matrix with orthogonal columns or rows.

    Parameters
    ----------
    m : int
        First dimension

    n : int
        Second dimension (if None, matrix is m x m)

    random_state : int or np.random.RandomState
        Specifies the state of the random number generator.

    Returns
    -------
    Q : ndarray
        An m x n random matrix. If m > n, the columns are orthonormal.
        If m < n, the rows are orthonormal. If m == n, the result is
        an orthogonal matrix.
    """
    rs = check_random_state(random_state)
    n = m if n is None else n

    Q = ortho_group.rvs(max(m, n), random_state=rs)

    if Q.shape[0] > m:
        Q = Q[:m]
    if Q.shape[1] > n:
        Q = Q[:, :n]

    return Q


def whiten(X, alpha, preserve_variance=True, eigval_tol=1e-7):
    """
    Return regularized whitening transform for a matrix X.

    Parameters
    ----------
    X : ndarray
        Matrix with shape `(m, n)` holding `m` observations
        in `n`-dimensional feature space. Columns of `X` are
        expected to be mean-centered so that `X.T @ X` is
        the covariance matrix.

    alpha : float
        Regularization parameter, `0 <= alpha <= 1`.

    preserve_variance : bool
        If True, rescale the (partial) whitening matrix so
        that the total variance, trace(X.T @ X), is preserved.

    eigval_tol : float
        Eigenvalues of covariance matrix are clipped to this
        minimum value.

    Returns
    -------

    X_whitened : ndarray
        Transformed data matrix.

    Z : ndarray
        Matrix implementing the whitening transformation.
        `X_whitened = X @ Z`.
    """

    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, np.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = np.linalg.eigh(X.T @ X)
    lam = np.maximum(lam, eigval_tol)

    # Compute diagonal of (partial) whitening matrix.
    # 
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:

        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = np.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * np.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= np.sqrt(np.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    return X @ Z, Z



def rand_struc_orth(n, n_transforms=3, random_state=None):
    """
    Draws random sign flips for structured orthogonal
    transformation. See also, `struc_orth_matvec` function.

    Parameters
    ----------
    n : int
        Dimensionality.

    n_transforms : int
        Number of sign flips to perform in between Hadamard
        transforms. Default is 3.

    random_state : int or np.random.RandomState
        Random number specification.
    """
    rs = check_random_state(random_state)
    Ds = np.ones((n_transforms, n), dtype=int)
    idx = rs.rand(n_transforms, n) > .5
    Ds[idx] = -1
    return Ds


def struc_orth_matvec(Ds, a, transpose=False):
    """
    Structured orthogonal matrix-vector multiply. Modifies
    vector `a` in-place.

    If transpose == False, then this computes:

        H @ Ds[-1] @ ... H @ Ds[1] @ H @ Ds[0] @ H @ a

    If transpose == True, then this computes:

        H @ Ds[0] @ ... H @ Ds[-2] @ H @ Ds[-1] @ H @ a

    Above, H is a normalized Hadamard matrix (i.e. normalized
    by )

    Parameters
    ----------
    Ds : ndarray
        (n_transforms x n) matrix specifying sign flips in
        between each Hadamard transform.

    a : ndarray
        Vector with n elements. An error is thrown if n is
        not a power of 2.

    transpose : bool
        If True, performs matrix-transpose times vector
        multiply. Default is False.
    """

    # Check inputs.
    if a.ndim != 1:
        raise ValueError("Expected array `a` to be a vector.")

    if Ds.ndim != 2:
        raise ValueError("Expected array `Ds` to be a matrix.")

    if Ds.shape[1] != a.size:
        raise ValueError(
            "Dimension mismatch. Expected Ds.shape[1] == a.size.")

    if ((a.size & (a.size - 1)) != 0):
        raise ValueError(
            "Expected length of `a` to be a power of two. "
            "Saw instead, len(a) == {}.".format(a.size))

    # Reverse order if transpose is desired.
    _Ds = Ds[::-1] if transpose else Ds

    # Perform series of Walsh-Hadamard Transforms and sign flips.
    fwht(a)
    for D in _Ds:
        a *= D
        fwht(a)

    # Normalize by sqrt(n) for each WH transform.
    a /= np.sqrt(a.size) ** (1 + len(Ds))


@numba.jit(nopython=True)
def fwht(a):
    """
    In-place Fast Walsh–Hadamard Transform.

    Source: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    """
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
