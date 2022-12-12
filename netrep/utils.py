"""
Miscellaneous helper functions.
"""
from typing import Tuple, Literal, Union, Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
import scipy.sparse
from scipy.stats import ortho_group
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_random_state


def align(
    X: npt.NDArray, 
    Y: npt.NDArray, 
    group: Literal["orth", "perm", "identity"] = "orth"
    ) -> Union[npt.NDArray, scipy.sparse.csr_matrix, scipy.sparse.dia_matrix]:
    """Return a matrix that optimally aligns 'X' to 'Y'. Note
    that the optimal alignment is the same for either the
    angular distance or the Euclidean distance since all
    alignments come from sub-groups of the orthogonal group.

    Parameters
    ----------
    X : (m x n) ndarray.
        Activation patterns across 'm' inputs and 'n' neurons,
        sampled from the first network (the one which is transformed
        by the alignment operation).

    Y : (m x n) ndarray.
        Activation patterns across 'm' inputs and 'n' neurons,
        sampled from the second network (the one which is fixed).

    group : Literal["orth", "perm", "identity"]
        Specifies the set of allowable alignment operations (a group of
        isometries). Must be one of ("orth", "perm", "identity").

    Returns
    -------
    T : (n x n) ndarray or sparse matrix.
        Linear operator such that 'X @ T' is optimally aligned to 'Y'.
        Note further that 'Y @ T.transpose()' is optimally aligned to 'X',
        by symmetry.
    """

    if group == "orth":
        return orthogonal_procrustes(X, Y)[0]

    elif group == "perm":
        ri, ci = linear_sum_assignment(X.T @ Y, maximize=True)
        n = ri.size
        return scipy.sparse.csr_matrix(
            (np.ones(n), (ri, ci)), shape=(n, n)
        )

    elif group == "identity":
        return scipy.sparse.eye(X.shape[1])

    else:
        raise ValueError(f"Specified group '{group}' not recognized.")


def sq_bures_metric_slow(A: npt.NDArray, B: npt.NDArray) -> float:
    """Slow way to compute the square of the Bures metric between two
    positive-definite matrices.
    """
    va, ua = np.linalg.eigh(A)
    Asq = ua @ (np.sqrt(va[:, None]) * ua.T)
    return (
        np.trace(A) + np.trace(B) - 2 * np.sum(np.sqrt(np.linalg.eigvalsh(Asq @ B @ Asq)))
    )


def sq_bures_metric(A: npt.NDArray, B: npt.NDArray) -> float:
    """Slow way to compute the square of the Bures metric between two
     positive-definite matrices.
    """
    va, ua = np.linalg.eigh(A)
    vb, ub = np.linalg.eigh(B)
    sva, svb = np.sqrt(va), np.sqrt(vb)
    return (
        np.sum(va) + np.sum(vb) - 2 * np.sum(
            np.linalg.svd(
                (sva[:, None] * ua.T) @ (ub * svb[None, :]),
                compute_uv=False
            )
        )
    )


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


def angular_distance(X: npt.NDArray, Y: npt.NDArray) -> float:
    """Computes angular distance based on Frobenius inner product between two matrices.

    Parameters
    ----------
    X : (m x n) ndarray
    Y : (m x n) ndarray

    Returns
    -------
    distance : float between zero and pi.
    """
    normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
    corr = np.dot(X.ravel(), Y.ravel()) / normalizer
    # numerical precision issues require us to clip inputs to arccos
    return np.arccos(np.clip(corr, -1.0, 1.0))


def trunc_svd(X: npt.NDArray, r: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Singular value decomposition, keeping top r components."""
    return randomized_svd(X, n_components=r, n_iter=5)


def econ_svd(X: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Economic Singular Value Decomposition (SVD)."""
    return np.linalg.svd(X, full_matrices=False)


def rand_orth(
    m: int, 
    n: Optional[int] = None, 
    random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> npt.NDArray:
    """Creates a random matrix with orthogonal columns or rows.

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


def whiten(
    X: npt.NDArray, 
    alpha: float, 
    preserve_variance: bool = True, 
    eigval_tol=1e-7
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return regularized whitening transform for a matrix X.

    Parameters
    ----------
    X : ndarray
        Matrix with shape `(m, n)` holding `m` observations
        in `n`-dimensional feature space. Columns of `X` are
        expected to be mean-centered so that `X.T @ X` is
        the covariance matrix.
    alpha : float
        Regularization parameter, `0 <= alpha <= 1`. When
        `alpha == 0`, the data matrix is fully whitened.
        When `alpha == 1` the data matrix is not transformed
        (`Z == eye(X.shape[1])`).
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



def rand_struc_orth(
    n: int, 
    n_transforms: int = 3, 
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> npt.NDArray:
    """Draws random sign flips for structured orthogonal
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
    """Structured orthogonal matrix-vector multiply. Modifies
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


# @numba.jit(nopython=True)
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
