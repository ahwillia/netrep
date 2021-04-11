import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
import scipy.sparse
from sklearn.utils.validation import check_array, check_random_state


def alignment(X, Y, group="orth"):
    """
    Return a matrix that optimally aligns 'X' to 'Y'. Note
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

    group : str
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


def barycenter(
        Xs, group="orth", ground_metric="euclidean",
        random_state=None, tol=1e-3, max_iter=100,
        warmstart=None
    ):
    """
    Estimate the average (Karcher/Frechet mean) of p networks in the
    metric space defined by:

        d*(X, Y) = min_{T} d(X, Y @ T)^2

    For some ground metric 'd' and alignment operations 'T'.

    Parameters
    ----------
    Xs : list of p matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare. Rotations are learned
        and applied in the n-dimensional space.

    group : str
        Specifies the set of allowable alignment operations (a group of
        isometries). Must be one of ("orth", "perm", "identity").

    random_state : np.random.RandomState
        Specifies state of the random number generator.

    tol : float
        Convergence tolerance

    max_iter : int, optional.
        Maximum number of iterations to apply.

    warmstart : (m x n) ndarray, optional
        If provided, Xbar is initialized to this estimate.

    Returns
    -------
    Xbar : (m x n) ndarray.
        Average activation matrix.
    """

    if ground_metric == "euclidean":
        return _euclidean_barycenter(Xs, group, random_state, tol, max_iter, warmstart)
    elif ground_metric == "angular":
        raise NotImplementedError("Barycenters with angular distance aren't implemented yet...")
    else:
        raise ValueError("Unexpected value for 'metric' keyword argument.")


def _euclidean_barycenter(Xs, group, random_state, tol, max_iter, warmstart):
    """
    Parameters
    ----------
    Xs : list of p matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare. Rotations are learned
        and applied in the n-dimensional space.

    group : str
        Specifies group of ("orth", "perm", "roll", "identity")

    random_state : np.random.RandomState

    tol : float
        Convergence tolerance

    max_iter : int, optional.
        Maximum number of iterations to apply.

    Returns
    -------
    Xbar : (m x n) ndarray.
        Average activation matrix.
    """

    # Handle simple case of no alignment operation. This is just a classic average.
    if group == "identity":
        return np.mean(Xs, axis=0)

    # Check input
    Xs = check_array(Xs, allow_nd=True)
    if Xs.ndim != 3:
        raise ValueError(
            "Expected 3d array with shape"
            "(n_datasets x n_observations x n_features), but "
            "got {}-d array with shape {}".format(Xs.ndim, Xs.shape))

    # If only one matrix is provided, the barycenter is trivial.
    if Xs.shape[0] == 1:
        return Xs[0]

    # Check random state and initialize random permutation over networks.
    rs = check_random_state(random_state)
    indices = rs.permutation(len(Xs))

    # Initialize barycenter.
    Xbar = Xs[indices[-1]] if (warmstart is None) else warmstart
    X0 = np.empty_like(Xbar)

    # Main loop
    itercount, n, chg = 0, 1, np.inf
    while (chg > tol) and (itercount < max_iter):
        
        # Save current barycenter for convergence checking.
        np.copyto(X0, Xbar)

        # Iterate over datasets.
        for i in indices:

            # Align i-th dataset to barycenter.
            Q = alignment(Xs[i], Xbar, group=group)

            # Take a small step towards aligned representation.
            Xbar = (n / (n + 1)) * Xbar + (1 / (n + 1)) * (Xs[i] @ Q)
            n += 1

        # Detect convergence.
        Q = alignment(Xs[i], Xbar, group=group)
        chg = np.linalg.norm((Xbar @ Q) - X0) / np.sqrt(Xbar.size)

        # Move to next iteration, with new random ordering over datasets.
        rs.shuffle(indices)
        itercount += 1

    return Xbar
