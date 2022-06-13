import itertools
import numpy as np
from tqdm import tqdm


def euclidean_tangent_space(Xs, Xbar, group="orth"):
    """
    Transform list of K matrices ('Xs'), into an approximate
    Euclidean space (tangent space) at a point Xbar.

    Note: assumes that the ground metric is Euclidean.

    Parameters
    ----------
    Xs : list of K matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare.

    Xbar : (m x n) ndarray.
        Reference point. Each element in 'Xs' is aligned to 'Xbar'.

    group : str
        Specifies group of alignment operations.

    Returns
    -------
    Xs_tang : list of K matrices, (m x n) ndarrays.
        Matrix-valued datasets in tangent space. These are the
        residuals of each element in 'Xs' after alignment to
        'Xbar'.
    """
    Xs_tang = np.empty((len(Xs), Xbar.shape[0], Xbar.shape[1]))
    for i, X in enumerate(Xs):
        Xs_tang[i] = Xbar - (X @ alignment(X, Xbar, group=group))
    return Xs_tang


def pairwise_distances(
        metric, traindata, testdata=None, verbose=True,
        enable_caching=False
    ):
    """
    Compute pairwise distances between a collection of
    networks. Similar to ``scipy.spatial.distance.pdist``.

    Parameters
    ----------
    metric : Metric
        Metric to evaluate pairwise distances

    traindata : list of K matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare.

    testdata : list of K matrices, (p x n) ndarrays, optional.
        If provided, metrics are fit to traindata
        and then evaluated on testdata.

    verbose : bool, optional
        Prints progress bar if True. (Default is True.)

    Returns
    -------
    train_dists : (K x K) symmetric matrix.
        Matrix of pairwise distances on training set.

    test_dists : (K x K) symmetric matrix, optional.
        Matrix of pairwise distances on the test set.
    """

    # Allocate space for distances.
    m = len(traindata)
    D_train = np.zeros((m, m))

    if testdata is not None:
        D_test = np.zeros((m, m))

    # Set up progress bar.
    if verbose:
        pbar = tqdm(total=(m * (m - 1)) // 2)

    # Fit partial whitening transforms to each dataset.
    if enable_caching:
        caches = [metric.partial_fit(trn) for trn in traindata]

    # Compute all pairwise distances.
    for i in range(m):
        for j in range(i + 1, m):

            # Fit metric.
            if enable_caching:
                metric.finalize_fit(caches[i], caches[j])
            else:
                metric.fit(traindata[i], traindata[j])

            # Evaluate distance on the training set.
            D_train[i, j] = metric.score(traindata[i], traindata[j])
            D_train[j, i] = D_train[i, j]

            # Evaluate distance on the test set.
            if testdata is not None:
                D_test[i, j] = metric.score(testdata[i], testdata[j])
                D_test[j, i] = D_test[i, j]

            # Update progress bar.
            if verbose:
                pbar.update(1)

    # Close progress bar.
    if verbose:
        pbar.close()

    return D_train if (testdata is None) else (D_train, D_test)


def cross_distances(
        metric, Xs, Ys, Xs_test=None, Ys_test=None, verbose=True
    ):
    """
    Compute pairwise distances between two collections
    of networks. Similar to ``scipy.spatial.distance.cdist``.

    Parameters
    ----------
    metric : Metric
        Metric to evaluate pairwise distances

    Xs : list of Nx matrices, (m x n) ndarrays.
        First set of matrix-valued datasets to compare.

    Ys : list of Ny matrices, (m x n) ndarrays.
        Second set of matrix-valued datasets to compare.

    Xs_test : list of Nx matrices, (p x n) ndarrays, optional.
        If provided, metrics are fit to data in Xs
        and then evaluated on Xs_test.

    Xs_test : list of Ny matrices, (p x n) ndarrays, optional.
        If provided, metrics are fit to data in Ys
        and then evaluated on Ys_test.

    verbose : bool, optional
        Prints progress bar if True. (Default is True.)

    Returns
    -------
    train_dists : (Nx x Ny) matrix.
        Matrix of pairwise distances on training set.

    test_dists : (Nx x Ny) matrix, optional.
        Matrix of pairwise distances on the test set.
    """

    # Allocate space for training distances.
    Nx, Ny = len(Xs), len(Ys)
    D_train = np.zeros((Nx, Ny))

    # Allocate space for testing distances.
    if (Xs_test is not None) and (Xs_test is not None):
        if len(Xs_test) != Nx:
            raise ValueError(
                "Length of Xs_test does not match train set."
            )
        if len(Ys_test) != Ny:
            raise ValueError(
                "Length of Ys_test does not match train set."
            )
        D_test = np.zeros((Nx, Ny))

    elif (Xs_test is None) and (Ys_test is not None):
        raise ValueError(
            "If 'Ys_test' is specified. 'Xs_test' must also"
            "be specified."
        )

    elif (Ys_test is not None) and (Ys_test is None):
        raise ValueError(
            "If 'Xs_test' is specified. 'Ys_test' must also"
            "be specified."
        )

    else:
        D_test = None

    # Create progress bar.
    if verbose:
        pbar = tqdm(total=(Nx * Ny))

    # Compute distances.
    for i, j in itertools.product(range(Nx), range(Ny)):
        metric.fit(Xs[i], Ys[j])
        D_train[i, j] = metric.score(Xs[i], Ys[j])

        if D_test is not None:
            D_test[i, j] = metric.score(Xs_test[i], Ys_test[j])

        if verbose:
            pbar.update(1)

    # Close progress bar.
    if verbose:
        pbar.close()

    # Return distance matrices.
    return D_train if (D_test is None) else (D_train, D_test)
