import numpy as np
from sklearn.utils.validation import check_array, check_random_state
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from netrep.metrics import LinearMetric


def procrustes_average(Xs, random_state=None, tol=1e-3, max_iter=100):
    """
    Compute the average of p networks in the Procrustes metric space.

    Parameters
    ----------
    Xs : list of p matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare. Rotations are learned
        and applied in the n-dimensional space.

    random_state : int or np.random.RandomState
        Specifies the state of the random number generator.

    tol : float
        Convergence tolerance

    max_iter : int, optional.
        Maximum number of iterations to apply.

    Returns
    -------
    Xbar : (m x n) ndarray.
        Average activation matrix.
    """

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

    # Initialize barycenter
    random_state = check_random_state(random_state)
    indices = random_state.permutation(len(Xs))
    Xbar = Xs[indices[-1]]
    X0 = np.empty_like(Xbar)

    # Main loop
    itercount, n, chg = 0, 1, np.inf
    while (chg > tol) and (itercount < max_iter):
        
        # Save current barycenter for convergence checking.
        np.copyto(X0, Xbar)

        # Iterate over datasets
        random_state.shuffle(indices)
        for i in indices:

            # Align i-th dataset to barycenter.
            Q, _ = orthogonal_procrustes(Xs[i], Xbar)

            # Take a small step towards aligned representation.
            Xbar = (n / (n + 1)) * Xbar + (1 / (n + 1)) * (Xs[i] @ Q)
            n += 1

        # Detect convergence.
        Q, _ = orthogonal_procrustes(Xbar, X0)
        chg = np.linalg.norm((Xbar @ Q) - X0) / np.sqrt(Xbar.size)

        # Move to next iteration, with new random ordering over datasets.
        indices = random_state.permutation(len(Xs))
        itercount += 1

    return Xbar


def procrustes_kmeans(
        Xs, n_clusters, dist_matrix=None, max_iter=100, random_state=None
    ):
    """
    Perform K-means clustering in the metric space defined by
    the Procrustes metric.

    Parameters
    ----------
    Xs : list of p matrices, (m x n) ndarrays.
        Matrix-valued datasets to compare. Rotations are learned
        and applied in the n-dimensional space.

    n_clusters : int
        Number of clusters to fit.

    dist_matrix : pairwise distances, (p x p) symmetric matrix, optional.
        Pairwise distances between all p networks. This is used
        to seed the k-means algorithm by a k-means++ procedure.

    max_iter : int, optional.
        Maximum number of iterations to apply.

    random_state : int or np.random.RandomState
        Specifies the state of the random number generator.


    Returns
    -------
    centroids : (n_clusters x n) ndarray.
        Cluster centroids.

    labels : length-p ndarray.
        Vector holding the cluster labels for each network.

    cent_dists : (n_clusters x p) ndarray
        Matrix holding the distance from each cluster centroid
        to each network.
    """

    # Initialize random number generator.
    rs = check_random_state(random_state)

    # Initialize Procrustes metric.
    proc_metric = LinearMetric(alpha=1.0)

    # Check input.
    Xs = check_array(Xs, allow_nd=True)
    if Xs.ndim != 3:
        raise ValueError(
            "Expected 3d array with shape"
            "(n_datasets x n_observations x n_features), but "
            "got {}-d array with shape {}".format(Xs.ndim, Xs.shape))

    # Initialize pairwise distances between all networks.
    if dist_matrix is None:
        dist_matrix = pairwise_distances(proc_metric, Xs, verbose=False)

    # Pick first centroid randomly
    init_centroid_idx = [rs.choice(len(Xs))]
    init_dists = dist_matrix[idx[0]] ** 2

    # Pick additional clusters according to k-means++ procedure.
    for k in range(1, n_clusters):
        init_centroid_idx.append(
            rs.choice(len(Xs), p = init_dists / init_dists.sum())
        )
        init_dists = np.minimum(
            init_dists,
            dist_matrix[init_centroid_idx[-1]] ** 2
        )

    # Collect centroids.
    centroids = [np.copy(Xs[i]) for i in idx]

    # Determine cluster labels for each datapoint.
    labels = np.array(
        [np.argmin(dist_matrix[j][idx]) for j in range(len(Xs))]
    )

    # Initialize distance to centroids matrix
    cent_dists = np.zeros((n_clusters, Xs.shape[0]))

    # Main loop.
    for i in range(max_iter):

        # Update cluster centroids.
        for k in range(n_clusters):
            centroids[k] = procrustes_barycenter(
                [X for X, c in zip(Xs, labels) if c == k],
                random_state=rs, max_iter=10,
            )

        # Compute distance from each datapoint to each centroid.
        for j in range(len(Xs)):
            for k, cent in enumerate(centroids):
                proc_metric.fit(Xs[j], cent)
                cent_dists[k, j] = proc_metric.score(Xs[j], cent)

        # Compute new cluster labels.
        new_labels = np.argmin(cent_dists, axis=0)

        # Check convergence.
        converged = np.all(labels == new_labels)
        labels = new_labels

        # Break loop if converged.
        if converged:
            break

    return centroids, labels, cent_dists


def pairwise_distances(metric, traindata, testdata=None, verbose=True):
    """
    Compute pairwise distances between a collection of
    networks.

    Parameters
    ----------
    metric : instance of a MetricMixin class.
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

    # Compute all pairwise distances.
    for i in range(m):
        for j in range(i + 1, m):

            # Fit metric.
            metric.fit(traindata[i], traindata[j])

            # Evaluate distance on the training set.
            D_train[i, j] = metric.score(traindata[i], traindata[j])
            D_train[j, i] = D_train[i, j]

            # Evaluate distance on the test set.
            if testdata is not None:
                testdists[i, j] = metric.score(testdata[i], testdata[j])
                testdists[j, i] = testdists[j, i]

            # Update progress bar.
            if verbose:
                pbar.update(1)

    # Close progress bar.
    if verbose:
        pbar.close()

    return D_train if (testdata is None) else (D_train, D_test)
