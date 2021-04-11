from netrep.barycenter import barycenter
from netrep.metrics import LinearMetric


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
            centroids[k] = barycenter(
                [X for X, c in zip(Xs, labels) if c == k],
                group="orth", random_state=rs, max_iter=10,
                warmstart=centroids[k]
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
