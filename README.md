# Generalized Shape Metrics on Neural Representations

![Generalized Shape Metrics on Neural Representations](https://user-images.githubusercontent.com/636625/139737239-5e3054fe-0465-4c9b-b148-a43acc62aa8e.png)

In neuroscience and in deep learning, quantifying the (dis)similarity of neural representations across networks is a topic of substantial interest.

This code package computes [*metrics*](https://en.wikipedia.org/wiki/Metric_(mathematics)) &mdash; notions of distance that satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality) &mdash; between neural representations. If we record the activity of `K` networks, we can compute all pairwise distances and collect them into a `K × K` distance matrix. The triangle inequality ensures that all of these distance relationships are, in some sense, self-consistent. This self-consistency enables us to apply off-the-shelf algorithms for clustering and dimensionality reduction, which are available through many open-source packages such as [scikit-learn](https://scikit-learn.org/).

We published a [**conference paper (Neurips '21)**](https://arxiv.org/abs/2110.14739) describing these ideas.

```
@inproceedings{neural_shape_metrics,
  author = {Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman},
  title = {Generalized Shape Metrics on Neural Representations},
  year = {2021},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  url = {https://arxiv.org/abs/2110.14739}
}
```

We also presented an early version of this work at Cosyne (see [**7 minute summary on youtube**](https://www.youtube.com/watch?v=Lt_Vo-tQcW0)) in early 2021.

**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (alex.h.willia@gmail.com) if you have questions.

## A short and preliminary guide

To install, set up standard python libraries (https://ipython.org/install.html) and then install via `pip`:

```
git clone https://github.com/ahwillia/netrep
cd netrep/
pip install -e .
```

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.

### Computing the distance between two networks 

The metrics implemented in this library are extensions of [Procrustes distance](https://en.wikipedia.org/wiki/Procrustes_analysis). Some useful background can be found in Dryden &amp; Mardia's textbook on [*Statistical Shape Analysis*](https://www.wiley.com/en-us/Statistical+Shape+Analysis%3A+With+Applications+in+R%2C+2nd+Edition-p-9780470699621). A forthcoming preprint will describe the various metrics in more detail. For now, please see the short video description above and reach out to us if you have more questions.

The code uses an API similar to [scikit-learn](https://scikit-learn.org/), so we recommend familiarizing yourself with that package.

We start by defining a metric object. The simplest metric to use is `LinearMetric`, which has a hyperparameter `alpha` which regularizes the alignment operation:

```python
from netrep.metrics import LinearMetric

# Rotationally invariant metric (fully regularized).
proc_metric = LinearMetric(alpha=1.0, center_columns=True)

# Linearly invariant metric (no regularization).
cca_metric = LinearMetric(alpha=0.0, center_columns=True)
```

Valid values for the regularization term are `0 <= alpha <= 1`. When `alpha == 0`, the resulting metric is similar to CCA and allows for an invertible linear transformation to align the activations. When `alpha == 1`, the model is fully regularized and only allows for rotational alignments.

We reccomend starting with the fully regularized model where `alpha == 1`.

Next, we define the data, which are stored in matrices `X` and `Y` that hold paired activations from two networks. Each row of `X` and `Y` contains a matched sample of neural activations. For example, we might record the activity of 500 neurons in visual cortex in response to 1000 images (or, analogously, feed 1000 images into a deep network and store the activations of 500 hidden units). We would collect the neural responses into a `1000 x 500` matrix `X`. We'd then repeat the experiment in a second animal and store the responses in a second matrix `Y`.

By default if the number of neurons in `X` and `Y` do not match, we zero-pad the dataset with fewer neurons to match the size of the larger dataset. This can be justified on the basis that zero-padding does not distort the geometry of the dataset, it simply embeds it into a higher dimension so that the two may be compared. Alternatively, one could preprocess the data by using PCA (for example) to project the data into a common, lower-dimensional space. The default zero-padding behavior can be deactivated as follows:

```python
LinearMetric(alpha=1.0, zero_pad=True)  # default behavior

LinearMetric(alpha=1.0, zero_pad=False)  # throws an error if number of columns in X and Y don't match
```

Now we are ready to fit alignment transformations (which account for the neurons being mismatched across networks). Then, we evaluate the distance in the aligned space. These are respectively done by calling `fit(...)` and `score(...)` functions on the metric instance.

```python
# Given
# -----
# X : ndarray, (num_samples x num_neurons), activations from first network.
#
# Y : ndarray, (num_samples x num_neurons), activations from second network.
#
# metric : an instance of LinearMetric(...)

# Fit alignment transformations.
metric.fit(X, Y)

# Evaluate distance between X and Y, using alignments fit above.
dist = metric.score(X, Y)
```

Since the model is fit and evaluated by separate function calls, it is very easy to cross-validate the estimated distances:

```python
# Given
# -----
# X_train : ndarray, (num_train_samples x num_neurons), training data from first network.
#
# Y_train : ndarray, (num_train_samples x num_neurons), training data from second network.
#
# X_test : ndarray, (num_test_samples x num_neurons), test data from first network.
#
# Y_test : ndarray, (num_test_samples x num_neurons), test data from second network.
#
# metric : an instance of LinearMetric(...)

# Fit alignment transformations to the training set.
metric.fit(X_train, Y_train)

# Evaluate distance on the test set.
dist = metric.score(X_test, Y_test)
```

In fact, we can use scikit-learn's built-in cross-validation tools, since `LinearMetric` extends the `sklearn.base.BaseEstimator` class. So, if you'd like to do 10-fold cross-validation, for example:

```python
from sklearn.model_selection import cross_validate
results = cross_validate(metric, X, Y, return_train_score=True, cv=10)
results["train_score"]  # holds 10 distance estimates between X and Y, using training data.
results["test_score"]   # holds 10 distance estimates between X and Y, using heldout data.
```

We can also call `transform(...)` function to align the activations

```python
# Fit alignment transformations.
metric.fit(X, Y)

# Apply alignment transformations.
X_aligned, Y_aligned = metric.transform(X, Y)

# Now, e.g., you could use PCA to visualize the data in the aligned space...
```

## Stochastic shape metrics

We also provide a way to compare between stochastic neural responses (e.g. biological neural network responses to stimulus repetitions, latent activations in variational autoencoders). The API is similar to `LinearMetric()`, requires slightly differently-formatted inputs.

**1) Stochastic shape metrics using** `GaussianStochasticMetric()`


```python
# Given
# -----
# Xi : Tuple[ndarray, ndarray]
# The first array is (num_classes x num_neurons) array of means and the second array is (num_classes x num_neurons x num_neurons) covariances of first network.
#
# Xj : Tuple[ndarray, ndarray]
# Same as Xi, but for the second network's responses.
#
# alpha: float between [0, 2]. 
#    When alpha=2, this reduces to the deterministic shape metric. When alpha=1, this is the 2-Wasserstein between two Gaussians. When alpha=0, this is the Bures metric between the two sets of covariance matrices.

# Fit alignment transformations to the training set.
metric = GaussianStochasticMetric(alpha)
metric.fit(Xi, Xj)

# Evaluate the distance between the two networks
dist = metric.score(Xi, Xj)
```

**2) Stochastic shape metrics using** `EnergyStochasticMetric()`



### Computing distances between many networks

Things start to get really interesting when we start to consider larger cohorts containing more than just two networks. The `netrep.multiset` file contains some useful methods. Let `Xs = [X1, X2, X3, ..., Xk]` be a list of `num_samples x num_neurons` matrices similar to those described above. We can do the following:

**1) Computing all pairwise distances.** The following returns a symmetric `k x k` matrix of distances.

```python
metric = LinearMetric(alpha=1.0)
dist_matrix = pairwise_distances(metric, Xs, verbose=False)
```

By setting `verbose=True`, we print out a progress bar which might be useful for very large datasets.

We can also split data into training sets and test sets.

```python
# Split data into training and testing sets
splitdata = [np.array_split(X, 2) for X in Xs]
traindata = [X_train for (X_train, X_test) in splitdata]
testdata = [X_test for (X_train, X_test) in splitdata]

# Compute all pairwise train and test distances.
train_dists, test_dists = pairwise_distances(metric, traindata, testdata=testdata)
```

**2) Using the pairwise distance matrix.** Many of the methods in [`sklearn.cluster`](https://scikit-learn.org/stable/modules/clustering.html#clustering) and [`sklearn.manifold`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold) will work and operate directly on these distance matrices.

For example, to perform *clustering* over the cohort of networks, we could do:

```python
# Given
# -----
# dist_matrix : (num_networks x num_networks) symmetric distance matrix, computed as described above.

# DBSCAN clustering
from sklearn.cluster import DBSCAN
cluster_ids = DBSCAN(metric="precomputed").fit_transform(dist_matrix)

# Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
cluster_ids = AgglomerativeClustering(n_clusters=5, affinity="precomputed").fit_transform(dist_matrix)

# OPTICS
from sklearn.cluster import OPTICS
cluster_ids = OPTICS(metric="precomputed").fit_transform(dist_matrix)

# Scipy hierarchical clustering
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
hierarchy.ward(squareform(dist_matrix)) # return linkage
```

We can also visualize the set of networks in 2D space by using manifold learning methods:

```python
# Given
# -----
# dist_matrix : (num_networks x num_networks) symmetric distance matrix, computed as described above.

# Multi-dimensional scaling
from sklearn.manifold import MDS
lowd_embedding = MDS(dissimilarity="precomputed").fit_transform(dist_matrix)

# t-distributed Stochastic Neighbor Embedding
from sklearn.manifold import TSNE
lowd_embedding = TSNE(dissimilarity="precomputed").fit_transform(dist_matrix)

# Isomap
from sklearn.manifold import Isomap
lowd_embedding = Isomap(dissimilarity="precomputed").fit_transform(dist_matrix)

# etc., etc.
```

**3) K-means clustering and averaging across networks**

We can average across networks using the metric spaces defined above. Specifically, we can compute a [Fréchet/Karcher mean](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean) in the metric space. See also the section on *"Generalized Procrustes Analysis"* in Gower & Dijksterhuis (2004). 

```python
from netrep.multiset import procrustes_average
Xbar = procrustes_average(Xs, max_iter=100, tol=1e-4)
```

Further, we can extend the well-known [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) algorithm to the metric space defined by Procrustes distance.

```python
from netrep.multiset import procrustes_kmeans

# Fit 3 clusters
n_clusters = 3
centroids, labels, cent_dists = procrustes_kmeans(Xs, n_clusters)
```

## An incomplete list of related work

Dabagia, Max, Konrad P. Kording, and Eva L. Dyer (forthcoming). "[Comparing high-dimensional neural recordings by aligning their low-dimensional latent representations](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/9/630/files/2020/05/Dabagia_Comparing2020.pdf).” Nature Biomedical Engineering

Degenhart, A. D., Bishop, W. E., Oby, E. R., Tyler-Kabara, E. C., Chase, S. M., Batista, A. P., & Byron, M. Y. (2020). [Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity](https://www.nature.com/articles/s41551-020-0542-9?proof=t). Nature biomedical engineering, 4(7), 672-685.

Gower, J. C., & Dijksterhuis, G. B. (2004). Procrustes problems (Vol. 30). Oxford University Press.

Gallego, J. A., Perich, M. G., Chowdhury, R. H., Solla, S. A., & Miller, L. E. (2020). [Long-term stability of cortical population dynamics underlying consistent behavior](https://www.nature.com/articles/s41467-018-06560-z). Nature neuroscience, 23(2), 260-270.

Haxby, J. V., Guntupalli, J. S., Nastase, S. A., & Feilong, M. (2020). [Hyperalignment: Modeling shared information encoded in idiosyncratic cortical topographies](https://elifesciences.org/articles/56601). Elife, 9, e56601.

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019, May). [Similarity of neural network representations revisited](https://arxiv.org/abs/1905.00414). In International Conference on Machine Learning (pp. 3519-3529). PMLR.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). [Representational similarity analysis-connecting the branches of systems neuroscience](https://doi.org/10.3389/neuro.06.004.2008). Frontiers in systems neuroscience, 2, 4.

Maheswaranathan, N., Williams, A. H., Golub, M. D., Ganguli, S., & Sussillo, D. (2019). [Universality and individuality in neural dynamics across large populations of recurrent networks](https://arxiv.org/abs/1907.08549). Advances in neural information processing systems, 2019, 15629.

Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). [Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability](https://arxiv.org/abs/1706.05806). arXiv preprint arXiv:1706.05806.

