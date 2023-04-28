# Generalized Shape Metrics on Neural Representations

![Generalized Shape Metrics on Neural Representations](https://user-images.githubusercontent.com/636625/139737239-5e3054fe-0465-4c9b-b148-a43acc62aa8e.png)

In neuroscience and in deep learning, quantifying the (dis)similarity of neural representations across networks is a topic of substantial interest.

This code package computes [*metrics*](https://en.wikipedia.org/wiki/Metric_(mathematics)) &mdash; notions of distance that satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality) &mdash; between neural representations. If we record the activity of `K` networks, we can compute all pairwise distances and collect them into a `K × K` distance matrix. The triangle inequality ensures that all of these distance relationships are, in some sense, self-consistent. This self-consistency enables us to apply off-the-shelf algorithms for clustering and dimensionality reduction, which are available through many open-source packages such as [scikit-learn](https://scikit-learn.org/).

Two conference papers **([Neurips '21](https://arxiv.org/abs/2110.14739), [ICLR '23](https://arxiv.org/abs/2211.11665))** describe the approach

```
@inproceedings{neural_shape_metrics,
  author = {Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman},
  title = {Generalized Shape Metrics on Neural Representations},
  year = {2021},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
}

@inproceedings{stochastic_neural_shape_metrics,
  author = {Lyndon R. Duong and Jingyang Zhou and Josue Nassar and Jules Berman and Jeroen Olieslagers and Alex H. Williams},
  title = {Representational dissimilarity metric spaces for stochastic neural networks},
  year = {2023},
  booktitle = {International Conference on Learning Representations},
}
```

We presented an early version of this work at COSYNE 2021 (see [**7 minute summary on youtube**](https://www.youtube.com/watch?v=Lt_Vo-tQcW0)), and a full workshop talk at  COSYNE 2023 ([**30 minute talk on youtube**](https://www.youtube.com/watch?v=e02DWc2z8Hc)).

**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (alex.h.willia@gmail.com) if you have questions.

## A short and preliminary guide

To install, set up standard python libraries (<https://ipython.org/install.html>) and then install via `pip`:

```
git clone https://github.com/ahwillia/netrep
cd netrep/
pip install -e .
```

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.

### Computing the distance between two networks

The metrics implemented in this library are extensions of [Procrustes distance](https://en.wikipedia.org/wiki/Procrustes_analysis). Some useful background can be found in Dryden &amp; Mardia's textbook on [*Statistical Shape Analysis*](https://www.wiley.com/en-us/Statistical+Shape+Analysis%3A+With+Applications+in+R%2C+2nd+Edition-p-9780470699621).

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

We also provide a way to compare between stochastic neural responses (e.g. biological neural network responses to stimulus repetitions, or latent activations in variational autoencoders). The API is similar to `LinearMetric()`, but requires differently-formatted inputs.

**1) Stochastic shape metrics using** `GaussianStochasticMetric()`

The first method models network response distributions as multivariate Gaussians, and computes distances based on the analytic solution to the 2-Wasserstein distance between two Gaussians. This involves computing class-conditional means and covariances for each network, then computing the metric as follows.

```python
# Given
# -----
# Xi : Tuple[ndarray, ndarray]
#    The first array is (num_classes x num_neurons) array of means and the second array is (num_classes x num_neurons x num_neurons) covariance matrices of first network.
#
# Xj : Tuple[ndarray, ndarray]
#    Same as Xi, but for the second network's responses.
#
# alpha: float between [0, 2]. 
#    When alpha=2, this reduces to the deterministic shape metric. When alpha=1, this is the 2-Wasserstein between two Gaussians. When alpha=0, this is the Bures metric between the two sets of covariance matrices.

# Fit alignment
metric = GaussianStochasticMetric(alpha)
metric.fit(Xi, Xj)

# Evaluate the distance between the two networks
dist = metric.score(Xi, Xj)
```

**2) Stochastic shape metrics using** `EnergyStochasticMetric()`

We also provide stochastic shape metrics based on the Energy distance. This metric is non-parametric (does not make any response distribution assumptions). It can therefore take into account higher-order moments between neurons.

```python
# Given
# -----
# Xi : ndarray, (num_classes x num_repeats x num_neurons)
#    First network's responses.
#
# Xj : ndarray, (num_classes x num_repeats x num_neurons)
#    Same as Xi, but for the second network's responses.
#

# Fit alignment
metric = EnergyStochasticMetric()
metric.fit(Xi, Xj)

# Evaluate the distance between the two networks
dist = metric.score(Xi, Xj)
```

### Computing distances between many networks

Things start to get really interesting when we start to consider larger cohorts containing more than just two networks. The `netrep.multiset` file contains some useful methods. Let `Xs = [X1, X2, X3, ..., Xk]` be a list of `num_samples x num_neurons` matrices similar to those described above. We can do the following:

**Computing all pairwise distances.** The following returns a symmetric `k x k` matrix of distances.

```python
metric = LinearMetric(alpha=1.0)

# Compute kxk distance matrices (leverages multiprocessing).
dist_matrix, _ = metric.pairwise_distances(Xs)
```

By setting `verbose=True`, we print out a progress bar which might be useful for very large datasets.

We can also split data into training sets and test sets.

```python
# Split data into training and testing sets
splitdata = [np.array_split(X, 2) for X in Xs]
traindata = [X_train for (X_train, X_test) in splitdata]
testdata = [X_test for (X_train, X_test) in splitdata]

# Compute all pairwise train and test distances.
train_dists, test_dists = metric.pairwise_distances(traindata, testdata)
```
