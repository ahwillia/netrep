import numpy as np
from joblib import Parallel, delayed
import itertools
from netrep.metrics import LinearMetric
from sklearn.base import BaseEstimator
from netrep.base import MetricMixin
from netrep.validation import check_equal_shapes
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted, check_random_state
from netrep.utils import angular_distance


def convolve_metric(metric, X, Y):
    """
    Computes representation metric between convolutional layers,
    convolving activations with boundary conditions.

    Parameters
    ----------
    metric : Metric
        Specifies metric to compute.
    X : ndarray
        Activations from first layer (images x height x width x channel)
    Y : ndarray
        Activations from second layer (images x height x width x channel)

    Returns
    -------
    dists : ndarray
        Matrix with shape (height x width). Holds `metric.score()` for
        X and Y, convolving over the two spatial dimensions.
    """

    # Inputs are (images x height x width x channel) tensors, holding activations.
    X, Y = check_equal_shapes(X, Y, nd=4, zero_pad=metric.zero_pad)
    m, h, w, c = X.shape

    # Flattened Y tensor.
    Yf = Y.reshape(-1, c)

    # Compute metric over all possible offsets.
    pbar = tqdm(total=(w * h))
    dists = np.full((h, w), -1.0)
    for i, j in itertools.product(range(h), range(w)):

        # Apply shift to X tensor, then flatten.
        shifts = (i - (h // 2), j - (w // 2))
        Xf = np.roll(X, shifts, axis=(1, 2)).reshape(-1, c)

        # Fit and evaluate metric.
        metric.fit(Xf, Yf)
        dists[i, j] = metric.score(Xf, Yf)

        # Update progress bar.
        pbar.update()

    pbar.close()
    return dists


# class ConvLinearMetric(BaseEstimator, MetricMixin):
    
#     def __init__(
#             self, alpha=1.0, center_columns=True, zero_pad=False,
#             n_jobs=3, verbose=False
#         ):

#         if (alpha > 1) or (alpha < 0):
#             raise ValueError(
#                 "Regularization parameter `alpha` must be between zero and one.")

#         self.alpha = alpha
#         self.center_columns = center_columns
#         self.zero_pad = zero_pad
#         self.n_jobs = n_jobs
#         self.verbose = verbose

#     def fit(self, X, Y):

#         X, Y = check_equal_shapes(X, Y, nd=4, zero_pad=self.zero_pad)
#         m, h, w, c = X.shape
#         self.height_, self.width_ = h, w

#         if self.center_columns:
#             X -= np.mean(X, axis=(0, 1, 2), keepdims=True)
#             Y -= np.mean(X, axis=(0, 1, 2), keepdims=True)

#         parallel = Parallel(
#             n_jobs=self.n_jobs,
#             verbose=self.verbose
#         )
#         scores = parallel(
#             delayed(_fit_and_score_shifted_metric)(
#                 X, Y, hs, ws, self.alpha, self.center_columns, self.zero_pad
#             ) for (ws, hs) in itertools.product(range(h), range(w))
#         )
#         # distances = [_fit_and_score_shifted_metric(
#         #     X, Y, hs, ws, self.alpha, self.center_columns, self.zero_pad
#         # ) for (ws, hs) in shifts]

#         idx = np.argmax(d[0] for d in scores)
#         self.h_shift_ = scores[idx][1]
#         self.w_shift_ = scores[idx][2]

#         # Re-fit the linear metric at the optimal shift. Although we already
#         # computed this above within the parallel loop, it is too costly
#         # memory-wise to keep them all around. Recomputing should be reasonably fast.
#         self.channel_metric_ = LinearMetric(
#             alpha=self.alpha,
#             center_columns=self.center_columns,
#             zero_pad=False  # we zero pad before calling channel_metric_
#         )
#         self.channel_metric_.fit(*self.shift_and_flatten(X, Y))

#     def shift_and_flatten(self, X, Y):

#         # Check inputs.
#         check_is_fitted(self, attributes=["h_shift_", "w_shift_"])
#         X, Y = check_equal_shapes(X, Y, nd=4, zero_pad=self.zero_pad)
#         if (X.shape[1] != self.height_) or (X.shape[2] != self.width_):
#             raise ValueError(
#                 "Height and width dimensions do not match what was fit..."
#             )

#         # Apply shift and flattening.
#         c = X.shape[-1]
#         return (
#             np.roll(X, (0, self.h_shift_, self.w_shift_, 0)).reshape(-1, c),
#             Y.reshape(-1, c)
#         )

#     def transform(self, X, Y, unflatten=False):
#         mX, mY = self.shift_and_flatten(X, Y)
#         tX, tY = self.channel_metric_.transform(mX, mY)
#         if unflatten:
#             return (
#                 tX.reshape(X.shape[0], self.height_, self.width_, -1),
#                 tY.reshape(Y.shape[0], self.height_, self.width_, -1)
#             )
#         else:
#             return tX, tY

#     def score(self, X, Y):
#         return angular_distance(*self.transform(X, Y))


# def _fit_and_score_shifted_metric(
#         X, Y, h_shift, w_shift,
#         alpha, center_columns, zero_pad
#     ):

#     # Apply shift to X. Then reshape X and Y into matrices.
#     c = X.shape[-1]
#     mX = np.roll(X, (0, h_shift, w_shift, 0)).reshape(-1, c)
#     mY = Y.reshape(-1, c)

#     # # Define alignment metic across channels.
#     # metric = LinearMetric(
#     #     alpha=alpha,
#     #     center_columns=center_columns,
#     #     zero_pad=zero_pad
#     # )
#     # # Fit metric, return score and params.
#     # metric.fit(mX, mY)
#     # return (metric.score(mX, mY), h_shift, w_shift)

#     score = np.sum(np.linalg.svd(mX.T @ mY, compute_uv=False))
#     return (score, h_shift, w_shift)

