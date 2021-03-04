import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_symmetric
from netrep.utils import whiten, angular_distance, centered_kernel
from netrep.validation import check_equal_shapes
from numpy.testing import assert_array_almost_equal
from sklearn.base import BaseEstimator

class LinearMetric(BaseEstimator):

    def __init__(self, alpha=1.0, center_columns=True, zero_pad=True):
        """
        Parameters
        ----------
        alpha : float
            Regularization parameter between zero and one. When
            (alpha == 1.0) the metric only allows for rotational
            alignments. When (alpha == 0.0) the metric allows for
            any invertible linear transformation.

        center_columns : bool
            If True, learn a mean-centering operation in addition
            to the linear/rotational alignment.

        zero_pad : bool
            If False, an error is thrown if representations are
            provided with different dimensions. If True, the smaller
            matrix is zero-padded prior to allow for an alignment.
            Some amount of regularization (alpha > 0) is required to
            align zero-padded representations.
        """

        if (alpha > 1) or (alpha < 0):
            raise ValueError(
                "Regularization parameter `alpha` must be between zero and one.")

        self.alpha = alpha
        self.center_columns = center_columns
        self.zero_pad = zero_pad

    def fit(self, X, Y):
        """
        Fits transformation matrices (Wx, Wy) and, if
        center_columns == True, bias terms (mx_, my_).

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        """

        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        n_obs, n_feats = X.shape

        if self.center_columns:
            # Subtract off mean before whitening.
            self.mx_ = mx = np.mean(X, axis=0)
            self.my_ = my = np.mean(Y, axis=0)
            Xw, Zx = whiten(X - mx[None, :], self.alpha, preserve_variance=True)
            Yw, Zy = whiten(Y - my[None, :], self.alpha, preserve_variance=True)

        else:
            # Don't subtract off mean.
            Xw, Zx = whiten(X, self.alpha, preserve_variance=True)
            Yw, Zy = whiten(Y, self.alpha, preserve_variance=True)

        # Compute SVD of cross-covariance matrix.
        U, _, Vt = np.linalg.svd(Xw.T @ Yw)

        # Compute alignment transformations.
        self.Wx_ = Zx @ U
        self.Wy_ = Zy @ Vt.T

        return self

    def transform(self, X, Y):
        """
        Applies linear alignment transformations to X and Y.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        tX : ndarray
            Transformed version of X.
        tY : ndarray
            Transformed version of Y.
        """
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        return self.transform_X(X), self.transform_Y(Y)

    def fit_score(self, X, Y):
        """
        Fits alignment by calling `fit(X, Y)` and then evaluates
        the distance by calling `score(X, Y)`.
        """
        return self.fit(X, Y).score(X, Y)

    def score(self, X, Y):
        """
        Computes the angular distance between X and Y in
        the aligned space.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Angular distance between X and Y.
        """
        return angular_distance(*self.transform(X, Y))

    def score_euclidean(self, X, Y):
        """
        Computes the average Euclidean distance between the
        rows of X and Y in the aligned space.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Angular distance between X and Y.
        """
        resid = np.subtract(*self.transform(X, Y))
        return np.mean(np.linalg.norm(resid, axis=1))

    def transform_X(self, X):
        """Transform X into the aligned space."""
        check_is_fitted(self, attributes=["Wx_"])
        if (X.shape[1] != self.Wx_.shape[0]):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(X)))
        if self.center_columns:
            return (X - self.mx_[None, :]) @ self.Wx_
        else:
            return (X @ self.Wx_)

    def transform_Y(self, Y):
        """Transform X into the aligned space."""
        check_is_fitted(self, attributes=["Wy_"])
        if (Y.shape[1] != self.Wy_.shape[0]):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(Y)))
        if self.center_columns:
            return (Y - self.my_[None, :]) @ self.Wy_
        else:
            return Y @ self.Wy_
