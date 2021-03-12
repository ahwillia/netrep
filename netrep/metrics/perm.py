import numpy as np
from netrep.validation import check_equal_shapes
from netrep.utils import angular_distance
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_symmetric


class PermutationMetric(BaseEstimator):

    def __init__(self, center_columns=True, zero_pad=True):
        """
        Parameters
        ----------
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

        if self.center_columns:
            self.mx_ = mx = np.mean(X, axis=0)
            self.my_ = my = np.mean(Y, axis=0)
            X = X - mx[None, :]
            Y = Y - mx[None, :]

        self.Px_, self.Py_ = lsa(X.T @ Y, maximize=True)

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
        check_is_fitted(self, attributes=["Px_"])
        if (X.shape[1] != len(self.Px_)):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(X)))
        if self.center_columns:
            return (X - self.mx_[None, :])[:, self.Px_]
        else:
            return X[:, self.Px_]

    def transform_Y(self, Y):
        """Transform X into the aligned space."""
        check_is_fitted(self, attributes=["Py_"])
        if (Y.shape[1] != len(self.Py_)):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(Y)))
        if self.center_columns:
            return (Y - self.my_[None, :])[:, self.Py_]
        else:
            return Y[:, self.Py_]

