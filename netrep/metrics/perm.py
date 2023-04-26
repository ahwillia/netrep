from __future__ import annotations
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment as lsa
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from netrep.validation import check_equal_shapes
from netrep.utils import angular_distance

class PermutationMetric(BaseEstimator):
    """Computes distance between two sets of optimally permutation-aligned representations.
    """

    def __init__(
            self,
            center_columns: bool = True,
            zero_pad: bool = True,
            score_method: Literal["angular", "euclidean"] = "angular"
        ):
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

        score_method : {'angular','euclidean'}, default='angular'
            String specifying ground metric.
        """

        if score_method not in ("euclidean", "angular"):
            raise ValueError(
                "Expected `score_method` parameter to be in {'angular','euclidean'}. " +
                f"Found instead score_method == '{score_method}'."
            )

        self.center_columns = center_columns
        self.zero_pad = zero_pad
        self.score_method = score_method

    def partial_fit(self, X: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Computes partial whitening transformation for a neural response matrix.
        """
        if self.center_columns:
            mx = np.mean(X, axis=0)
            Xphi = X - mx[None, :]
        else:
            mx = np.zeros(X.shape[1])
            Xphi = X
        return (mx, Xphi)

    def finalize_fit(
        self, 
        cache_X: Tuple[npt.NDArray, npt.NDArray], 
        cache_Y: Tuple[npt.NDArray, npt.NDArray]
    ) -> PermutationMetric:
        """Takes outputs of 'partial_fit' function and finishes fitting permutation 
        matrices (Px, Py) and bias terms (mx, my) to align a pair of neural activations.
        """

        # Extract whitened representations.
        self.mx_, X = cache_X
        self.my_, Y = cache_Y

        # Fit optimal permutation matrices.
        self.Px_, self.Py_ = lsa(X.T @ Y, maximize=True)

        return self

    def fit(self, X: npt.NDArray, Y: npt.NDArray) -> PermutationMetric:
        """Fits permutation matrices (Px, Py) and bias terms (mx, my) to align a pair of 
        neural activation matrices.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        """
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        return self.finalize_fit(
            self.partial_fit(X),
            self.partial_fit(Y)
        )

    def transform(
        self, 
        X: npt.NDArray, 
        Y: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Applies linear alignment transformations to X and Y.

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
        return self._transform_X(X), self._transform_Y(Y)

    def fit_score(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """Fits alignment by calling `fit(X, Y)` and then evaluates
        the distance by calling `score(X, Y)`.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        
        Returns
        -------
        dist : float
            Distance between optimally aligned X and Y.
        """
        return self.fit(X, Y).score(X, Y)

    def score(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """Computes the distance between X and Y in the aligned
        space.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Distance between X and Y.
        """
        if self.score_method == "angular":
            return angular_distance(*self.transform(X, Y))
        else: # self.score_method == "euclidean":
            return np.linalg.norm(
                np.subtract(*self.transform(X, Y)), ord="fro"
            )


    def _transform_X(self, X: npt.NDArray) -> npt.NDArray:
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

    def _transform_Y(self, Y: npt.NDArray) -> npt.NDArray:
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

