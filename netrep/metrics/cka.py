import numpy as np
from sklearn.base import BaseEstimator

class LinearCKA:

    def __init__(self, center_columns=True):
        self.center_columns = center_columns

    def fit(self, X, Y):
        pass

    def score(self, X, Y):
        """
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

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        # Compute angular distance between (sample x sample) covariance matrices.
        return angular_distance(X @ X.T, Y @ Y.T)
