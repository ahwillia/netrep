import numpy as np
from sklearn.base import BaseEstimator

from netrep.utils import angular_distance
from netrep.metrics.hsic import HSIC

class LinearCKA:
    """
    Calculation of the linear CKA score
        Default calculation differs differs from the one outlined in
        Kornblith et al. (2019). It introduces an arccos(.)
        into the final calculation so that the result satisfies
        the conditions of a metric.
    """

    def __init__(self, center_columns=True):
        self.center_columns = center_columns

    def fit(self, X, Y):
        pass

    def score(self, X, Y, mode = 'angular'):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons_x) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons_y) matrix of activations.

        mode : str
            particular methods of calculating CKA
                default = 'angular'

            'angular' : arccos over the CKA estimator from Kornblith et al 2019
            'biased'  : CKA estimator constructed via biased HSIC estimator from Gretton et al 2005
            'unbiased': CKA estimator constructed via debiased HSIC estimator from Song et al 2012

        Returns
        -------
        dist : float
            Distance between X and Y.
        """

        assert X.shape[0] == Y.shape[0], f'Number of samples do not align! (X.shape = {X.shape}, Y.shape = {Y.shape})'

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        if mode == 'angular':
            # Compute angular distance between (sample x sample) covariance matrices.
            return angular_distance(X @ X.T, Y @ Y.T)

        elif mode == 'biased' or mode == 'unbiased':
            # HSIC-based computation of CKA inherits bias/debiased structure of HSIC
            hsic = HSIC()
            X_at_Y = hsic.score(X, Y, mode = mode)
            X_at_X = hsic.score(X, X, mode = mode)
            Y_at_Y = hsic.score(Y, Y, mode = mode)
            return X_at_Y/np.sqrt(X_at_X * Y_at_Y)
        
        else:
            raise ValueError(f'mode {mode} does not exist')
