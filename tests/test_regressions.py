import os
import unittest
import logging
import csv
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal as assert_almost_equal
from rescompy import regressions


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17

class TestTikhonov(unittest.TestCase):
    """Test the regressions.default function."""
    
    def test(self):
        """Test the function on random data."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Grab a default regressor with default arguments.
        regressor = regressions.tikhonov()
        
        # Create some linear data.
        As = rng.normal(size=(12))
        Bs = rng.normal(size=(12))
        t = np.linspace(0, 1, 100, False)
        s = np.zeros((100, 10))
        v = np.zeros((100, 2))
        for i in range(12):
            if i < 10:
                s[:, i] = As[i]*t + Bs[i]
            else:
                v[:, i-10] = As[i]*t + Bs[i]
        
        # Fit with regressor.
        W = regressor(s, v)
        
        # Compare fitted outputs to target outputs.
        assert_almost_equal(np.dot(s, W), v)

class TestJacobian(unittest.TestCase):
    def test(self):

        def _load(file):
            name = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'data', file)
            res = []
            with open(name) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    res.append([float(r) for r in row])
            return np.array(res).T


        v = _load('jacobian_target.csv')
        s = _load('jacobian_features.csv')
        D = _load('jacobian_feature_derivatives.csv')  # Eq. (14) in [1]
        R_J = _load('jacobian_reg_matrix.csv')         # Eq. (11) in [1]
        B = _load('B.csv')


        timesteps,esn_size = D.shape
        dim = v.shape[1]

        # construct dg_du from the D derivative data
        # the feature is ordered [1 r r^2 u] to feature size is 265
        feature_size = 2*esn_size+dim+1
        dg_du = np.zeros((timesteps, feature_size, dim))
        # leakage (alpha) is 1.0
        for i in range(timesteps):
            DB = (D[i,:] * B).T
            dg_du[i, 1:esn_size+1, :] = DB
            r = s[i, 1:esn_size+1]
            dg_du[i, esn_size+1:-dim, :] = (2*r * DB.T).T
            dg_du[i,-dim:,:] = np.eye(dim)


        beta_T = 1e-6
        beta_J = 1e-8

        regressor = regressions.jacobian(beta_T, beta_J)

        # the Tikhonov part
        R_T = np.eye(s.shape[1])
        # calculate expected W_out using reference R_J and R_T
        W_ref = np.linalg.solve(s.T @ s + beta_T*R_T + beta_J*R_J, s.T @ v)


        W = regressor(s, v, dg_du)
        assert_almost_equal(W, W_ref)

if __name__ == '__main__':
    unittest.main()