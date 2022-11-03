import unittest
import logging
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
        # Create the random state for reproducibility.
        rng = default_rng(SEED)

        regressor = regressions.jacobian(1e-6, 1e-8)

        # ESN size is 100, 50 timesteps and 2 dimensions
        esn_size = 100
        timesteps = 50
        dim = 2
        s = rng.uniform(size=(timesteps, esn_size))
        v = rng.uniform(size=(timesteps, dim))
        dg_du = rng.uniform(size=(timesteps, esn_size, dim))

        # the Tikhonov part
        R_T = s.T @ s + np.eye(s.shape[1])

        # the Jacobian part (calculated explicitly, to compare
        # to the fast np.tensordot based method used)
        R_J = np.zeros((esn_size, esn_size))
        for i in range(timesteps):
            R_J += dg_du[i,:,:] @ dg_du[i,:,:].T

        W = regressor(s, v, dg_du)
        assert_almost_equal(s @ W, v)

if __name__ == '__main__':
    unittest.main()