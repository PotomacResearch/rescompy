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

class TestDefault(unittest.TestCase):
    """Test the regressions.default function."""
    
    def test(self):
        """Test the function on random data."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Grab a default regressor with default arguments.
        regressor = regressions.default()
        
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


if __name__ == '__main__':
    unittest.main()