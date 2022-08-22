import unittest
import logging
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal as assert_equal
from numpy.testing import assert_array_almost_equal as assert_almost_equal
from rescompy import features


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


class TestStatesOnly(unittest.TestCase):
    """Test the features.states_only function."""
            
    def test(self):
        """Test the function on vector data."""

        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create some dummy data.
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100, 3))
        
        # Grab the feature vectors.
        s = features.states_only(r, u)
        
        # Compare to expected results.
        assert_equal(s, r)
                

class TestStatesAndInputs(unittest.TestCase):
    """Test the features.states_and_inputs function."""
        
    def test(self):
        """Test the function on vector data."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create some dummy data.
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100, 3))
        
        # Grab the feature vectors.
        s = features.states_and_inputs(r, u)
        
        # Compare to expected results.
        self.assertEqual(s.shape, (100, 13))
        assert_equal(s[:, :10], r)
        assert_equal(s[:, 10:], u)
                

class TestStatesAndConstant(unittest.TestCase):
    """Test the features.states_and_constant function."""
    
    def test(self):
        """Test the function on vector data."""

        # Create the random state for reproducibility.
        rng = default_rng(SEED)

        # Create some dummy data.        
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100, 3))
        
        # Grab the feature vectors.
        s = features.states_and_constant(r, u)
        
        # Compare to expected results.
        self.assertEqual(s.shape, (100, 11))
        assert_equal(s[:, :10], r)
        assert_equal(s[:, 10], np.zeros((100)) + 1)


class TestStatesAndInputsAndConstant(unittest.TestCase):
    """Test the features.states_and_inputs_and_constant function."""

    def test(self):
        """Test the function on vector data."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)

        # Create some dummy data.
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100, 3))
        
        # Grab the feature vectors.
        s = features.states_and_inputs_and_constant(r, u)
        
        # Compare to expected results.
        self.assertEqual(s.shape, (100, 14))
        assert_equal(s[:, :10], r)
        assert_equal(s[:, 10:13], u)
        assert_equal(s[:, 13], np.zeros((100)) + 1)


class TestGetPolynomial(unittest.TestCase):
    """Test the features.get_polynomial function."""
        
    def test_3(self):
        """Test features.get_polynomial(3) on vector data."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create some dummy data.
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100, 3))

        # Grab the feature vectors.
        s = features.get_polynomial(3)(r, u)
        
        # Compare to expected results.
        self.assertEqual(s.shape, (100, 30))
        assert_equal(s[:, :10], r)
        assert_almost_equal(s[:, 10:20], r**2)
        assert_almost_equal(s[:, 20:30], r**3)


if __name__ == '__main__':
    unittest.main()