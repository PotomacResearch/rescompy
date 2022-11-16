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
        assert_equal(features.states_only.feature_size(100, 50), 100)

        
                

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
        assert_equal(features.states_and_inputs.feature_size(100, 50), 150)

    def test_jacobian(self):
        rng = default_rng(SEED)
        num_inputs = 3
        num_timesteps = 100
        dim_res = 10

        r = rng.uniform(size=(num_timesteps, dim_res))
        u = rng.uniform(size=(num_timesteps,num_inputs))
        dr_du = rng.uniform(size=(num_timesteps, dim_res, num_inputs))


        dg_du = features.states_and_inputs.jacobian(dr_du, u)



        self.assertEqual(dg_du.shape, (num_timesteps, dim_res+num_inputs, num_inputs))
        assert_equal(dg_du[:, :dim_res, :], dr_du)
        assert_equal(dg_du[:, dim_res:, :], np.tile(np.eye(num_inputs), (num_timesteps, 1, 1)))



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
        assert_equal(features.states_and_constant.feature_size(100, 50), 101)

    def test_jacobian(self):
        rng = default_rng(SEED)
        num_inputs = 3
        num_timesteps = 100
        dim_res = 10

        r = rng.uniform(size=(num_timesteps, dim_res))
        u = rng.uniform(size=(num_timesteps,num_inputs))
        dr_du = rng.uniform(size=(num_timesteps, dim_res, num_inputs))


        dg_du = features.states_and_constant.jacobian(dr_du, u)

        self.assertEqual(dg_du.shape, (num_timesteps, dim_res+1, num_inputs))
        assert_equal(dg_du[:, :dim_res, :], dr_du)
        assert_equal(dg_du[:, dim_res, :], np.zeros((num_timesteps, num_inputs)))

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
        assert_equal(features.states_and_inputs_and_constant.feature_size(100, 50), 151)

    def test_jacobian(self):
        rng = default_rng(SEED)
        num_inputs = 3
        num_timesteps = 100
        dim_res = 10

        r = rng.uniform(size=(num_timesteps, dim_res))
        u = rng.uniform(size=(num_timesteps,num_inputs))
        dr_du = rng.uniform(size=(num_timesteps, dim_res, num_inputs))


        dg_du = features.states_and_inputs_and_constant.jacobian(dr_du, u)

        assert_equal(dg_du.shape, (num_timesteps, dim_res+num_inputs+1, num_inputs))

        assert_equal(dg_du[:, :dim_res, :], dr_du)
        assert_equal(dg_du[:, dim_res:dim_res+num_inputs, :], np.tile(np.eye(num_inputs), (num_timesteps, 1, 1)))
        assert_equal(dg_du[:, dim_res+num_inputs, :], np.zeros((num_timesteps, num_inputs)))

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
        raise NotImplementedError('need to add feature size')

    def test_jacobian(self):
        rng = default_rng(SEED)
        r = rng.uniform(size=(100, 10))
        u = rng.uniform(size=(100,3))
        dr_du = rng.uniform(size=(100, 10, 3))

        dg_du = features.states_only.jacobian(dr_du, u)
        raise NotImplementedError()

if __name__ == '__main__':
    unittest.main()