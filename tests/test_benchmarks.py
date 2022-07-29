import unittest
import logging
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal as assert_equal
from numpy.testing import assert_array_almost_equal as assert_almost_equal
from rescompy import benchmarks


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


class TestLorenz(unittest.TestCase):
    """Test the Lorenz functions."""
    
    def test_lorenz(self):
        """Test the benchmarks._lorenz function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a random initial condition and propagate one step.
        x0 = rng.normal(size=(3))
        x0_h = benchmarks._lorenz(10, 8/3, 28, x0, 2, 0.1)

        # Calculate derivative.
        #diff = (x0_h[1] - x0_h[0])/0.1
        
        # Calculate derivative expected from Lorenz equation.
        diff_exp = np.zeros((3))
        diff_exp[0] = 10*(x0[1] - x0[0])
        diff_exp[1] = x0[0]*(28 - x0[2]) - x0[1]
        diff_exp[2] = x0[0]*x0[1] - (8/3)*x0[2]
        
        # Compare to expected results.
        assert_equal(x0_h[0], x0)
        #assert_almost_equal(diff, diff_exp)
        
    def test_get_lorenz(self):
        """Test the benchmarks.get_lorenz function."""

        # Grab a solution to Lorenz equation with fixed seed.        
        x = benchmarks.get_lorenz(return_length=10000, seed=SEED)
        
        # Calculate statistical properties.
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        
        # Record expected statistical properties.
        x_mean_exp = np.array([0.09996541, 0.11773928, 25.03746617])
        x_var_exp = np.array([67.56160112, 83.60570564, 60.72378638])
        
        # Compare to expected results.
        self.assertEqual(x.shape, (10000, 3))
        assert_almost_equal(x_mean, x_mean_exp, decimal=-1)
        assert_almost_equal(x_var, x_var_exp, decimal=-1)


class TestMackeyGlass(unittest.TestCase):
    """Test the Mackey-Glass functions."""
    
    def test_mackey_glass(self):
        """Test the benchmarks._mackey_glass function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a random initial condition and propagate one step.
        x0 = rng.normal(size=(171, 1))
        x0_h = benchmarks._mackey_glass(0.1, 0.2, 10, 170, x0, 172,
                                        0.1)
        
        # Calculate derivative.
        diff = (x0_h[171] - x0_h[170])/0.1
        
        # Calculate derivative expected from Mackey-Glass equation.
        diff_exp = 0.2*x0[0] / (1 + x0[0]**10) - 0.1*x0[170]
        
        # Compare to expected results.
        assert_equal(x0_h[:171], x0)
        assert_almost_equal(diff, diff_exp)
        
    def test_get_mackey_glass(self):
        """Test the benchmarks.get_mackey_glass function."""
        
        # Grab a solution to Mackey-Glass equation with fixed seed.  
        x = benchmarks.get_mackey_glass(return_length=10000, seed=SEED)
        
        # Calculate statistical properties.
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        
        # Record expected statistical properties.
        x_mean_exp = np.array([0.93261555])
        x_var_exp = np.array([0.04973853])
        
        # Compare to expected results.
        self.assertEqual(x.shape, (10000, 1))
        assert_almost_equal(x_mean, x_mean_exp)
        assert_almost_equal(x_var, x_var_exp)
        

class TestDuffing(unittest.TestCase):
    """Test the Duffing functions."""
    
    def test_duffing(self):
        """Test the benchmarks._duffing function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a random initial condition and propagate one step.        
        x0 = rng.normal(size=(2))
        x0_h = benchmarks._duffing(0.3, -1.0, 1.0, 0.55, 0.2, x0,
                                   np.pi/4, 2, 0.1)
        
        # Calculate derivative.
        diff = (x0_h[1] - x0_h[0])/0.1
        
        # Calculate derivative expected from Duffing equation.
        diff_exp = np.zeros((2))
        diff_exp[0] = x0[1]
        diff_exp[1] = 0.55*np.cos(0.2*np.pi/4) - 0.3*x0[1] \
                          + 1.0*x0[0] - 1.0*x0[0]**3
        
        # Compare to expected results.
        assert_equal(x0_h[0], x0)
        assert_almost_equal(diff, diff_exp)
        
    def test_get_duffing(self):
        """Test the benchmarks.get_duffing function."""
        
        # Grab a solution to Duffing equation with fixed seed.  
        x = benchmarks.get_duffing(return_length=10000, seed=SEED)
        
        # Calculate statistical properties.
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        
        # Record expected statistical properties.
        x_mean_exp = np.array([-0.03621573, 0.00830427])
        x_var_exp = np.array([0.80494678, 0.34095772])
        
        # Compare to expected results.
        self.assertEqual(x.shape, (10000, 2))
        assert_almost_equal(x_mean, x_mean_exp)
        assert_almost_equal(x_var, x_var_exp)
        

class TestVanDerPol(unittest.TestCase):
    """Test the Van der Pol functions."""
    
    def test_van_der_pol(self):
        """Test the benchmarks._van_der_pol function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a random initial condition and propagate one step.
        x0 = rng.normal(size=(2))
        x0_h = benchmarks._van_der_pol(10, x0, 2, 0.1)
        
        # Calculate derivative.
        diff = (x0_h[1] - x0_h[0])/0.1
        
        # Calculate derivative expected from Van der Pol equation.
        diff_exp = np.zeros((2))
        diff_exp[0] = x0[1]
        diff_exp[1] = 10*(1 - (x0[0]**2)) * x0[1] - x0[0]
        
        # Compare to expected results.
        assert_equal(x0_h[0], x0)
        assert_almost_equal(diff, diff_exp)
        
    def test_get_van_der_pol(self):
        """Test the benchmarks.get_van_der_pol function."""
        
        # Grab a solution to Duffing equation with fixed seed. 
        x = benchmarks.get_van_der_pol(return_length=10000, seed=SEED)
        
        # Calculate statistical properties.
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        
        # Record expected statistical properties.
        x_mean_exp = np.array([-0.08532811, 0.03886185])
        x_var_exp = np.array([2.82956963, 2.40681179])
        
        # Compare to expected results.
        self.assertEqual(x.shape, (10000, 2))
        assert_almost_equal(x_mean, x_mean_exp)
        assert_almost_equal(x_var, x_var_exp)
        
                
if __name__ == '__main__':
    unittest.main()