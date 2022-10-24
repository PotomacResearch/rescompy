import unittest
import logging
import numba
import os
import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal as assert_equal
from numpy.testing import assert_array_almost_equal as assert_almost_equal
from scipy import sparse as sparse
from scipy.optimize import minimize
import rescompy


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


class TestStandardizer(unittest.TestCase):
    """Tests features of the rescompy.Standardizer class."""
    
    def test_repr(self):
        """Test that the custom __repr__ function works."""
        
        # Create standardizer from dummy signal.
        u = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.5
        standardizer = rescompy.Standardizer(u)
        
        # Confirm the string representation starts with 'Standardizer.'
        self.assertEqual(str(standardizer)[1:13], 'Standardizer')
    
    def test_use_case_one(self):
        """Test for a common use-case: standardizing a 1D array to 0 mean and
        unit variance."""
        
        # Create standardizer from dummy signal.
        u = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.5        
        standardizer = rescompy.Standardizer(u, shift_factor='mean')

        # Standardize and un-standardize dummy signal.
        v = standardizer.standardize(u)
        u_p = standardizer.unstandardize(v)
        
        # Calculate statistics.
        v_mean = np.mean(v, axis=0)
        v_var = np.var(v, axis=0)

        # Compare to expected values.
        assert_almost_equal(v_mean, 0)
        assert_almost_equal(v_var, 1)
        assert_almost_equal(u, u_p)
        
    def test_use_case_two(self):
        """Test for a common use-case: standardizing a 2D array to 0 mean and
        unit maximum value."""
        
        # Create standardizer from dummy 2D signal.
        u0 = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.5
        u1 = np.sin(np.linspace(0, 4*np.pi, 100)) - 0.25
        u = np.vstack((u0, u1)).T        
        standardizer = rescompy.Standardizer(u, shift_factor='mean',
                                             scale_factor='max')
        
        # Standardize and un-standardize dummy signal.
        v = standardizer.standardize(u)
        u_p = standardizer.unstandardize(v)
        
        # Calculate statistics.
        v_mean = np.mean(v, axis=0)
        v_max = np.max(v, axis=0)
        
        # Compare to expected values.
        assert_almost_equal(v_mean, [0, 0])
        assert_almost_equal(v_max, [1, 1])
        assert_almost_equal(u, u_p)
    
    def test_use_case_three(self):
        """Test a more complicated use-case: standardizing a 3D array to be
        non-negative with unit range, applying the same transformation to every
        dimension."""
        
        # Create dummy (11, 11)-D signal.
        u = np.zeros((100, 11, 11))
        x = np.linspace(-1, 1, 11, True)
        xx, yy = np.meshgrid(x, x)
        for i in range(100):
            u[i] = np.exp((xx + 0.002*i)**2 + (yy + 0.003*i)**2)
            
        # Define custom shift and scale functions.
        def custom_shift(x, axis):
            return np.min(x, axis)
        def custom_scale(x, axis):
            return np.max(x, axis) - np.min(x, axis)
        
        # Create standardizer from dummy signal.
        standardizer = rescompy.Standardizer(u, axis=(0, 1, 2),
                                             shift_factor=custom_shift,
                                             scale_factor=custom_scale)
        
        # Standardize and un-standardize signal.
        v = standardizer.standardize(u)
        u_p = standardizer.unstandardize(v)
        
        # Calculate statistics.
        v_min = np.min(v)
        v_range = np.max(v) - np.min(v)
        
        # Compare to expected values.
        assert_almost_equal(v_min, 0)
        assert_almost_equal(v_range, 1)
        assert_almost_equal(u, u_p)


class TestTrainResult(unittest.TestCase):
    """Tests features of the rescompy.TrainResult class."""

    def test_repr(self):
        """Test that the custom __repr__ function works."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create TrainResult from dummy arguments.
        states = rng.uniform(-1, 1, (100, 10))
        inputs = rng.uniform(-1, 1, (100, 3))
        target_outputs = rng.uniform(-1, 1, (100, 3))
        feature_function = lambda r, u: r
        weights = rng.uniform(-1, 1, (10, 3))
        transient_length = 20
        train_result = rescompy.TrainResult(states, inputs,
                                            target_outputs,
                                            feature_function, weights,
                                            transient_length)
        
        # Confirm the string representation starts with 'TrainResult.'
        self.assertEqual(str(train_result)[1:12], 'TrainResult')

    def test_properties(self):
        """Test that the various properties work as expected."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create TrainResult from dummy arguments.
        states = rng.uniform(-1, 1, (100, 10))
        inputs = rng.uniform(-1, 1, (100, 3))
        target_outputs = rng.uniform(-1, 1, (100, 3))
        feature_function = lambda r, u: r
        weights = rng.uniform(-1, 1, (10, 3))
        transient_length = 20        
        train_result = rescompy.TrainResult(states, inputs,
                                            target_outputs,
                                            feature_function,
                                            weights,
                                            transient_length)
        
        # Confirm that various derived attributes were correctly calculated.
        features = train_result.features
        assert_equal(features, states)
        
        reservoir_outputs = train_result.reservoir_outputs
        assert_equal(reservoir_outputs, np.dot(states, weights))
        

    
    def test_errors(self):
        """Test that the various error properties work as expected."""
        
        # Create TrainResult from dummy arguments.
        # Rather than being random, these are based on a simple linear problem
        # so that expected errors are easily calculated.
        states = np.linspace(0, 1, 100, False)[:, None]
        states = states.repeat(10, 1)
        inputs = np.zeros((100, 1))
        target_outputs = np.linspace(0, 1.1, 100, False)[:, None]
        feature_function = lambda r, u: r
        weights = np.zeros((10, 1)) + 0.1
        transient_length = 20        
        train_result = rescompy.TrainResult(states, inputs,
                                            target_outputs,
                                            feature_function,
                                            weights,
                                            transient_length)
        
        # Calculate the expected errors.
        expected_rmse = np.linspace(0, 0.1, 100, False)
        expected_nrmse = expected_rmse/np.sqrt(np.var(target_outputs))
        
        # Compare to expected values.
        assert_almost_equal(train_result.rmse, expected_rmse)
        assert_almost_equal(train_result.nrmse, expected_nrmse)


class TestPredictResult(unittest.TestCase):
    """Tests features of the rescompy.PredictResult class."""

    def test_repr(self):
        """Test that the custom __repr__ function works."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create PredictResult from dummy arguments.
        inputs = rng.uniform(-1, 1, (100, 3))
        reservoir_states = rng.uniform(-1, 1, (100, 50))
        reservoir_outputs = rng.uniform(-1, 1, (100, 3))
        target_outputs = rng.uniform(-1, 1, (100, 3))
        resync_inputs = rng.uniform(-1, 1, (20, 3))
        resync_states = rng.uniform(-1, 1, (20, 10))


        predict_result = rescompy.PredictResult(inputs, 
            reservoir_outputs, reservoir_states, 
            target_outputs,resync_inputs, resync_states)
        
        # Confirm the string representation starts with 'PredictResult.'
        self.assertEqual(str(predict_result)[1:14], 'PredictResult')

    def test_errors(self):
        """Test that the various error properties work as expected."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create PredictResult from dummy arguments.
        # Rather than being random, some of these are based on a simple linear
        # problem so that expected errors are easily calculated.
        inputs = rng.uniform(-1, 1, (100, 1))
        reservoir_outputs = np.linspace(0, 1, 100, False)[:, None]
        target_outputs = np.linspace(0, 1.1, 100, False)[:, None]
        resync_inputs = rng.uniform(-1, 1, (20, 3))
        resync_states = rng.uniform(-1, 1, (20, 10))
        reservoir_states = rng.uniform(-1, 1, (100, 50))

        predict_result = rescompy.PredictResult(inputs, 
            reservoir_outputs,reservoir_states, 
            target_outputs,resync_inputs, resync_states)
        
        # Calculate the expected errors.
        expected_rmse = np.linspace(0, 0.1, 100, False)
        expected_nrmse = expected_rmse/np.sqrt(np.var(target_outputs))
        
        # Compare to expected values.
        assert_almost_equal(predict_result.rmse, expected_rmse)
        assert_almost_equal(predict_result.nrmse, expected_nrmse)
        
        # Compare to expected valid lengths.
        self.assertEqual(predict_result.unit_valid_length, 100)
        self.assertEqual(predict_result.valid_length(0.01, True), 4)
        self.assertEqual(predict_result.valid_length(0.01, False), 10)
        

class TestESN(unittest.TestCase):
    """Tests features of the rescompy.ESN class."""
    
    def test_init(self):
        """Tests that the ESN is created properly with specified
        hyperparameters."""
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Calculate expected fixed attributes.
        input_dimension = esn.B.shape[1]
        sizes = [esn.B.shape[0], esn.A.shape[0], esn.A.shape[1],
                 esn.C.shape[0]]
        connections = esn.A.getnnz()/100
        A = esn.A.toarray()
        
        # Calculate the attributes derived from random matrices.
        spectral_radius = np.max(np.abs(np.linalg.eig(A)[0]))
        input_strength = np.max(np.abs(esn.B))
        bias_strength = np.max(np.abs(esn.C))
        leaking_rate = esn.leaking_rate

        # Compare to expected values.        
        self.assertEqual(input_dimension, 3)
        for size in sizes:
            self.assertEqual(size, 100)
        self.assertAlmostEqual(connections, 10, delta=1e-3)
        self.assertAlmostEqual(spectral_radius, 0.99, delta=1e-3)
        self.assertAlmostEqual(input_strength, 1.0, delta=1e-3)
        self.assertAlmostEqual(bias_strength, 0.5, delta=1e-3)
        self.assertEqual(leaking_rate, 0.25)
        
    def test_getters_and_setters(self):
        """Test the getter and setter functions for various attributes."""
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)

        # Multiply all of the matrices by a different scaling factor.
        esn.A *= 2
        esn.B *= 3
        esn.C *= 4

        # Confirm that the spectral_radius, input_strength, and bias_strength
        # attributes have changed accordingly.
        self.assertAlmostEqual(esn.spectral_radius, 2*0.99, delta=1e-2)
        self.assertAlmostEqual(esn.input_strength, 3*1.0, delta=1e-2)
        self.assertAlmostEqual(esn.bias_strength, 4*0.5, delta=1e-2)        
        
    def test_repr(self):
        """Test that the custom __repr__ function works."""
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Confirm the string representation starts with 'Echo State Network.'
        self.assertEqual(str(esn)[:18], 'Echo State Network')
        
    def test_equal(self):
        """Test that the custom __eq__ function works."""
        
        # Create three dummy ESNs with two different seeds.
        esn1 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        esn2 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        esn3 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED+1)

        # Confirm that ESNs with the same seed are equal and with different
        # seeds are unequal.
        self.assertTrue(esn1 == esn2)
        self.assertFalse(esn1 == esn3)

    def test_save_and_load(self):
        """Test that an ESN can be saved and loaded correctly."""
        
        # Choose a location for saving the test ESN.
        name = '_rescompy_tests_TestESN_test_save_and_load_esn.dat'
        
        # Create and save a dummy ESN.
        esn_s = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        esn_s.save(name)
        
        # Load the ESN back.
        esn_l = rescompy.ESN.load(name)
        
        # Confirm that the loaded ESN matches the saved one.
        self.assertEqual(esn_s, esn_l)
        
        # Delete the saved ESN on disk.
        os.remove(name)
        
    def test_train(self):
        """Test the training method."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Create a batch of dummy inputs and target outputs.
        inputs = [rng.uniform(-1, 1, (100, 3)) for _ in range(5)]
        target_outputs = [rng.uniform(-1, 1, (100, 3)) for _ in range(5)]
        
        # Obtain TrainResult through train method.
        train_result = esn.train(10, inputs, target_outputs)
        
        # Confirm that the inputs and target_outputs were properly batched.
        for i in range(5):
            assert_equal(train_result.inputs[i*100:(i+1)*100],
                         inputs[i])
            assert_equal(train_result.target_outputs[i*100:(i+1)*100],
                         target_outputs[i])
            
        # Confirm that, even though data were random, the RMSE is reduced
        # after the transient for every batch.
        for i in range(5):
            self.assertTrue(np.mean(train_result.rmse[i*100:i*100+10]) >
                            np.mean(train_result.rmse[i*100+10:(i+1)*100]))
        
    def test_predict(self):
        """Test the prediction method."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Create a batch of dummy inputs and target outputs.
        inputs = [rng.uniform(-1, 1, (100, 3)) for _ in range(5)]
        target_outputs = [rng.uniform(-1, 1, (100, 3)) for _ in range(5)]
        
        # Obtain TrainResult through train method.
        train_result = esn.train(10, inputs, target_outputs)
        
        # Obtain corresponding predict method after closing the loop.
        _ = esn.predict(train_result, 10)

    def test_predict_no_train_result(self):
        """Test the prediction by providing the weights and initial state
        directly."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Create dummy inputs and target outputs.
        inputs = rng.uniform(-1, 1, (100, 3))
        target_outputs = rng.uniform(-1, 1, (100, 3))
        
        # Obtain TrainResult through train method.
        train_result = esn.train(10, inputs, target_outputs)
        
        # Create first predict result by providing the train_result directly.
        predict_result1 = esn.predict(train_result, 10)
        
        # Compare to a second predict result where weights and initial states
        # are provided directly.
        predict_result2 = esn.predict(train_result.weights, 10, 
                              initial_state=train_result.states[-1],
                              feature_function=lambda r, u: r)
        assert_almost_equal(predict_result1.reservoir_outputs,
                            predict_result2.reservoir_outputs)
        

class TestOptimizeHyperparameters(unittest.TestCase):
    """Tests features of the rescompy.optimize_hyperparameters function."""
    
    def test(self):
        """Test the function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy ESN with a fixed seed.
        esn = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Create dummy inputs and target outputs for training and testing.
        inputs_train = rng.uniform(-1, 1, (100, 3))
        target_outputs_train = rng.uniform(-1, 1, (100, 3))
        inputs_test = rng.uniform(-1, 1, (10, 3))
        target_outputs_test = rng.uniform(-1, 1, (10, 3))
        
        # Collect training and testing arguments in dictionaries.
        train_args = {'inputs': inputs_train,
                      'target_outputs': target_outputs_train,
                      'transient_length': 10}
        predict_args = {'inputs': inputs_test,
                        'target_outputs': target_outputs_test}
        
        # Define an fast-executing optimizer by limiting the number of function
        # evaluations.
        def optimizer(func, x0):
            result = minimize(func, x0, method='Nelder-Mead',
                              options={'maxiter': 10})
            return result.x
        
        # Optimize the hyperparameters.
        esn_opt = rescompy.optimize_hyperparameters(esn, train_args,
                                                    predict_args,
                                                    optimizer=optimizer,
                                                    verbose=False)
        
        # Confirm that the hyperparameters for the optimzied ESN have changed.
        self.assertNotEqual(esn.spectral_radius,
                            esn_opt.spectral_radius)
        self.assertNotEqual(esn.input_strength, esn_opt.input_strength)
        self.assertNotEqual(esn.bias_strength, esn_opt.bias_strength)


class TestCopy(unittest.TestCase):
    """Tests features of the rescompy.copy function."""
    
    def test_seed(self):
        """Test the normal copy operation, with an ESN seed but without
        requesting a new seed."""
        
        # Create a dummy ESN with a fixed seed.
        esn1 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Copy the ESN without a new seed.
        esn2 = rescompy.copy(esn1)
        
        # Confirm that the copy equals the original.
        self.assertTrue(esn1 == esn2)
        
        # Confirm that altering the copy does not alter the original.
        esn2.leaking_rate *= 2
        self.assertFalse(esn1 == esn2)
        
    def test_no_seed(self):
        """Test the normal copy operation, without an ESN seed or a requested
        new seed."""
        
        # Create a dummy ESN with a fixed seed.
        esn1 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Remove the seed.
        esn1.seed = None
        
        # Copy the ESN without a new seed.
        esn2 = rescompy.copy(esn1)
        
        # Confirm that the copy equals the original.
        self.assertTrue(esn1 == esn2)
        
        # Confirm that altering the copy does not alter the original.
        esn2.leaking_rate *= 2
        self.assertFalse(esn1 == esn2)
        
    def test_new_seed(self):
        """Test the copy operation with a new seed."""
        
        # Create a dummy ESN with a fixed seed.
        esn1 = rescompy.ESN(3, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Copy the ESN, but provide a new seed.
        esn2 = rescompy.copy(esn1, SEED+1)
        
        # Confirm that the fixed attributes of the copy match the original.
        self.assertEqual(esn1.input_dimension, esn2.input_dimension)
        self.assertEqual(esn1.size, esn2.size)
        self.assertEqual(esn1.connections, esn2.connections)
        self.assertEqual(esn1.leaking_rate, esn2.leaking_rate)
        
        # Confirm that attributes derived from random matrices are
        # approximately the same.
        self.assertAlmostEqual(esn1.spectral_radius,
                               esn2.spectral_radius, delta=1e-3)
        self.assertAlmostEqual(esn1.input_strength,
                               esn2.input_strength, delta=1e-1)
        self.assertAlmostEqual(esn1.bias_strength,
                               esn2.bias_strength, delta=1e-1)
        
        # Confirm that the copy does not exactly match the original.
        self.assertFalse(esn1 == esn2)

    
class TestMultVec(unittest.TestCase):
    """Tests features of the rescompy._mult_vec function."""
    
    def test(self):
        """Test the function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy sparse matrix and a dummy dense matrix to be
        # multiplied.
        A = sparse.csr_matrix(sparse.random(10, 10, 0.1,
                                            random_state=rng))
        B = rng.uniform(size=(10))
        
        # Execute the multiplication.
        X = rescompy._mult_vec(A.data, A.indices, A.indptr, A.shape, B)
        
        # Compare to expected results, calculated from non-compiled
        # multiplication.
        expected_X = np.dot(A.toarray().T, B)
        assert_almost_equal(X, expected_X)


class TestGetStatesDriven(unittest.TestCase):
    """Tests features of the rescompy._get_states_driven function."""
    
    def test(self):
        """Test the function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy sparse matrix and several dummy dense matrices that
        # describe the ESN dynamics.
        A = sparse.csr_matrix(sparse.random(10, 10, 0.1,
                                            random_state=rng))
        B = rng.uniform(size=(10, 3))
        C = rng.uniform(size=(10))
        u = rng.uniform(low=-1, high=1, size=(101, 3))
        r = rng.uniform(low=-1, high=1, size=(101, 10))
        
        # Propagate the ESN dynamics to obtain the states.
        X = rescompy._get_states_driven(u, r, A.data, A.indices,
                                     A.indptr, A.shape, B, C, 0.25)
        
        # Calculate expected states from the non-compiled equations.
        A = A.toarray()
        for i in range(100):
            r[i+1] = (1.0-0.25)*r[i] + 0.25*np.tanh(np.dot(B, u[i])
                + np.dot(A, r[i]) + C)
        
        # Compare to expected results.
        assert_almost_equal(X, r[1:])


class TestGetStatesAutonomousJit(unittest.TestCase):
    """Tests features of the rescompy._get_states_autonomous_jit function."""

    def test(self):
        """Test the function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy sparse matrix and several dummy dense matrices that
        # describe the ESN dynamics.
        A = sparse.csr_matrix(sparse.random(10, 10, 0.1,
                                            random_state=rng))
        B = rng.uniform(size=(10, 3))
        C = rng.uniform(size=(10))
        W = rng.uniform(size=(10, 3))
        u = rng.uniform(low=-1, high=1, size=(101, 3))
        v = rng.uniform(low=-1, high=1, size=(101, 3))
        r = rng.uniform(low=-1, high=1, size=(101, 10))
        
        # Define and compile feature and mapper functions.
        @numba.jit(nopython=True)
        def feature_function(r, u):
            return r
        @numba.jit(nopython=True)
        def mapper(inputs, outputs):
            return outputs
        
        # Propagate the ESN dynamics to obtain states and outputs.
        X, Y = rescompy._get_states_autonomous_jit(u, v, r,
               feature_function, mapper, A.data, A.indices, A.indptr,
               A.shape, B, C, W, 0.25)
        X = X
        Y = Y
        
        # Calculate expected states and outputs from the non-compiled
        # equations.
        A = A.toarray()
        for i in range(100):
            r[i+1] = (1.0-0.25)*r[i] + 0.25*np.tanh(np.dot(B, v[i])
                + np.dot(A, r[i]) + C)
            v[i+1] = np.dot(r[i+1], W)
            
        # Compare to expected results.
        assert_almost_equal(X, r[1:])
        assert_almost_equal(Y, v[1:])
        

class TestGetStatesAutonomous(unittest.TestCase):
    """Tests features of the rescompy._get_states_autonomous function."""

    def test(self):
        """Test the function."""
        
        # Create the random state for reproducibility.
        rng = default_rng(SEED)
        
        # Create a dummy sparse matrix and several dummy dense matrices that
        # describe the ESN dynamics.
        A = sparse.csr_matrix(sparse.random(10, 10, 0.1,
                                            random_state=rng))
        B = rng.uniform(size=(10, 3))
        C = rng.uniform(size=(10))
        W = rng.uniform(size=(10, 3))
        u = rng.uniform(low=-1, high=1, size=(101, 3))
        v = rng.uniform(low=-1, high=1, size=(101, 3))
        r = rng.uniform(low=-1, high=1, size=(101, 10))
        
        # Define non-compiled feature and mapper functions.
        def feature_function(r, u):
            return r
        def mapper(inputs, outputs):
            return outputs
        
        # Propagate the ESN dynamics to obtain states and outputs.
        X, Y = rescompy._get_states_autonomous(u, v, r,
               feature_function, mapper, A.data, A.indices, A.indptr,
               A.shape, B, C, W, 0.25)
        X = X
        Y = Y
        
        # Calculate expected states and outputs from the non-compiled
        # equations.
        A = A.toarray()
        for i in range(100):
            r[i+1] = (1.0-0.25)*r[i] + 0.25*np.tanh(np.dot(B, v[i])
                + np.dot(A, r[i]) + C)
            v[i+1] = np.dot(r[i+1], W)
            
        # Compare to expected results.
        assert_almost_equal(X, r[1:])
        assert_almost_equal(Y, v[1:])


if __name__ == '__main__':
    unittest.main()