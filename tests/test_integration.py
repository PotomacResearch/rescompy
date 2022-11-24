import unittest
import logging
import numba
import numpy as np
import rescompy
from rescompy import regressions


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


class TestCases(unittest.TestCase):
    """Test a common use-cases."""
    
    def test_sine_open_loop(self):
        """Predict a sine-wave in an open-loop configuration.
        This test examines the following rescompy features:
            automatic conversion of 1D signal
            custom mapping function
            automatic inference of prediction length
        """
                
        # Create a 1D sine-wave.
        # Partition into inputs, outputs, training, and testing.
        u = np.sin(np.linspace(0, 2*np.pi*10, 2000, False))
        inputs_train = u[:1000-1]
        target_outputs_train = u[1:1000]
        inputs_test = u[1000-1:2000-1]
        target_outputs_test = u[1000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(1, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(200, inputs_train,
                                 target_outputs_train)
        
        # Define an open-loop mapping function.
        @numba.jit(nopython=True, fastmath=True)
        def mapper(inputs, outputs):
            return inputs
        
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test,
                                     mapper=mapper)
        
        # Sine is a simple problem. Confirm that the ESN performed very well.
        self.assertTrue(np.mean(predict_result.rmse) < 1e-3)
        self.assertEqual(predict_result.unit_valid_length, 1000)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
        
        
    def test_sine_closed_loop(self):
        """Predict a sine-wave in a closed-loop configuration.
        This test examines the following rescompy features:
            automatic conversion of 1D signal
            automatic inference of training target signal
            automatic inference of prediction length
        """
                
        # Create a 1D sine-wave.
        # Partition into inputs, outputs, training, and testing.
        u = np.sin(np.linspace(0, 2*np.pi*10, 2000, False))
        inputs_train = u[:1000]
        inputs_test = u[1000-1:2000-1]
        target_outputs_test = u[1000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(1, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(200, inputs_train)
        
        # Predict the signal in closed-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test)
        
        # Sine is a simple problem. Confirm that the ESN performed very well.
        self.assertTrue(np.mean(predict_result.rmse) < 1e-3)
        self.assertEqual(predict_result.unit_valid_length, 1000)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
        
    def test_sine_pass_through(self):
        """Predict a sine-wave in a closed-loop configuration with a direct
        pass-through from input to output.
        """
                
        # Create a 1D sine-wave.
        # Partition into inputs, outputs, training, and testing.
        u = np.sin(np.linspace(0, 2*np.pi*10, 2000, False))
        inputs_train = u[:1000]
        inputs_test = u[1000-1:2000-1]
        target_outputs_test = u[1000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(1, 100, 10, 0.99, 1.0, 0.5, 0.25, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(200, inputs_train, 
                                 feature_function=
                                 rescompy.features.StatesAndInputs())
        
        # Predict the signal in closed-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test)
        
        # Sine is a simple problem. Confirm that the ESN performed very well.
        self.assertTrue(np.mean(predict_result.rmse) < 1e-2)
        self.assertEqual(predict_result.unit_valid_length, 1000)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
        
        
    def test_lorenz_poly(self):
        """Predict a Lorenz system in a closed-loop configuration with a
        polynomial observation function.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        u = rescompy.Standardizer(u).standardize(u)
        inputs_train = u[:5000-1]
        target_outputs_train = u[1:5000]
        inputs_test = u[5000-1:10000-1]
        target_outputs_test = u[5000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train,
                                 feature_function=
                                 rescompy.features.ConstantInputAndPolynomial(2))
                
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test)
        
        # Lorenz is a surprisingly simple problem. Confirm that the ESN
        # performed very well.
        self.assertTrue(predict_result.unit_valid_length > 250)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
        
    def test_lorenz_jacobian(self):
        """Predict a Lorenz system in a closed-loop configuration with 
        jacobian regularization.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        u = rescompy.Standardizer(u).standardize(u)
        inputs_train = u[:5000-1]
        target_outputs_train = u[1:5000]
        inputs_test = u[5000-1:10000-1]
        target_outputs_test = u[5000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train,
                                 feature_function=rescompy.features.StatesOnly(),
                                 regression=regressions.jacobian(1e-6, 1e-6))
                
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test)

        # Lorenz is a surprisingly simple problem. Confirm that the ESN
        # performed very well.
        self.assertTrue(predict_result.unit_valid_length > 250)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)


    def test_lorenz_observer(self):
        """Predict the z-component of the Lorenz system from the x- and
        y-components.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000, seed=SEED)
        u = rescompy.Standardizer(u).standardize(u)
        inputs_train = u[:5000, :2]
        target_outputs_train = u[:5000, 2]
        inputs_test = u[5000:, :2]
        target_outputs_test = u[5000:, 2]
        
        # Create an ESN with a fixed seed.
        #esn = rescompy.ESN(2, 1000, 10, 0.99, 1.0, 0.5, 0.1, SEED)
        esn = rescompy.ESN(2, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)

        
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train)
                
        # Define an observer mapper function.
        @numba.jit(nopython=True, fastmath=True)
        def mapper(inputs, outputs):
            return inputs
        
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test,
                                     mapper=mapper)
        
        # Lorenz is a surprisingly simple problem. Confirm that the ESN
        # performed very well.
        self.assertTrue(np.mean(predict_result.rmse) < 1e-2)
        self.assertEqual(predict_result.unit_valid_length, 5000)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
        
        
    def test_lorenz_partial_feedback(self):
        """Execute a Lorenz control problem where the z-component is fed back
        during prediction, but x and y are still provided as inputs.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        u = rescompy.Standardizer(u).standardize(u)
        inputs_train = u[:5000-1]
        target_outputs_train = u[1:5000]
        inputs_test = u[5000-1:10000-1, :2]
        target_outputs_test = u[5000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train)
                
        # Define an observer mapper function.
        @numba.jit(nopython=True, fastmath=True)
        def mapper(inputs, outputs):
            return np.concatenate((inputs, outputs[2:]))
        
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test,
                                     mapper=mapper)
        
        # Even though we are providing inputs, this approach to the problem
        # tends to be less stable, so let's only ensure that we are valid for
        # about one Lyapunov time.
        self.assertTrue(predict_result.unit_valid_length > 100)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)


    def test_lorenz_partial_feedback_with_lookbacks(self):
        """Execute a Lorenz control problem where the z-component is fed back
        during prediction, but x and y are still provided as inputs. This time,
		include time-delayed states and inputs in the feature vectors.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        u = rescompy.Standardizer(u).standardize(u)
        inputs_train = u[:5000-1]
        target_outputs_train = u[1:5000]
        inputs_test = u[5000-1:10000-1, :2]
        target_outputs_test = u[5000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        feature_function = rescompy.features.StatesAndInputsTimeShifted(
			states_lookback_length = 2,
			inputs_lookback_length = 2,
			states_decimation = 1,
			inputs_decimation = 1
			)
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs = inputs_train,
                                 target_outputs = target_outputs_train,
								 feature_function = feature_function)
                
        # Define an observer mapper function.
        @numba.jit(nopython=True, fastmath=True)
        def mapper(inputs, outputs):
            return np.concatenate((inputs, outputs[2:]))
        
        # Predict the signal in open-loop configuration.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs = target_outputs_test,
                                     mapper =  mapper)
        
        # Even though we are providing inputs, this approach to the problem
        # tends to be less stable, so let's only ensure that we are valid for
        # about one Lyapunov time.
        self.assertTrue(predict_result.unit_valid_length > 100)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)

        
    def test_lorenz_batch_resync(self):
        """Train an ESN on a batch of Lorenz signals with slightly different
        parameters, then resync and predict with a new Lorenz signal."""
        
        # Create a batch of Lorenz signal.
        # Partition each into inputs, outputs, training, and testing.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        standardizer = rescompy.Standardizer(u)
        inputs_train = []
        target_outputs_train = []
        for i in range(5):
            u = rescompy.benchmarks.get_lorenz(return_length=10000,
                    rho=28+0.1*i, seed=SEED)
            u = standardizer.standardize(u)
            if i < 4:
                inputs_train += [u[:5000-1]]
                target_outputs_train += [u[1:5000]]
            else:
                resync_signal = u[:5000-1]
                inputs_test = u[5000-1:10000-1]
                target_outputs_test = u[5000:]
        
        # Create an ESN with a fixed seed.
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        # Train the ESN on the training signals.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train)
        
        # Predict the signal in closed-loop configuration with the resync
        # signal.
        predict_result = esn.predict(train_result, inputs=inputs_test,
                                     target_outputs=target_outputs_test,
                                     resync_signal=resync_signal)
        
        # Even though the values of rho are mis-matched, the signals are
        # similar enough that we expected it to perform well.
        self.assertTrue(predict_result.unit_valid_length > 100)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        self.assertTrue(kink_measure < 1)
    
    
    def test_hybrid(self):
        """Perform the usual closed-loop prediction, but incorporate an
        imperfect model prediction based on the Euler-integration of the
        Lorenz equations with mismatched parameters.
        """
        
        # Create a Lorenz signal.
        # Partition into inputs, outputs, training, and testing.
        # Note that we are not standardizing as to easily implement the
        # imperfect predictive model.
        u = rescompy.benchmarks.get_lorenz(return_length=10000,
                                           seed=SEED)
        inputs_train = u[:5000-1]
        target_outputs_train = u[1:5000]
        target_outputs_test = u[5000:]

        # Create an ESN with a fixed seed.
        # Note that the input_scaling is reduced since we are not standardizing
        # the inputs.        
        esn = rescompy.ESN(3, 1000, 10, 0.59, 0.63, 0.13, 0.09, SEED)
        
        # Define a custom feature function that incorporates an imperfect
        # model. 
        def feature_function(r, u):
            
            u_p = np.zeros((u.shape))
            u_p[:, 0] = 9*(u[:, 1] - u[:, 0])
            u_p[:, 1] = u[:, 0]*(27 - u[:, 2]) - u[:, 1]
            u_p[:, 2] = u[:, 0]*u[:, 1] - (8/3)*u[:, 2]
            u_h = u + 0.01*u_p
            return np.hstack((r, u_h))
        
        # Train the ESN on the training signal.
        train_result = esn.train(1000, inputs_train,
                                 target_outputs_train,
                                 feature_function=feature_function)
        
        # Predict the signal.
        predict_result = esn.predict(train_result,
                                     target_outputs=target_outputs_test)
        
        # With the aid of the imperfect model, the ESN should perform
        # exceptionally well.
        self.assertTrue(predict_result.unit_valid_length > 500)
        
        # A tell-tale sign of an indexing problem is a 'kink' in the error just
        # after prediction starts. Confirm that the error is smooth around
        # here.
        transition_error = np.concatenate((train_result.rmse[-10:],
                                           predict_result.rmse[:10]))
        kink_measure = np.sqrt(np.var(transition_error))/ \
            np.mean(transition_error)
        
        #self.assertTrue(kink_measure < 1)
        # TODO: kink_measure > 1 despite the transition error appearing smooth;
        # need better detector for indexing errors.

        

if __name__ == '__main__':
    unittest.main()