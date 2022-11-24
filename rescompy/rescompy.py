"""rescompy.py

The main module for rescompy.

Contains basic function and class definitions for manipulation, training, and
forecasting with reservoir computers.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


# Import dependencies.
import pickle
import logging
import numba
import warnings
import numpy as np
import inspect
import scipy.stats
from typing import Optional, Union, Callable, Tuple, Literal, List
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.optimize import minimize
from numpy.random import default_rng
from numpy.random._generator import Generator
from numba.core.errors import NumbaPerformanceWarning
from tabulate import tabulate
from pydantic import validate_arguments
from .utils import utils
from .features import features
from .regressions import regressions


# Ignore unhelpful numba performance warnings.
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@numba.jit(nopython=True, fastmath=True)
def default_mapper(inputs, outputs):
    return outputs

class Standardizer:
    """The Standardizer class.
    
    This class is configured from a single time series and a few options. 

    Attributes:
        shape (Tuple[int, ...]): The expected shape of the time series that can
            be standardized / unstandardized.
            A value of None indicates that that axis can have any length.
        shift (np.ndarray): The factor or factors that are added to u as part
            of the standardizing transformation.
        scale (np.ndarray): The factor or factors that multiply u as part of
            the standardizing transformation.
    """

    AXES = Literal['auto']
    SHIFT_MODES = Literal['mean']
    SCALE_MODES = Literal['var', 'max']
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        u:            np.ndarray,
        axis:         Union[AXES, int, Tuple[int, ...]] = 'auto',
        shift_factor: Union[SHIFT_MODES, Callable]      = 'mean',
        scale_factor: Union[SCALE_MODES, Callable]      = 'var',
        ):
        """The initialization method for the standardizer class.
        
        Configures the Standardizer from a single time series or set of time
        series.

        Args:
            u (np.ndarray): The time series used to configure the standardizing
                transformation.
            axis ('auto', int, Tuple[int, ...]): The axis or axes along which
                to standardize the time series.
                Usually, this is interpreted as the time axis.
                If 'auto', will use the longest axis of u.
            shift_factor ('mean', Callable): The shifting factor.
                If 'mean', will shift u to zero mean.
                If a callable, must accept 'axis' as a keyword argument.
                If a callable, u will be shfited by subtracting
                shift_factor(u, axis=axis).
            scale_factor: The scaling factor.
                If 'var', will scale u to unit variance.
                If 'max', will scale u to unit maximum absolute value.
                If a callable, must accept 'axis' as a keyword argument.
                If a callable, will be scaled by dividing by
                scale_factor(u, axis=axis).
        """
        
        # If axis is set to 'auto', select None or longest axis.
        if axis == 'auto':
            if len(u.shape) == 1:
                axis = None
            else:
                axis = (np.argmax(u.shape),)
        
        # If axis is an integer, convert to a tuple.
        if isinstance(axis, int):
            axis = (axis,)
            
        # Calculate and store expected shape of acceptable signals.
        # Acceptable signals must have the same shape as u except along the 
        # above axis.
        if axis is None:
            self.shape = (None,)*len(u.shape)
        else:
            shape = list(u.shape)
            for axis_i in axis:
                shape[axis_i] = None
            self.shape = tuple(shape)
        
        # If shift_factor is set to 'mean', implement the mean function.
        if shift_factor == 'mean':
            shift_factor = lambda x, axis: np.mean(x, axis)
                
        # If scale_factor is set to 'var' or 'max', implement the respective
        # functions.
        if scale_factor == 'var':
            scale_factor = lambda x, axis: np.sqrt(np.var(x, axis))
        elif scale_factor == 'max':
            scale_factor = lambda x, axis: np.max(np.abs(x), axis)
              
        # Calculate and store the parameters of the standardizing transformation.
        self.shift = -shift_factor(u, axis=axis)
        self.scale = 1/scale_factor(u + self.shift, axis=axis)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def standardize(
        self,
        u: np.ndarray,
        ) -> np.ndarray:
        """The standardizing method for the Standardizer class.
        
        Transforms an input array according to the configured transformation
        and returns the result.
        
        Args:
            u (np.ndarray): The time series to be standardized.
                Must have the same shape as the u used to configure the
                standardizer, except along the time axis.

        Returns:
            result (np.ndarray): The transformed array.
        """

        # Check that the shape of u is compatible.
        utils.check_shape(u.shape, self.shape, 'u')

        # Transform u.            
        result = u + self.shift
        result *= self.scale
        
        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def unstandardize(
        self,
        u: np.ndarray,
        ) -> np.ndarray:
        """The unstandardizing method for the Standardizer class.
        
        Undoes the standardizing transformation and returns the result.
        
        Args:
            u (np.ndarray): The time series to be unstandardized.
                Must have the same shape as the u used to configure the
                standardizer, except along the time axis.

        Returns:
            result (np.ndarray): The transformed array.
        """
        
        # Check that the shape of u is compatible.
        utils.check_shape(u.shape, self.shape, 'u')
            
        # Transform u.
        result = u/self.scale
        result += -self.shift
        
        return result

    def __repr__(self):
        msg = f"<Standardizer for signals with shape {self.shape}>"
        return msg	


class TrainResult:
    """The Training Result class.
    
    Stores arrays relevant to the ESN training process and computes some basic
    properties of interest.
    
    Attributes:
        states (np.ndarray): The set of reservoir states over the training
            period, including the initial transient.
        inputs (np.ndarray): The reservoir inputs.
            The first axis must have the same length as the first axis of
            states.
        target_outputs (np.ndarray): The target outputs.
            The first axis must have the same length as the first axis of
            states.
        feature_function (Callable): The function that transforms the states
            and inputs to features.
        weights (np.ndarray): The array of output weights.
        transient_lengths (List[int]): The lengths of the initial transients.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        states:            List[np.ndarray],
        inputs:            List[np.ndarray],
        target_outputs:    Union[List[None],  List[np.ndarray]],
        transient_lengths: List[int],
        accessible_drives: List[int],
        feature_function:  Callable,
        weights:           np.ndarray,
        jacobian:          Optional[np.ndarray] = None
        ):
        """The initialization method of the TrainResult class.
        
        Instantiates the object and verifies the correctness of certain
        arguments.
        
        Args:
            states (list): Each entry contains the set of reservoir states over   
			    the training period, including the initial transient, for the 
                corresponding training task.
            inputs (list): Each entry contains reservoir inputs over   
			    the training period, including the initial transient, for the 
                corresponding training task.
                The first axis must have the same length as the first axis of
                states.
            target_outputs (list): The list of target_outputs stored during the
			    training stage.
			transient_lengths (list): Each entry contains the length of the 
                initial transient for the corresponding training task.
			accessible_drives (list): The list of training signal IDs that have
			    been stored during training. The entries of drive_results and 
				target_outputs corresponding to input signals that are not 
				identified in accessible_drives will be None.
            feature_function (Callable): The function that transforms the
                states and inputs to features.
            weights (np.ndarray): The array of output weights.
        """
		
        #Check for shape consistencies.
        for task_ind in range(len(inputs)):
            features = feature_function(states[task_ind], inputs[task_ind])
            utils.check_shape(inputs[task_ind].shape,
							  (states[task_ind].shape[0], None), 'inputs')
            utils.check_shape(weights.shape,
							  (features.shape[1], target_outputs[0].shape[1]),
							  'weights')
            if (target_outputs[task_ind].shape[0] > 1):
                utils.check_shape(target_outputs[task_ind].shape,
								  (states[task_ind].shape[0], None), 'target_outputs')

            else:
                utils.check_range(features.shape[0],
								  "The number of feature vectors per input signal",
								  1, 'eq', True)
        
        # Assign attributes.
        self.feature_function = feature_function
        self.accessible_drives = accessible_drives
        self.weights = weights
        self.jacobian = jacobian
        
		# Check if the feature function requires a lookback period to include
        # time-delayed states or inputs in feature vectors.
        if hasattr(feature_function, 'lookback_length'):
            self.lookback_length = feature_function.lookback_length
        else: self.lookback_length = 0

        self.listed_targets = target_outputs		
        self.listed_transients = transient_lengths
        self.listed_inputs = inputs
        self.listed_states = states
		
        self.states = None
        for task_ind in range(len(self.listed_inputs)):
            if (self.states is None):
                self.states = self.listed_states[task_ind]
                self.inputs = self.listed_inputs[task_ind]
                self.target_outputs = self.listed_targets[task_ind]
            else:
                self.states = np.concatenate((self.states,
											  self.listed_states[task_ind]))
                self.inputs = np.concatenate((self.inputs,
											  self.listed_inputs[task_ind]))
                self.target_outputs = np.concatenate((self.target_outputs,
													  self.listed_targets[task_ind]))
						
    @property
    def transient_length(self):
        """The transient length associated with the TrainResult object.
        If the same transient length applied to all training signals, this 
        this transient length is returned.
        If multiple transient lengths applied, an error will be raised."""
        
        transients = np.array(self.listed_transients)
        if transients == transients[0]:
            return transients[0]
        else:
            msg = "Transient length is not defined since different training " \
                      "tasks employed different transient lengths."
            logging.error(msg)
        
    @property
    def features(self):
        """The features property.
        Computes the reservoir features from the states and inputs using the
        feature_function."""
        features = None
        for task_ind in range(len(self.listed_inputs)):
            if (features is None):
                features = self.feature_function(self.listed_states[task_ind],
												 self.listed_inputs[task_ind])
            else:
                features = np.concatenate((features,
					self.feature_function(self.listed_states[task_ind],
					self.listed_inputs[task_ind])), axis = 0)
        return features
    
    @property
    def reservoir_outputs(self):
        """The reservoir outputs property.
        Computes the reservoir outputs from the features and weights."""
        return np.dot(self.features, self.weights)
    
    @property
    def rmse(self):
        """The root-mean-square error property.
        Computes the root-mean-square error as a function of time."""
        return np.sqrt(np.mean(
			np.square(self.reservoir_outputs -
			 self.target_outputs[self.lookback_length:]), axis=1))
    
    @property
    def nrmse(self):
        """The normalized root-mean-square-error property.
        Computes the normalized root-mean-square-error as a function of time.
        Note that this is done component-wise.
        If the variance of any component is 0, the NRMSE is not defined and an
        error will be raised."""
        
        norm = np.var(self.target_outputs, axis=0)
        if np.min(norm) == 0.0:
            msg = "NRMSE is not defined when a component of the " \
                      "target output has 0 variance."
            logging.error(msg)
        else:
            se = np.square(self.reservoir_outputs
                           - self.target_outputs[self.lookback_length:])
            return np.sqrt(np.mean(se/norm, axis=1))

    def __repr__(self):
        msg = f"<TrainResult with ({self.weights.shape[0]} x " \
                  f"{self.weights.shape[1]}) weights>"
        return msg


class PredictResult:
    """The Prediction Result class.
    
    Stores arrays relevant to the ESN prediction process and computes some
    basic properties of interest.
    
    Attributes:
        inputs (np.ndarray): The reservoir inputs during the testing period.
        reservoir_outputs (np.ndarray): The reservoir outputs.
            The first axis must have the same length as the first axis of
            inputs.
        reservoir_states (np.ndarray): The reservoir states over the prediction
            period.
        predict_length (int): The length of the prediction period. If zero,
		    a classification-style task with a single output 
			vector is assumed.
        target_outputs (np.ndarray): The target outputs.
            The first axis must have the same length as the first axis of
            inputs.
        resync_inputs (np.ndarray): The inputs used to re-synchronize the
            reservoir state.
        resync_states (np.ndarray): The reservoir states subject to the
            resynchronization signal.
            The first axis must have the same length as the first axis of
            resync_inputs.
        resync_outputs (np.ndarray): The reservoir outputs over the
            resynchronization period.
    """
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        inputs:            np.ndarray,
        reservoir_outputs: np.ndarray,
        reservoir_states:  np.ndarray,
        predict_length:    int,
        target_outputs:    Optional[np.ndarray] = None,
        resync_inputs:     Optional[np.ndarray] = None,
        resync_states:     Optional[np.ndarray] = None,
        resync_outputs:    Optional[np.ndarray] = None   
        ):
        """The initialization method of the PredictResult class.
        
        Instantiates the object and verifies the correctness of certain
        arguments.
        
        Args:
            inputs (np.ndarray): The reservoir inputs during the testing
                period.
            reservoir_outputs (np.ndarray): The reservoir outputs.
                The first axis must have the same length as the first axis of
                inputs.
            reservoir_states (np.ndarray): The reservoir states over the prediction
                period.
			predict_length (int): The length of the prediction period. If zero,
			    a classification-style task with a single output 
				vector is assumed.
            target_outputs (np.ndarray): The target outputs.
                The first axis must have the same length as the first axis of
                inputs.
            resync_inputs (np.ndarray): The inputs used to re-synchronize the
                reservoir state.
            resync_states (np.ndarray): The reservoir states subject to the
                resynchronization signal.
                The first axis must have the same length as the first axis of
                resync_inputs.
            resync_outputs (np.ndarray): The reservoir outputs over the
                resynchronization period.
        """
        
        # Check for shape consistencies.
        if (predict_length < 0):
            msg = "predict_length must be greater than or equal to zero."
            logging.error(msg)
        
        elif (predict_length == 0):
            utils.check_shape(reservoir_outputs.shape,
                          (1, None), 'reservoir_outputs')
            if target_outputs is not None:
                utils.check_shape(target_outputs.shape,
                              (1, None),
                              'target_outputs')
        
        else: 
            utils.check_shape(reservoir_outputs.shape,
                          (inputs.shape[0], None), 'reservoir_outputs')
            if target_outputs is not None:
                utils.check_shape(target_outputs.shape,
                              (inputs.shape[0], None),
                              'target_outputs') 
        
        if resync_inputs is not None and resync_states is not None:
            utils.check_shape(resync_states.shape, 
                              (resync_inputs.shape[0], None),
                              'resync_states')
        
        # Assign attributes.
        self.inputs = inputs
        self.reservoir_outputs = reservoir_outputs
        self.reservoir_states = reservoir_states
        self.target_outputs = target_outputs
        self.resync_inputs = resync_inputs
        self.resync_states = resync_states
        self.resync_outputs = resync_outputs
        self.predict_length = predict_length

    @property
    def component_errors(self) -> np.ndarray:
        """The component-wise error property.
        Computes the component-wise error for a single-target mapping.
        Is only defined when target_outputs are provided and
		predict_length = 0."""
        
        if self.predict_length != 0:
            msg = "component_errors are not defined for single-target " \
				"time-series prediction tasks. Call rmse or nrmse instead."
            logging.error(msg)      
        elif self.target_outputs is not None:
            return np.sqrt(np.mean(np.square(self.reservoir_outputs
                                             - self.target_outputs), axis=1))
        else:
            msg = "component_errors are not defined without target outputs."
            logging.error(msg)

    @property
    def rmse(self) -> np.ndarray:
        """The root-mean-square error property.
        Computes the root-mean-square error as a function of time for 
		time-series prediction tasks with predict_length greater than zero.
		Computes the root-mean-square error over all components of target 
		vector for single-target mappings with predict_length zero.
        Is only defined when target_outputs are provided."""
        
        if self.target_outputs is not None:
            if self.predict_length == 0:
	            return np.sqrt(np.mean(np.square(self.reservoir_outputs
	                                             - self.target_outputs))) 
            else:
                return np.sqrt(np.mean(np.square(self.reservoir_outputs
                                             - self.target_outputs), axis=1))
        
        else:
            msg = "RMSE is not defined without target outputs."
            logging.error(msg)
            
    @property
    def nrmse(self):
        """The normalized root-mean-square-error property.
        Computes the normalized root-mean-square-error as a function of time.
        Note that this is done component-wise.
        If the variance of any component is 0, the NRMSE is not defined and an
        error will be raised.
        Is only defined when target_outputs are provided."""
        
        if self.predict_length == 0:
            msg = "NRMSE is not defined for single-target mappings with " \
				"predict_length zero. Call component_wise_error or error instead."
            logging.error(msg)
        
        elif self.target_outputs is not None:
            
            norm = np.var(self.target_outputs, axis=0)
            if np.min(norm) == 0.0:
                msg = "NRMSE is not defined when a component of the " \
                          "target output has 0 variance."
                logging.error(msg)
            else:
                se = np.square(self.reservoir_outputs
                               - self.target_outputs)
                return np.sqrt(np.mean(se/norm, axis=1))
            
        else:
            msg = "NRMSE is not defined without target outputs."
            logging.error(msg)

    @property
    def unit_valid_length(self) -> int:
        """The unit valid length property.
        Computes the length of time before the NRMSE first exceeds 1.0."""
        if(self.predict_length > 0): return self.valid_length(1.0, True)
        else:
            msg = "unit_valid_length is not defined for single-target mapping" \
				" tasks with predict_length equal to zero"
            logging.error(msg)
            
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def valid_length(
        self,
        threshold:  float = 1.0,
        normalized: bool  = True
        ) -> int:
        """The valid length method for the PredictResult class.
        
        Computes the length of time before the (possibly normalized) RMSE
        exceeds a specified threshold.
        
        Args:
            threshold (float): The threshold for the error.
            normalized (bool): If true, will consider the NRMSE; otherwise,
                will consider the RMSE.
                
        Returns:
            valid_length (int): The computed valid length.
        """
        if(self.predict_length > 0):
            # Grab either the NRMSE or the RMSE.
            if normalized:
                errors = self.nrmse
            else:
                errors = self.rmse

            # Calculate the valid length based on the errors and the threshold.
            valid_length = 0
            for error in errors:
                if error <= threshold:
                    valid_length += 1
                else:
                    break

            # If the valid length is equal to the length of target_outputs, the
            # actual valid_length might be greater than calculated; warn the user
            # accordingly.              
            if valid_length == self.target_outputs.shape[0]:
                if normalized:
                    msg = "Normalized root-square-error does not exceed " \
                      f" {threshold}; true valid_length may be " \
                      "greater than reported."
                else:
                    msg = "Root-square-error does not exceed " \
                      f" {threshold}; true valid_length may be " \
                      "greater than reported."
                logging.warning(msg)
				
            return valid_length

        else:
            msg = "valid_length is not defined for single-target mapping tasks" \
				" with predict_length equal to zero"
            logging.error(msg)
                
            
    def __repr__(self):
        msg = "<PredictResult>"
        return msg


class ESN:
    """The Echo-State Network class.

    The main class of the rescompy module.
    
    This class represents an echo-state network instance and contains methods
    for analyzing, training, predicting, etc.
    
    Attributes:
        input_dimension (int): The dimension of the inputs.
        size (int): The size of (number of nodes in) the network.
        connections (float): The mean in-degree of the adjacency matrix.
            Must be greater than 0 and less than or equal to N. 
        leaking_rate (float): The leaking rate of the nodes.
            It is usually greater than 0 and less than or equal to 1.0.
        seed (int, Generator): An integer or Generator object for determining
            the random seed.
            This is for reproducibility in the random matrices A, B, and C.
        A (sparse.csr.csr_matrix): The adjacency matrix.
        B (np.ndarray): The input matrix.
        C (np.ndarray): The bias vector.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_dimension: int,
        size:            int                      = 100,
        connections:     float                    = 10.0,
        spectral_radius: float                    = 0.99,
        input_strength:  float                    = 1.0,
        bias_strength:   float                    = 0.0,
        leaking_rate:    float                    = 1.0,
        #seed:           Optional[int, Generator] = None,
        seed: Union[int, None, Generator] = None,
        ):
        """The Initialization method for a ESN object.

        This method initializes the ESN object by performing use-friendly type
        and value checks on the standard ESN hyperparameters, then creating
        random matrices describing the reservoir dynamics.

        Args:
            input_dimension (int): The dimension of the inputs.
            size (int): The size of (number of nodes in) the network.
            connections (float): The mean in-degree of the adjacency matrix.
                Must be greater than 0 and less than or equal to N. 
            spectral_radius (float): The spectral radius of the adjacency
                matrix.
                Must be greater than 0 and is usually on the order of 1.0.
            input_strength (float): The input coupling strength.
                Must be greater than 0 and is usually on the order of 1.0 when
                the input is standardized.
            bias_strength (float): The bias strength.
                Must be greater than or equal to 0 and is usually less than
                1.0.
            leaking_rate (float): The leaking rate of the nodes.
                It is usually greater than 0 and less than or equal to 1.0.
            seed (int, Generator): An integer or Generator object for
                determining the random seed.
                This is for reproducibility in the random matrices A, B, and C.
        """
        
        # Check that arguments are in the correct range.
        utils.check_range(connections, 'connections', 0.0, 'geq', True)
        utils.check_range(connections, 'connections', size, 'leq', True)
        utils.check_range(spectral_radius, 'spectral_radius', 0.0, 'g', True)
        utils.check_range(spectral_radius, 'spectral_radius', 1.0, 'l', False)
        utils.check_range(input_strength, 'input_strength', 0.0, 'g', True)  
        utils.check_range(bias_strength, 'bias_strength', 0.0, 'geq', True)  
        
        # Create the random state for reproducibility.
        rng = default_rng(seed)

        # Create the adjacency matrix A.
        def rvs(size):
            return rng.uniform(low=-1, high=1, size=size)
        # TODO: This needs a better fix.
        done = False
        while not done:
            try:
                A = sparse.random(size, size, density=connections/size,
                                  random_state=rng, data_rvs=rvs)
                v0 = rng.random(size)
                eigenvalues, _ = splinalg.eigs(A, k=1, v0=v0)
                A *= spectral_radius/np.abs(eigenvalues[0])
                done = True
            except:
                print ("failed to calculate SR, trying again...")

        # Create input matrix B.
        if isinstance(input_strength, float):
            B = rng.uniform(low=-input_strength, high=input_strength,
                            size=(size, input_dimension))
        else:
            B = rng.uniform(low=-1, high=1, size=(size,
                                                  input_dimension))
            for strength_ind in range(len(input_strength)):
                B[:, strength_ind] *= input_strength[strength_ind]

        # Create the bias vector C.
        C = rng.uniform(low=-bias_strength, high=bias_strength,
                        size=size)

        # Assign all of the required class attributes.
        self.size = size
        self.input_dimension = input_dimension
        self.connections = connections
        self._spectral_radius = spectral_radius
        self._input_strength = input_strength
        self._bias_strength = bias_strength
        self.leaking_rate = leaking_rate
        self.seed = seed
        self.A = sparse.csr_matrix(A)
        self.B = B
        self.C = C
        
    @property
    def spectral_radius(self):
        rng = default_rng(self.seed)
        v0 = rng.random(self.size)
        eigenvalues, _ = splinalg.eigs(self.A, k=1, v0=v0)
        return np.abs(eigenvalues[0])
    
    @spectral_radius.setter
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def spectral_radius(self, value: float):
        utils.check_range(value, 'spectral_radius', 0.0, 'g', True)
        utils.check_range(value, 'spectral_radius', 1.0, 'l', False)
        self.A *= value/self._spectral_radius
        
    @spectral_radius.deleter
    def spectral_radius(self):
        raise AttributeError("Cannot delete spectral_radius.")
        
    @property
    def input_strength(self):
        return np.max(np.abs(self.B))
    
    @input_strength.setter
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def input_strength(self, value: float):
        utils.check_range(value, 'input_strength', 0.0, 'g', True)
        self.B *= value/self._input_strength
        
    @input_strength.deleter
    def input_strength(self):
        raise AttributeError("Cannot delete input_strength.")
        
    @property
    def bias_strength(self):
        return np.max(np.abs(self.C))
    
    @bias_strength.setter
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def bias_strength(self, value: float):
        utils.check_range(value, 'bias_strength', 0.0, 'geq', True)
        self.C *= value/self._bias_strength
        
    @bias_strength.deleter
    def bias_strength(self):
        raise AttributeError("Cannot delete bias_strength.")

    def __repr__(self) -> str:
        header = 'Echo State Network with the following properties:\n'
        if isinstance(self.input_strength, float):
            table = [['size', self.size],
                     ['input_dimension', self.input_dimension],
                     ['connections', self.connections],
                     ['spectral_radius', self.spectral_radius],
                     ['input_strength', self.input_strength],
                     ['bias_strength', self.bias_strength],
                     ['leaking_rate', self.leaking_rate]]
        else:
            table = [['size', self.size],
                     ['input_dimension', self.input_dimension],
                     ['connections', self.connections],
                     ['spectral_radius', self.spectral_radius],
                     ['input_strength', '(multiple)'],
                     ['bias_strength', self.bias_strength],
                     ['leaking_rate', self.leaking_rate]]
        return header + str(tabulate(table))

    def __eq__(self, other) -> bool:
        values_self = self.__dict__
        values_other = other.__dict__
        equal = 1
        for value in values_self.keys():
            if value in values_other:
                if isinstance(values_self[value], np.ndarray):
                    try:
                        equal *= np.allclose(values_self[value],
                                             values_other[value])
                    except:
                        return False
                elif isinstance(values_self[value],
                                scipy.sparse.csr_matrix):
                    try:
                        equal *= np.allclose(values_self[
                                             value].toarray(),
                                             values_other[
                                             value].toarray())
                    except:
                        return False
                else:
                    equal *= values_self[value] == values_other[value]
            else:
                return False
        return bool(equal)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def save(
        self,
        file_name: str
        ):
        """The ESN saving method.
        
        This method uses a simple pickle funtion to save the object for later.
        
        Args:
            file_name (str): The name of the file location.
        """
        
        with open(file_name, 'wb') as temp_file:
            pickle.dump(self, temp_file)
    
    @staticmethod
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def load(
        file_name: str
        ):
        """The ESN loading method.
        
        This method uses a simple pickle function to load the object from disk.
        
        Args:
            file_name (str): The name of the file location.
            
        Returns:
            result (ESN): The loaded ESN object.
        """

        with open(file_name, 'rb') as temp_file:
            result = pickle.load(temp_file)
            
        return result

    def get_states(
        self, 
        initial_state:      np.ndarray, 
        inputs:             np.ndarray,
        ):
        
        states = initial_state[None].repeat(inputs.shape[0] + 1, axis=0)

        # Propagate the reservoir state with the input signal.
        return _get_states_driven(inputs, states,
                                         self.A.data, self.A.indices,
                                         self.A.indptr, self.A.shape,
                                         self.B, self.C,
                                         self.leaking_rate)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def train_saving_memory(
        self,
        transient_length:  Union[int, List[int]],
        inputs:            Union[np.ndarray, List[np.ndarray]],
        target_outputs:    Union[np.ndarray, List[np.ndarray], None] = None,
        initial_state:     Optional[np.ndarray]                      = None,
        feature_function:  Union[Callable, features.Feature]         = features.StatesOnly(),
        regression:        Optional[Callable]                        = regressions.batched_ridge(),
        batch_length:      int                                       = 100,
        ) -> TrainResult:
        """The training method.
                
        Args:
            transient_length: The length of the initial transient to be
                              discarded for each sample input signal.
							  If a single int is provided, it will be the 
							  transient length for all sample inputs.
            inputs: An array of ESN inputs, or a list of such arrays.
                    For each array the first dimension is the number of 
                    samples, and the second dimension must be equal to 
					self.input_dimension.
            target_outputs: An array of desired ESN outputs, or a list of such
							arrays.
                            The first dimension of each array is the number of 
							samples and must be equal to the first dimension of
							inputs.
                            If not provided, it is assumed that we are trying
                            to predict the inputs.
			initial_state: The initial reservoir state at the start of each
						   training task.
            feature_function: The function that forms the feature vectors from
                              the reservoir states.
            regression: The optimizer that minimizes the difference between
                        features and target_outputs. Must take as its arguments
						the matrices VS_T and SS_T where V is the matrix of 
						targets and S the matrix of features.
            batch_length (int): The number of time steps to process
                        before each update to the regression matrices.
                                   
        Returns:
            result: A TrainResult object containing information about the
                    training process.
        """
        
        # For downstream clarity, collect inputs and target_outputs in a list,
        # if not already.
        if not isinstance(transient_length, list):
            transient_length = [transient_length]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(target_outputs, list):
            target_outputs = [target_outputs]
            
            # If target_outputs was not provided, assume None for every input.
            if target_outputs[0] is None:
                target_outputs *= len(inputs)
		             
        # Confirm that the same number of inputs and target_outputs are
        # provided.
        if len(inputs) != len(target_outputs):
            msg = "len(target_outputs) must be None or match len(inputs)."
            logging.error(msg)
			
        # Confirm that the same number of inputs and transients are
        # provided.
        if len(transient_length) == 1: transient_length = transient_length * len(inputs)
        elif len(transient_length) != len(inputs): 
            msg = "len(transient_lengths) must be one or match len(inputs)."
            logging.error(msg)
        
        # Shape inputs and target_outputs, if necessary.
        for inputs_ind in range(len(inputs)):
            if len(inputs[inputs_ind].shape) == 1:
                inputs[inputs_ind] = inputs[inputs_ind][:, None]
        for target_outputs_ind in range(len(target_outputs)):
            if target_outputs[target_outputs_ind] is not None:
                if len(target_outputs[target_outputs_ind].shape) == 1:
                    target_outputs[target_outputs_ind] = \
                        target_outputs[target_outputs_ind][:, None]
        
        # Check if the feature function requires a lookback period to include
        # time-delayed states or inputs in feature vectors.
        if hasattr(feature_function, 'lookback_length'):
            lookback_length = feature_function.lookback_length
        else: lookback_length = 0

        # If no targets are provided, assume we are trying to predict the
        # inputs.
        for task_ind in range(len(inputs)):
            if target_outputs[task_ind] is None:
                target_outputs[task_ind] = np.copy(inputs[task_ind])[1:]
                inputs[task_ind] = np.copy(inputs[task_ind])[:-1]
                    
            # Check that arguments are in the correct range.
            utils.check_range(transient_length[task_ind], 'transient_length',
							  0, 'geq', True)
            utils.check_range(transient_length[task_ind], 'transient_length',
	                          inputs[task_ind].shape[0], 'l', True)
        
        # If no initial state is provided, start at all zeros.
        if initial_state is None:
            initial_state = np.zeros((self.size))
		
        # Check for the appropriate shapes.
        for task_ind in range(len(inputs)):
            if len(inputs) == 1: msg1 = 'inputs'
            else: msg1 = f'inputs[{task_ind}]'
            utils.check_shape(inputs[task_ind].shape,
                          (None, self.input_dimension), msg1)
        		
        # Get the propagated reservoir states.
        VS_T = None
        SS_T = None
        for task_ind in range(len(inputs)):
            batch_size = min(batch_length,
							 inputs[task_ind].shape[0] - transient_length[task_ind])
            num_batches = (inputs[task_ind].shape[0] -
						    transient_length[task_ind]) // batch_size
            remainder_length = (inputs[task_ind].shape[0] -
						    transient_length[task_ind]) % batch_size
            
            transient_i = transient_length[task_ind]
            inputs_i = inputs[task_ind][:transient_i+batch_length]
            states_i = self.get_states(initial_state, inputs_i)
            new_state = states_i[-1]
            lookback_inputs = inputs_i[-lookback_length-1:]
            lookback_states = states_i[-lookback_length-1:]
            
            features_i = feature_function(states_i[transient_i:],
										  inputs_i[transient_i:])
            targets_i = target_outputs[task_ind][transient_i + lookback_length:
									   transient_i + batch_size]
			
            if (SS_T is None):
                SS_T = features_i.T @ features_i
                VS_T = features_i.T @ targets_i
            else:
                SS_T += features_i.T @ features_i
                VS_T += features_i.T @ targets_i
            
            for batch_ind in range(1, num_batches):
                start_time = transient_i + batch_ind * batch_size
                inputs_i = inputs[task_ind][start_time: start_time + batch_size]
                states_i = self.get_states(new_state, inputs_i)
                new_state = states_i[-1]
                states_i = np.concatenate((lookback_states, states_i[1:]))
                inputs_i = np.concatenate((lookback_inputs, inputs_i[1:]))
                lookback_inputs = inputs_i[-lookback_length-1:]
                lookback_states = states_i[-lookback_length-1:]
                
                targets_i = target_outputs[task_ind][start_time:
										   start_time + batch_size]
                features_i = feature_function(states_i, inputs_i)

                SS_T += features_i.T @ features_i
                VS_T += features_i.T @ targets_i
            
            if (num_batches > 0 and remainder_length):
                start_time = transient_i + num_batches * batch_size
                inputs_i = inputs[task_ind][start_time:]
                states_i = self.get_states(new_state, inputs_i)
                new_state = states_i[-1]
                states_i = np.concatenate((lookback_states, states_i[1:]))
                inputs_i = np.concatenate((lookback_inputs, inputs_i[1:]))
                lookback_inputs = inputs_i[-lookback_length-1:]
                lookback_states = states_i[-lookback_length-1:]
                
                targets_i = target_outputs[task_ind][start_time:]
                features_i = feature_function(states_i, inputs_i)

                SS_T += features_i.T @ features_i
                VS_T += features_i.T @ targets_i

        if (num_batches > 1 or (num_batches == 1 and remainder_length)):
            inputs_saved = [inputs_i[lookback_length:]]
            states_saved = [states_i[lookback_length:]]
        else:
            inputs_saved = inputs_i
            states_saved = states_i
        targets_saved = [targets_i]
        transients_saved = [transient_i]

        # use the signature of the regression to figure out the parameters
        params = inspect.signature(regression).parameters
        # assume u = input, s = feature, v = target, g is the feature function
        map = {
		  'SS_T': SS_T,
		  'VS_T': VS_T
        }
        # check that there are no unknown parameters
        if any([k not in map for k in params.keys()]):
            logging.error('Please ensure regression function signature is valid')
        # construct the parameter dict 
        args = {arg: map[arg] for arg in params.keys()}

        # Optimize the output weights      
        weights = regression(**args)
        
        # Construct and return the training result.
        return TrainResult(states_saved, inputs_saved, targets_saved,
						   transients_saved, [-1],
						   feature_function, weights)


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def train_batched(
        self,
        transient_lengths: Union[int, List[int]],
        inputs:            Union[np.ndarray, List[np.ndarray]],
        target_outputs:    Union[np.ndarray, List[np.ndarray], None] = None,
        initial_state:     Optional[np.ndarray]                      = None,
        dg_du:             Optional[np.ndarray]                      = None,
        feature_function:  Union[Callable, features.Feature]         = features.StatesOnly(),
        regression:        Optional[Callable]                        = regressions.batched_ridge(),
        batch_size:        int                                       = 10,
        accessible_drives: Union[int, List[int], str]                = "all",
        ) -> TrainResult:
        """The training method.
                
        Args:
            transient_lengths: The length of the initial transient to be
                              discarded for each sample input signal.
							  If a single int is provided, it will be the 
							  transient length for all sample inputs.
            inputs: An array of ESN inputs, or a list of such arrays.
                    For each array the first dimension is the number of 
                    samples, and the second dimension must be equal to 
					self.input_dimension.
            target_outputs: An array of desired ESN outputs, or a list of such
							arrays.
                            The first dimension of each array is the number of 
							samples and must be equal to the first dimension of
							inputs.
                            If not provided, it is assumed that we are trying
                            to predict the inputs.
			initial_state: The initial reservoir state at the start of each
						   training task.
            feature_function: The function that forms the feature vectors from
                              the reservoir states.
            regression: The optimizer that minimizes the difference between
                        features and target_outputs. Must take as its arguments
						the matrices VS_T and SS_T where V is the matrix of 
						targets and S the matrix of features.
            batch_size (int): The number of sample ESN inputs to process
                        before each update to the regression matrices.
            accessible_drives (str, int, list): The training signals whose
                        associated states and inputs objects will be available
                        int the returned TrainResult object. Defaults to all.
                                   
        Returns:
            result: A TrainResult object containing information about the
                    training process.
        """
        
        # For downstream clarity, collect inputs and target_outputs in a list,
        # if not already.
        if not isinstance(transient_lengths, list):
            transient_lengths = [transient_lengths]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(target_outputs, list):
            target_outputs = [target_outputs]
            
            # If target_outputs was not provided, assume None for every input.
            if target_outputs[0] is None:
                target_outputs *= len(inputs)
		
        if isinstance(accessible_drives, int):
            accessible_drives = [accessible_drives]
        elif accessible_drives == "final":
            accessible_drives = [len(inputs) - 1]
        elif accessible_drives == "all":
            accessible_drives = list(np.arange(len(inputs)))
            
        # Confirm that the same number of inputs and target_outputs are
        # provided.
        if len(inputs) != len(target_outputs):
            msg = "len(target_outputs) must be None or match len(inputs)."
            logging.error(msg)
			
        # Confirm that the same number of inputs and transients are
        # provided.
        if len(transient_lengths) == 1: transient_lengths = transient_lengths * len(inputs)
        elif len(transient_lengths) != len(inputs): 
            msg = "len(transient_lengths) must be one or match len(inputs)."
            logging.error(msg)
        
        # Shape inputs and target_outputs, if necessary.
        for inputs_ind in range(len(inputs)):
            if len(inputs[inputs_ind].shape) == 1:
                inputs[inputs_ind] = inputs[inputs_ind][:, None]
        for target_outputs_ind in range(len(target_outputs)):
            if target_outputs[target_outputs_ind] is not None:
                if len(target_outputs[target_outputs_ind].shape) == 1:
                    target_outputs[target_outputs_ind] = \
                        target_outputs[target_outputs_ind][:, None]
        
        # Check if the feature function requires a lookback period to include
        # time-delayed states or inputs in feature vectors.
        if hasattr(feature_function, 'lookback_length'):
            lookback_length = feature_function.lookback_length
        else: lookback_length = 0
        
        # If no targets are provided, assume we are trying to predict the
        # inputs.
        for task_ind in range(len(inputs)):
            if target_outputs[task_ind] is None:
                target_outputs[task_ind] = np.copy(inputs[task_ind])[1:]
                inputs[task_ind] = np.copy(inputs[task_ind])[:-1]
                    
            # Check that arguments are in the correct range.
            utils.check_range(transient_lengths[task_ind], 'transient_length',
							  0, 'geq', True)
            utils.check_range(transient_lengths[task_ind], 'transient_length',
	                          inputs[task_ind].shape[0], 'l', True)
        
        # If no initial state is provided, start at all zeros.
        if initial_state is None:
            initial_state = np.zeros((self.size))
		
        # Check for the appropriate shapes.
        for task_ind in range(len(inputs)):
            if len(inputs) == 1: msg1 = 'inputs'
            else: msg1 = f'inputs[{task_ind}]'
            utils.check_shape(inputs[task_ind].shape,
                          (None, self.input_dimension), msg1)
        
        num_samples = len(inputs)
        batch_size = min(num_samples, batch_size)
		
        num_batches = num_samples // batch_size
        num_remainders = num_samples % batch_size
		
        # Get the propagated reservoir states.
        # Separately, record the states post-transient for training.
        features = None
        targets_saved = [None] * len(accessible_drives)
        inputs_saved = [None] * len(accessible_drives)
        states_saved = [None] * len(accessible_drives)
        transients_saved = [None] * len(accessible_drives)
        save_ind = 0
        VS_T = None
        SS_T = None
        for batch_ind in range(num_batches):
            for task_ind in range(batch_size):
                total_ind = batch_ind * batch_size + task_ind
                transient_i = transient_lengths[total_ind]
                inputs_i = inputs[total_ind]
                states_i = self.get_states(initial_state, inputs_i)
                targets_i = target_outputs[total_ind]
                features_i = feature_function(states_i[transient_i:],
											  inputs_i[transient_i:])
                targets_train_i = targets_i[transient_i + lookback_length:]
                if total_ind in accessible_drives:
                    inputs_saved[save_ind] = inputs_i
                    targets_saved[save_ind] = targets_i
                    states_saved[save_ind] = states_i
                    transients_saved[save_ind] = transient_i
                    save_ind += 1
                if features is None:
                    features = features_i
                    targets_train = targets_train_i
                else:
                    features = np.concatenate((features, features_i))
                    targets_train = np.concatenate((targets_train,
													targets_train_i))
            if (SS_T is None):
                SS_T = features.T @ features
                VS_T = features.T @ targets_train
            else:
                SS_T += features.T @ features
                VS_T += features.T @ targets_train
            
            features = None

        if (num_remainders):
            for task_ind in range(num_remainders):
                total_ind = num_batches * batch_size + task_ind
                transient_i = transient_lengths[total_ind]
                inputs_i = inputs[total_ind]
                states_i = self.get_states(initial_state, inputs_i)
                targets_i = target_outputs[total_ind]
                features_i = feature_function(states_i[transient_i:],
													inputs_i[transient_i:])
                targets_train_i = targets_i[transient_i + lookback_length:]
                if total_ind in accessible_drives:
                    inputs_saved[save_ind] = inputs_i
                    targets_saved[save_ind] = targets_i
                    states_saved[save_ind] = states_i
                    transients_saved[save_ind] = transient_i
                    save_ind += 1
                if features is None:
                    features = features_i
                    targets_train = targets_train_i
                else:
                    features = np.concatenate((features, features_i))
                    targets_train = np.concatenate((targets_train,
													targets_train_i))
            if (SS_T is None):
                SS_T = features.T @ features
                VS_T = features.T @ targets_train
            else:
                SS_T += features.T @ features
                VS_T += features.T @ targets_train

        # use the signature of the regression to figure out the parameters
        params = inspect.signature(regression).parameters
        # assume u = input, s = feature, v = target, g is the feature function
        map = {
		  'SS_T': SS_T,
		  'VS_T': VS_T
        }
        # check that there are no unknown parameters
        if any([k not in map for k in params.keys()]):
            logging.error('Please ensure regression function signature is valid')
        # construct the parameter dict 
        args = {arg: map[arg] for arg in params.keys()}
              
        weights = regression(**args)
      
        # Construct and return the training result.
        return TrainResult(states_saved, inputs_saved, targets_saved,
				   transients_saved, accessible_drives,
				   feature_function, weights, jacobian=args.get('dg_du', None))

    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def train(
        self,
        transient_length:  Union[int, List[int]],
        inputs:            Union[np.ndarray, List[np.ndarray]],
        target_outputs:    Union[np.ndarray, List[np.ndarray], None] = None,
        initial_state:     Optional[np.ndarray]                      = None,
        dg_du:             Optional[np.ndarray]                      = None,
        feature_function:  Union[Callable, features.Feature]         = features.StatesOnly(),
        regression:        Optional[Callable]                        = regressions.tikhonov(),
        accessible_drives: Union[int, List[int], str]                = "all",
        ) -> TrainResult:
        """The training method.
                
        Args:
            transient_lengths: The length of the initial transient to be
                              discarded for each sample input signal.
							  If a single int is provided, it will be the 
							  transient length for all sample inputs.
            inputs: An array of ESN inputs, or a list of such arrays.
                    For each array the first dimension is the number of 
                    samples, and the second dimension must be equal to 
					self.input_dimension.
            target_outputs: An array of desired ESN outputs, or a list of such
							arrays.
                            The first dimension of each array is the number of 
							samples and must be equal to the first dimension of
							inputs.
                            If not provided, it is assumed that we are trying
                            to predict the inputs.
			initial_state: The initial reservoir state at the start of each
						   training task.
            feature_function: The function that forms the feature vectors from
                              the reservoir states.
            regression: The optimizer that minimizes the difference between
                        features and target_outputs.
			accessible_drives (str, int, list): The training signals whose
                        associated states and inputs objects will be available
                        int the returned TrainResult object. Defaults to all.
                                   
        Returns:
            result: A TrainResult object containing information about the
                    training process.
        """
        
        # For downstream clarity, collect inputs and target_outputs in a list,
        # if not already.
        if not isinstance(transient_length, list):
            transient_length = [transient_length]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(target_outputs, list):
            target_outputs = [target_outputs]
            
            # If target_outputs was not provided, assume None for every input.
            if target_outputs[0] is None:
                target_outputs *= len(inputs)

        if isinstance(accessible_drives, int):
            accessible_drives = [accessible_drives]
        elif accessible_drives == "final":
            accessible_drives = [len(inputs) - 1]
        elif accessible_drives == "all":
            accessible_drives = list(np.arange(len(inputs)))
    
        # Confirm that the same number of inputs and target_outputs are
        # provided.
        if len(inputs) != len(target_outputs):
            msg = "len(target_outputs) must be None or match len(inputs)."
            logging.error(msg)
			
        # Confirm that the same number of inputs and transients are
        # provided.
        if len(transient_length) == 1: transient_length = transient_length * len(inputs)
        elif len(transient_length) != len(inputs): 
            msg = "len(transient_lengths) must be one or match len(inputs)."
            logging.error(msg)
        
        # Shape inputs and target_outputs, if necessary.
        for inputs_ind in range(len(inputs)):
            if len(inputs[inputs_ind].shape) == 1:
                inputs[inputs_ind] = inputs[inputs_ind][:, None]
        for target_outputs_ind in range(len(target_outputs)):
            if target_outputs[target_outputs_ind] is not None:
                if len(target_outputs[target_outputs_ind].shape) == 1:
                    target_outputs[target_outputs_ind] = \
                        target_outputs[target_outputs_ind][:, None]
        
        # Check if the feature function requires a lookback period to include
        # time-delayed states or inputs in feature vectors.
        if hasattr(feature_function, 'lookback_length'):
            lookback_length = feature_function.lookback_length
        else: lookback_length = 0
        
        # If no targets are provided, assume we are trying to predict the
        # inputs.
        for task_ind in range(len(inputs)):
            if target_outputs[task_ind] is None:
                target_outputs[task_ind] = np.copy(inputs[task_ind])[1:]
                inputs[task_ind] = np.copy(inputs[task_ind])[:-1]
                    
            # Check that arguments are in the correct range.
            utils.check_range(transient_length[task_ind], 'transient_length',
							  0, 'geq', True)
            utils.check_range(transient_length[task_ind], 'transient_length',
	                          inputs[task_ind].shape[0], 'l', True)
        
        # If no initial state is provided, start at all zeros.
        if initial_state is None:
            initial_state = np.zeros((self.size))
		
        # Check for the appropriate shapes.
        for task_ind in range(len(inputs)):
            if len(inputs) == 1: msg1 = 'inputs'
            else: msg1 = f'inputs[{task_ind}]'
            utils.check_shape(inputs[task_ind].shape,
                          (None, self.input_dimension), msg1)            
        
        # Get the propagated reservoir states.
        # Separately, record the states post-transient for training.
        features = None
        targets_saved = [None] * len(accessible_drives)
        inputs_saved = [None] * len(accessible_drives)
        states_saved = [None] * len(accessible_drives)
        transients_saved = [None] * len(accessible_drives)
        save_ind = 0
        for task_ind in range(len(inputs)):
            transient_i = transient_length[task_ind]
            inputs_i = inputs[task_ind]
            states_i = self.get_states(initial_state, inputs_i)
            targets_i = target_outputs[task_ind]
            features_i = feature_function(states_i[transient_i:],
										  inputs_i[transient_i:])
            targets_train_i = targets_i[transient_i + lookback_length:]
            if task_ind in accessible_drives:
                inputs_saved[save_ind] = inputs_i
                targets_saved[save_ind] = targets_i
                states_saved[save_ind] = states_i
                transients_saved[save_ind] = transient_i
                save_ind += 1
            if features is None:
                states_train = states_i
                inputs_train = inputs_i
                features = features_i
                targets_train = targets_train_i
            else:
                states_train = np.concatenate((states_train, states_i))
                inputs_train = np.concatenate((inputs_train, inputs_i))
                features = np.concatenate((features, features_i))
                targets_train = np.concatenate((targets_train,
												targets_train_i))
        
        # use the signature of the regression to figure out the parameters
        params = inspect.signature(regression).parameters
        # assume u = input, s = feature, v = target, g is the feature function
        map = {
            'u': inputs_train,
            's': features,
            'v': targets_train,
            'g': feature_function,
            'dg_du': dg_du
        }
        # check that there are no unknown parameters
        if any([k not in map for k in params.keys()]):
            logging.error('Please ensure regression function signature is valid')
        # construct the parameter dict
        args = {arg: map[arg] for arg in params.keys()}
        # calculate dg_du if needed
        if 'dg_du' in args:
            if args['dg_du'] is None:
                if not hasattr(feature_function, 'jacobian'):
                    logging.error('Regression function requires feature jacobian')
                dr_du = _calc_dr_du(states_train, inputs_train, self.size, 
                    self.input_dimension, self.A, self.B, self.C, self.leaking_rate)
                args['dg_du'] = feature_function.jacobian(states_train, inputs_train, dr_du)
                
        weights = regression(**args)
        
        # Construct and return the training result.
        return TrainResult(states_saved, inputs_saved, targets_saved,
						   transients_saved, accessible_drives,
						   feature_function, weights, jacobian=args.get('dg_du', None))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))        
    def predict(
        self,
        train_result:     Union[TrainResult, np.ndarray],
        predict_length:   Optional[int]        = None,
        inputs:           Optional[np.ndarray] = None,
        target_outputs:   Optional[np.ndarray] = None,
        initial_state:    Optional[np.ndarray] = None,
        resync_signal:    Optional[np.ndarray] = None,
        mapper:           Optional[Callable]   = default_mapper,
        feature_function: Optional[Callable]   = None,
        lookback_states:  Optional[np.ndarray] = None,
		lookback_inputs:  Optional[np.ndarray] = None,
        ) -> PredictResult:
        """The prediction method.
                
        Args:
            train_result: The trained result object containing the output
                          weights, reservoir states, etc.
            predict_length: The length of time to predict.
                            If predict_length is zero, no autonomous evolution
							of the reservoir will be performed. Instead, 
							external inputs are required to drive the reservoir
							and perform a mapping to a single target vector.
            inputs: The external inputs to the ESN.
            target_outputs: The target outputs during the testing period.
                            This is needed to compute performance statistics in
                            the PredictResult object.
            initial_state: The initial state of the reservoir at the beginning
                           of the prediction period.
                           If not provided, the final state of the train_result
                           will be used.
            resync_signal:  A signal used to synchronize the ESN and derive an
                           initial state.
                           If initial_state is also provided, resynchronization
                           will start from this state.
            mapper: A function that maps external inputs and ESN outputs to
                    ESN inputs.
                    Essentially, this function specifies what, if any, output
                    variables are fed back, and where.
            feature_function: The function that forms the feature vectors from
                              the reservoir states.
                              Only used if weights are provided in place of
                              train_result.
            lookback_states: Reservoir states prior to the prediction period to
							 to be used when the feature function requires 
							 time-delayed states. Must have shape
							 (feature_function.states_lookback_length + 1,
							  self.size)
            lookback_inputs: Reservoir inputs prior to the prediction period to 
							 use with feature functions that require 
							 time-delayed inputs. May also be used to provide 
							 an initial input for a feature function that 
							 requires only the current reservoir input.
							 Must have shape
							 (feature_function.inputs_lookback_length + 1,
							  self.input_dimension).
		
        Returns:
            result: The computed prediction result.
        """
        
        # Grab weights and feature function from train_result, if provided.
        if isinstance(train_result, TrainResult):
            weights = train_result.weights
            if feature_function is not None:
                msg = 'feature_function is provided, but ignored; ' \
                          'using train_result.feature_function ' \
                          'instead.'
                logging.warning(msg)
            feature_function = train_result.feature_function
        else:
            weights = train_result

        # Shape inputs, target_outputs, and resync_signal, if necessary.
        if inputs is not None:
            if len(inputs.shape) == 1:
                inputs = inputs[:, None]
        if target_outputs is not None:
            if len(target_outputs.shape) == 1:
                target_outputs = target_outputs[:, None]
        if resync_signal is not None:
            if len(resync_signal.shape) == 1:
                resync_signal = resync_signal[:, None]
        
        if predict_length == 0:
            if target_outputs is not None:
                utils.check_shape(target_outputs.shape, (1, None),
								  'target_outputs')
            if inputs is None:
                msg = "predict_length is zero but no inputs are provided."\
                      " Please provide a inputs to perform a mapping." 
                logging.error(msg)
            if initial_state is None and resync_signal is None:
                msg = "predict_length is zero but no initial state "\
                      "or resync signal is provided. Prediction will start " \
                      "from the final state of the provided train_result."
                logging.error(msg)
		
        # Automatically set the predict length, if not provided.
        elif predict_length is None:
            if inputs is None:
                if target_outputs is None:
                    msg = "Cannot infer predict_length from inputs or " \
                          "target_outputs; provide value for one of " \
                          "predict_length, inputs, or target_outputs."
                    logging.error(msg)
                else:
                    predict_length = target_outputs.shape[0]
            else:
                if target_outputs is None:
                    predict_length = inputs.shape[0]
                else:
                    predict_length = min(inputs.shape[0],
                                         target_outputs.shape[0])
                    if inputs.shape[0] > target_outputs.shape[0]:
                        msg = "Lengths of inputs and target_ouputs " \
                              "are inconsistent; using target_outputs " \
                              "to infer predict_length."
                        logging.warning(msg)
                    elif inputs.shape[0] < target_outputs.shape[0]:
                        msg = "Lengths of inputs and target_ouputs " \
                              "are inconsistent; using inputs " \
                              "to infer predict_length."
                        logging.warning(msg)
        
        # Check if the feature function requires a lookback period to include
        # time-delayed states or inputs in feature vectors.
        if hasattr(feature_function, 'inputs_lookback_length'):
            inputs_lookback_length = feature_function.inputs_lookback_length
        else: inputs_lookback_length = 0
        if hasattr(feature_function, 'states_lookback_length'):
            states_lookback_length = feature_function.states_lookback_length
        else: states_lookback_length = 0
        
        if lookback_states is not None:
            utils.check_shape(lookback_states.shape, (None, self.size),
							  'lookback_states')
            utils.check_range(lookback_states.shape[0], "lookback_states length",
							  states_lookback_length + 1, 'eq', True)
            if(resync_signal is not None):
                msg = "lookback_states and a resync_signal are both provided. " \
	                    "lookback_states will be ignored and calculated from " \
	                    "the resynchronization period instead."
                logging.warning(msg)
            elif(initial_state is not None):
                if (lookback_states[-1] != initial_state):
                    msg = "lookback_states and an initial_state are provided " \
	                    "but the final lookback_state does not match the " \
	                    "initial_state and will be replaced by initial_state."
                    logging.warning(msg)
			
        if lookback_inputs is not None:
            if(resync_signal is not None):
                msg = "lookback_inputs and a resync_signal are both provided. " \
	                    "lookback_inputs will be ignored and calculated from " \
	                    "the resynchronization period instead."
                logging.warning(msg)
            else:
                utils.check_shape(lookback_inputs.shape,
								  (None, self.input_dimension), 'lookback_inputs')
                utils.check_range(lookback_inputs.shape[0], "lookback_inputs length",
							  inputs_lookback_length + 1, 'eq', True)

        # If a resync signal is given:
        # - if no initial state is given, reset the reservoir to all zero state and
        # drive it with the resync signal
        # - if an initial state is given, set the reservoir to this initial
        # state and drive it with the resync signal.
        if resync_signal is not None:
            if initial_state is None:
                	resync_states = self.get_states(np.zeros(self.size), resync_signal)
            else:
                resync_states = self.get_states(initial_state, resync_signal)
            
            resync_outputs = feature_function(resync_states, resync_signal) @ weights
            lookback_states = resync_states[-(states_lookback_length + 1):
											].reshape((-1, self.size))
            lookback_inputs = resync_signal[-(inputs_lookback_length + 1):
											].reshape((-1, self.input_dimension))
   
        elif initial_state is not None:
            # Next, prioritize using a provided initial state.
            # We assume in this case that no initial input is needed, but log
            # this info for later debugging.
            # This will only cause an issue if the feature function needs an
            # input.
            # Shape the initial_state if provided as a 1D array.
            if len(initial_state.shape) == 1: initial_state = initial_state[None]
            if (lookback_states is not None):
                lookback_states[-1] = initial_state[0]
            else: lookback_states = initial_state
            if (lookback_inputs is None):
                lookback_inputs = np.zeros((1, self.input_dimension))
                msg = "No way of calculating initial_input is " \
                    "provided; this may cause problems if " \
                    "feature_function requires an input."
                logging.info(msg)
        
        elif isinstance(train_result, TrainResult):
            
            # Finally, if neither is provided, attempt to use the
            # TrainResult object.
            if(lookback_states is None):
                lookback_states = train_result.states[-(states_lookback_length + 1):]
                lookback_states = lookback_states.reshape((states_lookback_length + 1, -1))
            if(lookback_inputs is None):
                lookback_inputs = train_result.inputs[-(inputs_lookback_length + 1):]
                lookback_inputs = lookback_inputs.reshape((inputs_lookback_length + 1, -1))
			
        else:
            # If we get here, there was not enough information to calculate
            # an initial state and we must raise an error.
            msg = "Must provide a TrainResult object for " \
                    "train_result, or provide an " \
                    "initial_state, or provide a resync_signal."
            logging.error(msg)
        
        initial_state = lookback_states[-1][None]
        initial_input = lookback_inputs[-1][None]
        
        if predict_length == 0:
            states = self.get_states(initial_state.reshape((-1,)), inputs)
            outputs = feature_function(states, inputs) @ weights
                
        else:
            # If inputs aren't provided, just allocate space for them.
            if inputs is None:
                inputs = initial_input.repeat(predict_length, axis=0)
  
            # Calculate initial output.
            initial_output = feature_function(lookback_states,
                                          lookback_inputs) @ weights
            
            # Allocate memory for states and outputs.
            states = initial_state.repeat(predict_length + 1, axis=0)
            outputs = initial_output.repeat(predict_length + 1, axis=0)
            feedbacks = initial_input.repeat(predict_length + 1, axis=0)
            states = np.concatenate((lookback_states, states[1:]), axis = 0)
            feedbacks = np.concatenate((lookback_inputs, feedbacks[1:]), axis = 0)

            # Try first to use the jitted version of _get_states_autonomous for
            # faster performance.
            try: 

                # For speed and successful jitting in certain cases, ensure that
                # the data arrays are contiguous.
                if not inputs.data.contiguous:
                    inputs = np.ascontiguousarray(inputs)
                if not outputs.data.contiguous:
                    outputs = np.ascontiguousarray(outputs)
                if not states.data.contiguous:
                    states = np.ascontiguousarray(states)
                if not feedbacks.data.contiguous:
                    feedbacks = np.ascontiguousarray(feedbacks)				

                # If function is not jitted, attempt to jit it and verify it works.
                if not hasattr(feature_function, 'inspect_llvm'):
                    feature_function_jit = numba.jit(nopython=True,
                                           fastmath=True)(feature_function)
                    _ = feature_function_jit(states, feedbacks)
                    msg = "Successfully compiled feature_function."
                    logging.info(msg)
                
                # Otherwise, grab the already jitted function.
                else:
                    feature_function_jit = feature_function
            
                # If successful, calculate states and outputs using the compiled
                # state propagation function.
                states, outputs = _get_states_autonomous_jit(inputs, 
                              outputs, states, feedbacks, feature_function_jit,
                              mapper, states_lookback_length,
                              inputs_lookback_length,
							  self.A.data, self.A.indices,
                              self.A.indptr, self.A.shape, self.B, self.C,
                              weights, self.leaking_rate)
            
            # If function is not jittable or for some other reason does not run
            # with the jitted _get_states_autonomous, 
            except (TypeError, numba.UnsupportedError):
                msg = "Could not compile the autonomous state " \
                      "propagation function. Trying a non-compiled " \
                      "version instead."
                logging.warning(msg)
                states, outputs = _get_states_autonomous(inputs, 
                              outputs, states, feedbacks, feature_function,
                              mapper, states_lookback_length,
                              inputs_lookback_length,
                              self.A.data, self.A.indices,
                              self.A.indptr, self.A.shape, self.B, self.C,
                              weights, self.leaking_rate)  
        
        if resync_signal is None:
            return PredictResult(inputs, outputs, states, predict_length,
								 target_outputs, None, None, None)
        else:
            return PredictResult(inputs, outputs, states, predict_length,
								 target_outputs, resync_signal, resync_states,
								 resync_outputs)
		

@validate_arguments(config=dict(arbitrary_types_allowed=True))    
def optimize_hyperparameters(
    esn:                 ESN,
    train_args:          Union[dict, List[dict]],
    predict_args:        Union[dict, List[dict]],
    loss:                Optional[Callable]          = None,
    optimizer:           Optional[Callable]          = None,
    seed:                Union[int, None, Generator] = None,
    allow_matrix_flip:   Optional[bool]              = True,
    allow_negative_leak: Optional[bool]              = True,
    verbose:             bool                        = True,
    ):
    """The Optimize Hyperparameters function.
    
    Takes an existing rescompy.ESN object and optimizes several
    hyperparameters, minimizing the prediction error for a given task or sets
    of tasks.
    
    Args:
        esn (ESN): The unoptimized ESN object.
        train_args (dict, List[dict]): The arguments for the training method.
        predict_args(dict, List[dict]): The arguments for the prediction
                                        method.
        loss (Callable): The function for computing the loss from a prediction
                         result.
                         If not provided, a default loss will be provided that
                         returns predict_result.rmse
        optimizer (Callable): The function for minimizing a scalar function.
                              If not provided, a Nelder-Mead optimizer will be
                              used.
        seed (int, None, Generator): An integer for determining the random seed
                                     of the random matrices.
                                     
    Returns:
        new_esn (ESN): The optimized ESN object.
    """

    if loss is None:
        def loss(predict_result):
            return np.mean(predict_result.rmse)

    if optimizer is None:
        def optimizer(func, x0):
            result = minimize(func, x0, method='Nelder-Mead')
            return result.x
    
    if not isinstance(train_args, list):
        train_args = [train_args]

    if not isinstance(predict_args, list):
        predict_args = [predict_args]
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    # Copy the referenced ESN.
    base_esn = copy(esn)
    
    # Normalize the spectral radius, input strength, and bias strength.
    v0 = rng.random(base_esn.size)
    eigenvalues, _ = splinalg.eigs(base_esn.A, k=1, v0=v0)
    base_esn.A *= 1/np.abs(eigenvalues[0])
    base_esn.B *= 1/np.max(np.abs(base_esn.B))
    if np.max(np.abs(base_esn.C)) != 0:
        base_esn.C *= 1/np.max(np.abs(base_esn.C))
    
    if verbose:
        header = f"{'spectral_radius':15s} | {'input_strength':15s} " \
                 f"| {'bias_strength':15s} | {'leaking_rate':15s} " \
                 f"|| {'loss':15s}"
        print(header)
    
    # Define an objective function based on the specifications.
    def loss_outer(x):
        
        # Copy the referenced ESN.
        new_esn = copy(base_esn)
        
        # Adjust the hyperparameters.
        if allow_matrix_flip:
            new_esn.A *= x[0]
            new_esn.B *= x[1]
            new_esn.C *= x[2]
        else:
            new_esn.A *= np.abs(x[0])
            new_esn.B *= np.abs(x[1])
            new_esn.C *= np.abs(x[2])
        if allow_negative_leak:
            new_esn.leaking_rate = x[3]
        else:
            new_esn.leaking_rate = np.abs(x[3])
        
        cumulative_loss = 0
        for i in range(len(train_args)):
            train_result = new_esn.train(**train_args[i])
            predict_result = new_esn.predict(train_result,
                                             **predict_args[i])
            cumulative_loss += loss(predict_result)

        if verbose:
            line = f"{x[0]:>15.12f} | {x[1]:>15.12f} " \
                     f"| {x[2]:>15.12f} | {x[3]:>15.12f} " \
                     f"|| {cumulative_loss:>15.12f}"
            print(line)
        return cumulative_loss
    
    # Optimize the outer loss function.
    x0 = [esn.spectral_radius, esn.input_strength, esn.bias_strength,
          esn.leaking_rate]
    result = optimizer(loss_outer, x0)
    
    # Return an ESN with the optimal hyperparameters.
    new_esn = copy(base_esn)
    new_esn.A *= result[0]
    new_esn.B *= result[1]
    new_esn.C *= result[2]
    new_esn.bias_strength = np.abs(result[2])
    new_esn.leaking_rate = result[3]
    
    # Raise some warnings if odd results were found.
    if result[0] < 0 and not allow_matrix_flip:
        msg = "The sign of the input matrix was flipped during " \
                  "optimization."
        logging.warning(msg)
    if result[1] < 0 and not allow_matrix_flip:
        msg = "The sign of the recurrent matrix was flipped during " \
                  "optimization."
        logging.warning(msg)
    if result[2] < 0 and not allow_matrix_flip:
        msg = "The sign of the bias vector was flipped during " \
                  "optimization."
        logging.warning(msg)
    if result[3] < 0 and not allow_negative_leak:
        msg = "The sign of the leaking rate is negative after " \
                  "optimization."
        logging.warning(msg)
    return new_esn

    
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def copy(
    esn:      ESN,
    new_seed: Optional[int] = None
    ) -> ESN:
    """The rescompy Copy function.
    
    Takes an existing rescompy.ESN object and returns a shallow copy.
    
    If a new seed is provided, will copy only the hyperparameters and generate
    a new ESN accordingly.

    Args:
        esn: The ESN to be copied.
        new_seed: The seed for the new ESN.
                  If None, will use the seed from the existing ESN; that is,
                  the new ESN will have the exact same random matrices.

    Returns:
        new_esn: The copied ESN.
    """
    
    args = esn.__dict__
    if new_seed is None:
        seed = args['seed']
    else:
        seed = new_seed
    new_esn = ESN(input_dimension=args['input_dimension'],
                  size=args['size'],
                  connections=args['connections'],
                  spectral_radius=args['_spectral_radius'],
                  input_strength=args['_input_strength'],
                  bias_strength=args['_bias_strength'],
                  leaking_rate=args['leaking_rate'],
                  seed=seed)
    new_esn.A = esn.A.copy()
    new_esn.B = esn.B.copy()
    new_esn.C = esn.C.copy()
    return new_esn


@numba.jit(nopython=True, fastmath=True)
def _mult_vec(
    data:    np.ndarray,
    indices: np.ndarray,
    indptr:  np.ndarray,
    shape:   tuple,
    mat:     np.ndarray
    ) -> np.ndarray:
    """The compiled Multiply Vectors function.
    
    A compiled function for quick multiplication of a sparse matrix with a
    dense matrix.
    
    This function is intended for internal use and does not provide type- or
    shape-checking.
    
    Args:
        data (np.ndarray): CSR format data array of the sparse matrix.
        indices (np.ndarray): CSR format index array of the sparse matrix.
        indptr (np.ndarray): CSR format index pointer array of the sparse
                             matrix.
        shape (tuple): The shape of the sparse matrix.
        mat (np.ndarray): The dense matrix.
    Returns:
        out (np.ndarray): The multiplication result.
    """

    out = np.zeros(shape[0])
    for i in range(mat.size):
        for k in range(indptr[i], indptr[i+1]):
            out[indices[k]] += data[k] * mat[i]
    return out


@numba.jit(nopython=True, fastmath=True)
def _get_states_driven(
    u:         np.ndarray,
    r:         np.ndarray,
    A_data:    np.ndarray,
    A_indices: np.ndarray,
    A_indptr:  np.ndarray,
    A_shape:   tuple,
    B:         np.ndarray,
    C:         np.ndarray,
    leakage:   float,
    ) -> np.ndarray:
    """The Get Driven States function.
    
    A compiled function for quick computation of reservoir states subject to a
    driving input signal.
    
    This function is intended for internal use and does not provide type- or
    shape-checking.
    
    Args:
        u (np.ndarray): Reservoir inputs.
        r (np.ndarray): Reservoir states.
        A_data (np.ndarray): CSR format data array of the adjacency matrix A.
        A_indices (np.ndarray): CSR format index array of the adjacency matrix
                                A.
        A_indptr (np.ndarray): CSR format index pointer array of the adjacency
                               matrix A.
        A_shape (np.ndarray): The shape of the adjacency matrix A.
        B (np.ndarray): The input matrix.
        C (np.ndarray): The bias vector.
        leakage (float): The leaking rate.
    
    Returns:
        r (np.ndarray): The computed reservoir states.
    """    

    for i in range(r.shape[0]-1):
        r[i+1] = (1.0-leakage)*r[i] + leakage*np.tanh(B @ u[i]
            + _mult_vec(A_data, A_indices, A_indptr, A_shape, r[i])
            + C)
    return r[1:]


def _get_states_autonomous(
    u:                       np.ndarray,
    v:                       np.ndarray,
    r:                       np.ndarray,
	f:                       np.ndarray,
    feature:                 Callable,
    mapper:                  Callable,
    states_lookback:         int,
    inputs_lookback:         int,
    A_data:                  np.ndarray,
    A_indices:               np.ndarray,
    A_indptr:                np.ndarray,
    A_shape:                 tuple,
    B:                       np.ndarray,
    C:                       np.ndarray,
    W:                       np.ndarray,
    leakage:                 float,
    ) -> np.ndarray:
    """The uncompiled Get Autonomous States function.
    
    A function for quick computation of reservoir states in closed-loop mode.
    
    This function is intended for internal use and does not provide type- or
    shape-checking.
    
    Args:
        u (np.ndarray): Reservoir inputs.
        v (np.ndarray): Reservoir outputs.
        r (np.ndarray): Reservoir states.
		f (np.ndarray): Feedback. Reservoir inputs as defined by mapper.
        feature (Callable): The feature function.
        mapper (Callable): A function defining the feedback.
        states_lookback (int): The number of times steps in the past 
                             required to reach the first reservoir state 
                             included in the feature vector for the current
                             time step.
        inputs_lookback (int): The number of times steps in the past 
                             required to reach the first reservoir input 
                             included in the feature vector for the current
                             time step.
        A_data (np.ndarray): CSR format data array of the adjacency matrix A.
        A_indices (np.ndarray): CSR format index array of the adjacency matrix
                                A.
        A_indptr (np.ndarray): CSR format index pointer array of the adjacency
                               matrix A.
        A_shape (np.ndarray): The shape of the adjacency matrix A.
        B (np.ndarray): The input matrix.
        C (np.ndarray): The bias vector.
        W (np.ndarray): The output weights.
        leakage (float): The leaking rate.
    
    Returns:
        r (np.ndarray): The computed reservoir states.
    """

    diff = states_lookback - inputs_lookback
    for i in range(states_lookback, r.shape[0]-1):
        f[i+1-diff] = mapper(u[i-states_lookback], v[i-states_lookback])
        r[i+1] = (1.0-leakage)*r[i] + leakage*np.tanh(
            B @ f[i+1-diff]
            + _mult_vec(A_data, A_indices, A_indptr, A_shape, r[i])
            + C)
        v[i-states_lookback+1] = feature(
		 np.reshape(r[i+1-states_lookback: i+2], (states_lookback+1,-1)),
		 np.reshape(f[i+1-diff-inputs_lookback: i+2-diff],
		   (inputs_lookback+1,-1))
		 ) @ W
    return r[states_lookback+1:], v[1:]

_get_states_autonomous_jit = numba.jit(nopython=True, fastmath=True)(_get_states_autonomous)


def _calc_D(states_train, inputs_train, size, A, B, C):
    initial_state = states_train[0,:]

    # calculate D, which is f'(a) where a is the activation at each node
    # D is a matrix of size (num_timesteps x reservoir dimension)

    # assume that state at time t = -1 is the same as the initial state
    Ar = (A @ np.vstack((np.reshape(initial_state, (1, size)), states_train)).T).T
    Bu = (B @ inputs_train.T).T
    D = np.square(np.reciprocal(np.cosh(Ar[:-1,:] + Bu + C)))
    return D

def _calc_dr_du(states_train, inputs_train, size, 
    input_dimension, A, B, C, leaking_rate):

    D = _calc_D(states_train, inputs_train, size, 
        A, B, C)

    num_timesteps = states_train.shape[0]

    # dr_du at time step i is a (reservoir dimension x # of inputs)
    # jacobian matrix = D[i] @ B
    dr_du = np.zeros((num_timesteps, size, input_dimension))
    for i in range(num_timesteps):
        dr_du[i,:,:] = leaking_rate*(D[i,:] * B.T).T
    return dr_du