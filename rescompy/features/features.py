"""features.py

The Features submodule for recomspy.

This module contains commonly-used feature vector functions.

A feature function takes a set of reservoir states and produces the features
that enter a training algorithm.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


import numba
import numpy as np
from pydantic import validate_arguments
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

class Feature(ABC):
    """ 
    The Feature Abstract Base Class
    If you want to use a class (instead of just a Callable), the 
    following functions need to be defined.
    """

    """
    __call__ -- overload the call dunder method to do the
    actual extraction of a feature from the reservoir states
    and the input signal

        Args:
            r (np.ndarray): The reservoir states.
            u (np.ndarray): The input signal.
            
        Returns:
            s (np.ndarray): The feature vectors.
    """
    @abstractmethod
    def __call__(r: np.ndarray, u: np.ndarray):
        pass

    """
    jacobian -- calculates the jacobian of the feature
    vector with respect to the changes in the inputs.
    Most of the features below do not depend on the 
    specific value of the inputs, but these are included
    for customized hybrid functions.

    The Jacobian is a 3-D tensor ordered as (timestep, 
    feature dimension, input dimension).  See examples below
    for more details.

        Args:
            r (np.ndarray): The reservoir states
            u (np.ndarray): The input signal.
            dr_du (np.ndarray): The derivative of the reservoir states
                with respect to the inputs.  3-D tensor ordered the
                way the output should be ordered.
            
        Returns:
            dg_du (np.ndarray): The derivative of the feature vectors 
                with respect to inputs.
    """
    @abstractmethod
    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        pass

    """
    feature_size -- calculates the size of the feature
    vector as a function of the esn size and the number of inputs.

        Args:
            esn_size (int): The size of the reservoir
            input_dim (np.ndarray): The dimensionality of the input signal.
            
        Returns:
            size (np.ndarray): The size of the feature vector.
    """    
    @abstractmethod
    def feature_size(self, esn_size:int, input_dim:int):
        pass


class StatesOnly(Feature):
    """The States-only Feature function.
    Simply returns the reservoir state, unaltered."""
    @staticmethod
    def __call__(r: np.ndarray, u: np.ndarray):
        s = np.copy(r)    
        return s
    
    @staticmethod
    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        return dr_du

    @staticmethod
    def feature_size(esn_size:int, input_dim: int):
        return esn_size

class StatesAndInputs(Feature):
    """The States-and-Inputs Feature function.
    
    Concatenates the reservoir states with the inputs."""

    @staticmethod
    def __call__(r: np.ndarray, u: np.ndarray):
        s = np.hstack((r, u))    
        return s

    @staticmethod
    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        return np.hstack((
            dr_du, 
            np.tile(np.eye(u.shape[1]), (dr_du.shape[0], 1,1))
        ))

    @staticmethod
    def feature_size(esn_size,input_dim): 
        return esn_size+input_dim

class StatesAndConstant(Feature):
    """The States-and-Constant Feature function.
    
    Concatenates the reservoir states with a constant. """
    @staticmethod
    def __call__(r: np.ndarray, u: np.ndarray):
        s = np.copy(r)    
        const = np.zeros((r.shape[0], 1)) + 1
        s = np.hstack((s, const))    
        return s

    @staticmethod
    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        return np.hstack((
            dr_du, 
            np.zeros((dr_du.shape[0], 1, u.shape[1]))
        ))
    
    @staticmethod
    def feature_size(esn_size,input_dim): 
        return 1+esn_size

class StatesAndInputsAndConstant(Feature):
    """The States-and-Inputs-and-Constant Feature function.
    
    Concatenates the reservoir states with the inputs and a constant."""
    @staticmethod
    def __call__(r: np.ndarray, u: np.ndarray):
        s = np.hstack((r, u))
        const = np.zeros((r.shape[0], 1)) + 1
        s = np.hstack((s, const))    
        return s

    @staticmethod
    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        return np.hstack((
            dr_du, 
            np.tile(np.eye(u.shape[1]), (dr_du.shape[0], 1,1)),
            np.zeros((dr_du.shape[0], 1, u.shape[1]))
        ))

    @staticmethod
    def feature_size(esn_size,input_dim): 
        return 1+input_dim+esn_size

@dataclass
class ConstantInputAndPolynomial(Feature):
    """The Polynomial Feature-getting function.
    
    Returns feature function that returns a concatenation of
    [1, u, r, r^2, ..., r^degree.
    
    Args:
        degree (int): The maximum degree of the polynomial.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    degree: int = 2
    
    def __call__(self, r, u):
        const = np.zeros((r.shape[0], 1)) + 1
        s = np.hstack((const, u, r))
        for poly_ind in range(2, self.degree+1):
            s = np.concatenate((s, r**poly_ind), axis=1)
        return s

    def feature_size(self, esn_size:int,input_dim:int): 
        return 1+input_dim+esn_size*self.degree


    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        raise NotImplementedError()

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def MRS_feature(
        decimation:        int = 1,
        max_num_states:    int = 10  
    ):
    """The Mixed Reservoir State feature-getting function.
    
    Returns feature function that returns a concatenation of
    of evenly-spaced reservoir states over a driving period.
    
    Args:
        decimation (int): The decimation time between states.
        max_num_states (int): The maximum number of reservoir states to 
		                      concatenate into a single MRS feature vector
                              (may be required for memory purposes).
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    @numba.jit(nopython = True, fastmath = True)
    def mixed_reservoir_states(r, u) -> np.ndarray:
        r = r.reshape((-1, r.shape[-1])) 
        num_time_steps = r.shape[0]
        num_states = min((num_time_steps - 1) // decimation + 1, max_num_states)
        chosen_states = num_time_steps - 1 \
			- np.linspace(0, decimation * (num_states - 1), num_states).astype(np.int32)
        s = r[chosen_states].reshape((-1, num_states * r.shape[-1]))
        
        return s
    
    #mixed_reservoir_states.lookback_length = chosen_states[-1]
    
    return mixed_reservoir_states


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def states_and_inputs_time_shifted(
        states_lookback_length:   int = 0,
        inputs_lookback_length:   int = 0,
        states_decimation:        int = 1,
        inputs_decimation:        int = 1 
    ):
    """The time-shifted states feature-getting function.
    
    Returns feature function that returns a concatenation of
    of time-shifted reservoir states and inputs.
    
    Args:
        states_lookback_length (int): The number of times steps in the past 
                               required to reach the first reservoir state 
                               included in the feature vector for the current
                               time step.
        inputs_lookback_length (int): The number of times steps in the past 
                               required to reach the first reservoir input 
                               included in the feature vector for the current
                               time step.
        states_decimation (int): The decimation time between states.
        inputs_decimation (int): The decimation time between inputs.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    if (states_decimation < 1):
        msg = "states_decimation must be greater than or equal to 1."
        logging.error(msg)
        raise(ValueError(msg))
        
    if (inputs_decimation < 1):
        msg = "inputs_decimation must be greater than or equal to 1."
        logging.error(msg)
        raise(ValueError(msg))

    if (states_lookback_length < 0):
        msg = "states_lookback_length must be greater than or equal to 0."
        logging.error(msg)
        raise(ValueError(msg))
        
    if (inputs_lookback_length < 0):
        msg = "inputs_lookback_length must be greater than or equal to 0."
        logging.error(msg)
        raise(ValueError(msg))
    
    if (inputs_decimation > inputs_lookback_length and inputs_decimation > 1):
        msg = "The inputs decimation time is larger than inputs_lookback_length. "\
		      "Feature vectors will contain the current reservoir input only."
        logging.warning(msg)
		
    if (states_decimation > states_lookback_length and states_decimation > 1):
        msg = "The states decimation time is larger than states_lookback_length."\
		      "Feature vectors will contain the current reservoir state only."
        logging.warning(msg)
    
    lookback_length = max(states_lookback_length, inputs_lookback_length)
    
    @numba.jit(nopython = True, fastmath = True)
    def time_delayed(r, u):
        r = r.reshape((-1, r.shape[-1]))
        u = u.reshape((-1, u.shape[-1]))
        
        if (r.shape[0] == states_lookback_length + 1 and
			u.shape[0] == inputs_lookback_length + 1):
            s = r[-1].reshape((1, -1))
            for shift in range(states_decimation, states_lookback_length + 1,
							   states_decimation):
                s = np.hstack((s, r[-(shift+1)].reshape((1, -1))))
            
            s = np.hstack((s, u[-1].reshape((1, -1))))
            for shift in range(inputs_decimation, inputs_lookback_length + 1,
							   inputs_decimation):
                s = np.hstack((s, u[-(shift+1)].reshape((1, -1))))
        
        else:
            s = r[lookback_length:]
            for shift in range(states_decimation, states_lookback_length + 1,
							   states_decimation):
                s = np.hstack((s, r[lookback_length-shift:-shift]))
        
            s = np.hstack((s, u[lookback_length:]))
            for shift in range(inputs_decimation, inputs_lookback_length + 1,
							   inputs_decimation):
                s = np.hstack((s, u[lookback_length-shift:-shift]))
        
        return s
    
    time_delayed.states_lookback_length = states_lookback_length
    time_delayed.inputs_lookback_length = inputs_lookback_length
    time_delayed.lookback_length = lookback_length
    
    return time_delayed


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def states_only_time_shifted(
        states_lookback_length:   int = 0,
        states_decimation:        int = 1 
    ):
    """The time-shifted states feature-getting function.
    
    Returns feature function that returns a concatenation of
    of time-shifted reservoir states.
    
    Args:
        states_lookback_length (int): The number of times steps in the past 
                               required to reach the first reservoir state 
                               included in the feature vector for the current
                               time step.
        states_decimation (int): The decimation time between states.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    if (states_decimation < 1):
        msg = "states_decimation must be greater than or equal to 1."
        logging.error(msg)
        raise(ValueError(msg))
    		
    if (states_decimation > states_lookback_length):
        msg = "The states decimation time is larger than states_lookback_length."\
		      "Feature vectors will contain the current reservoir state only."
        logging.warning(msg)
    
    @numba.jit(nopython = True, fastmath = True)
    def time_delayed(r, u):
        r = r.reshape((-1, r.shape[-1]))
        u = u.reshape((-1, u.shape[-1]))
        
        s = r[states_lookback_length:]
        for shift in range(states_decimation, states_lookback_length + 1, states_decimation):
            s = np.hstack((s, r[states_lookback_length-shift:-shift]))

        return s
    
    time_delayed.states_lookback_length = states_lookback_length
    time_delayed.lookback_length = states_lookback_length
    
    return time_delayed


@numba.jit(nopython=True, fastmath=True)
def final_state_only(
    r:                  np.ndarray,
    u:                  np.ndarray,
    ) -> np.ndarray:
    """The States-only Feature function.
    
    Simply returns the final reservoir state of a driving period.
    
    Args:
        r (np.ndarray): The reservoir states.
        u (np.ndarray): The input signal.
        
    Returns:
        s (np.ndarray): The feature vector.
    """
    
    r = r.reshape((-1, r.shape[-1]))
    s = np.copy(r[-1])    
    return s