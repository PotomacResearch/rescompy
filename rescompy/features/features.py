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
import logging


@numba.jit(nopython=True, fastmath=True)
def states_only(
    r:                  np.ndarray,
    u:                  np.ndarray,
    ) -> np.ndarray:
    """The States-only Feature function.
    
    Simply returns the reservoir state, unaltered.
    
    Args:
        r (np.ndarray): The reservoir states.
        u (np.ndarray): The input signal.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    s = np.copy(r)    
    return s


@numba.jit(nopython=True, fastmath=True)
def states_and_inputs(
    r:                  np.ndarray,
    u:                  np.ndarray,
    ) -> np.ndarray:
    """The States-and-Inputs Feature function.
    
    Concatenates the reservoir states with the inputs.
    
    Args:
        r (np.ndarray): The reservoir states.
        u (np.ndarray): The input signal.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    s = np.copy(r)
    s = np.hstack((r, u))    
    return s


@numba.jit(nopython=True, fastmath=True)
def states_and_constant(
    r:                  np.ndarray,
    u:                  np.ndarray,
    ) -> np.ndarray:
    """The States-and-Constant Feature function.
    
    Concatenates the reservoir states with a constant.
    
    Args:
        r (np.ndarray): The reservoir states.
        u (np.ndarray): The input signal.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """

    s = np.copy(r)    
    const = np.zeros((r.shape[0], 1)) + 1
    s = np.hstack((s, const))    
    return s


@numba.jit(nopython=True, fastmath=True)
def states_and_inputs_and_constant(
    r:                  np.ndarray,
    u:                  np.ndarray,
    ) -> np.ndarray:
    """The States-and-Inputs-and-Constant Feature function.
    
    Concatenates the reservoir states with the inputs and a constant.
    
    Args:
        r (np.ndarray): The reservoir states.
        u (np.ndarray): The input signal.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    s = np.hstack((r, u))
    const = np.zeros((r.shape[0], 1)) + 1
    s = np.hstack((s, const))    
    return s


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_polynomial(
    degree: int = 2,
    ) -> np.ndarray:
    """The Polynomial Feature-getting function.
    
    Returns feature function that returns a concatenation of
    r, r^2, ..., r^degree.
    
    Args:
        degree (int): The maximum degree of the polynomial.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    @numba.jit(nopython=True, fastmath=True)
    def polynomial(r, u):
        s = np.copy(r)
        for poly_ind in range(2, degree+1):
            s = np.concatenate((s, r**poly_ind), axis=1)
        return s
    
    return polynomial


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
    
    return mixed_reservoir_states


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def Time_Shifted_feature(
        lookback_length:   int = 1,
        decimation:        int = 1 
    ):
    """The time-shifted states feature-getting function.
    
    Returns feature function that returns a concatenation of
    of time-shifted reservoir states.
    
    Args:
        lookback_length (int): The number of previous time steps required to
                               form the feature vector at the current time
                               step.
        decimation (int): The decimation time between states.
        
    Returns:
        s (np.ndarray): The feature vectors.
    """
    
    if (decimation > lookback_length):
        msg = "The decimation time is larger than the lookback_length. Feature " \
           "vectors will consist of the current reservoir state only."
        logging.warning(msg)
    
    @numba.jit(nopython = True, fastmath = True)
    def shifted_nodes(r, u):
        r = r.reshape((-1, r.shape[-1]))
        features = r[lookback_length:]
        for shift in range(decimation, lookback_length + 1, decimation):
            features = np.concatenate(
			    (features, r[lookback_length-shift:-shift]),
			    axis = 1)
        return features
    
    return shifted_nodes


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