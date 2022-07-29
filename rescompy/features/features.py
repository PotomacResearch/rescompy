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


#@numba.jit(nopython=True, fastmath=True)
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
    if len(r.shape) == 2:
        const = np.zeros((r.shape[0], 1)) + 1
        s = np.hstack((s, const))    
    else:
        s = np.hstack((s, np.array([1])))
    return s


#@numba.jit(nopython=True, fastmath=True)
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
    if len(r.shape) == 2:
        const = np.zeros((r.shape[0], 1)) + 1
        s = np.hstack((s, const))    
    else:
        s = np.hstack((s, np.array([1])))
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
    
    #@numba.jit(nopython=True, fastmath=True)
    def polynomial(r, u):
        s = np.copy(r)
        for poly_ind in range(2, degree+1):
            if len(r.shape) == 2:
                s = np.concatenate((s, r**poly_ind), axis=1)
            else:
                s = np.concatenate((s, r**poly_ind), axis=0)
        return s
    
    return polynomial