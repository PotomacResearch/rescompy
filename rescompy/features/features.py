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

states_only.jacobian = lambda dr_du, u: dr_du



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
    
    s = np.hstack((r, u))    
    return s

states_and_inputs.jacobian = lambda dr_du, u: \
    np.hstack((
        dr_du, 
        np.tile(np.eye(u.shape[1]), (dr_du.shape[0], 1,1))
    ))


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

states_and_constant.jacobian = lambda dr_du, u: \
    np.hstack((
        dr_du, 
        np.zeros((dr_du.shape[0], 1, u.shape[1]))
    ))


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

states_and_inputs_and_constant.jacobian = lambda dr_du, u: \
    np.hstack((
        dr_du, 
        np.tile(np.eye(u.shape[1]), (dr_du.shape[0], 1,1)),
        np.zeros((dr_du.shape[0], 1, u.shape[1]))
    ))

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