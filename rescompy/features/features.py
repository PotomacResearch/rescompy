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
