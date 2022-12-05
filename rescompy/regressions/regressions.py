"""regressions.py

The Regressions submodule for rescompy.

This module implements training algorithms commonly used in reservoir
computing.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'



import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional, Union
from pydantic import validate_arguments
from ..utils import utils


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tikhonov(
    reg:    Optional[float] = 1e-6,
    solver: Optional[str]   = 'cholesky',
    ):
    """The Default Regression function.
    
    Performs a ridge regression fit.
    
    Args:
        reg (float): The ridge regression parameter.
        solver (str): The method for solving the regression problem.
                      See sklearn.linear_model.Ridge for details.

    Returns:
        A function that performs a ridge regression
        that takes         
            s (np.ndarray): The feature vectors
            v (np.ndarray): The target outputs
        and returns
            W (np.ndarray): The fitted weights
    """

    def inner(
        s:      np.ndarray,
        v:      np.ndarray
    ):
        clf = Ridge(alpha=reg, fit_intercept=False, solver=solver)
        clf.fit(s, v)
        return clf.coef_.T
    return inner

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def jacobian(
    beta_T: Optional[float] = 1e-6,
    beta_J: Optional[float] = 1e-6,
    ):
    """The Default Regression function.
    
    Performs a ridge regression fit.
    
    Args:
        beta_T (float): The Tikhonov regularization parameter
        beta_J (float): The Jacobian regularization parameter

    Returns:
        A function that performs a ridge regression
        that takes         
            s (np.ndarray): The feature vectors
            v (np.ndarray): The target outputs
            u (np.ndarray): The inputs
            dg_du (np.ndarray): The derivative of the states wrt the input
        and returns
            W (np.ndarray): The fitted weights
    """

    def inner(
        s:      np.ndarray,
        v:      np.ndarray,
        dg_du:  np.ndarray
    ):

        T_train = s.shape[0]
        R_T = np.eye(s.shape[1])
        R_J = np.tensordot(np.transpose(dg_du, axes=[2, 0, 1]), 
            np.transpose(dg_du, axes=[0, 2, 1]), axes=([1,0], [0,1]))/T_train

        return np.linalg.solve(s.T @ s + beta_T*R_T + beta_J*R_J, s.T @ v)
    return inner

'''
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def mean_jacobian(
    mean_jacobian: np.ndarray,
    beta_T: Optional[float] = 1e-6,
    beta_J: Optional[float] = 1e-6,
    ):
    """The Default Regression function.
    
    Performs a ridge regression fit.
    
    Args:
        beta_T (float): The Tikhonov regularization parameter
        beta_J (float): The Jacobian regularization parameter

    Returns:
        A function that performs a ridge regression
        that takes         
            s (np.ndarray): The feature vectors
            v (np.ndarray): The target outputs
            u (np.ndarray): The inputs
            dg_du (np.ndarray): The derivative of the states wrt the input
        and returns
            W (np.ndarray): The fitted weights
    """

    def inner(
        s:      np.ndarray,
        v:      np.ndarray,
    ):

        R_T = s.T @ s + beta_T * np.eye(s.shape[1])
        R_J = beta_J*mean_jacobian

        return np.linalg.solve(R_T+R_J, s.T @ v)
    return inner    
'''

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def batched_ridge(
    regularization:    Union[float, np.ndarray] = 1e-6,
    prior_guess:       Optional[np.ndarray] = None,
    ):
    """The Default Batched Ridge Regression function.
    
    Args:
	    regularization: The regularization strength in the ridge regression formula.
	                    If passed as a float, this value will regularize all features.
	                    To regularize each feature differently, an array of
						regularization strengths may be passed.
		prior_guess (np.ndarray): An initial guess for the output weights.
		
    Returns:
        A function that performs a ridge regression
        that takes 
            SS_T (np.ndarray): The feature vectors information matrix
            VS_T (np.ndarray): The targets and features information matrix
        and returns
            W (np.ndarray): The fitted weights
    """

    # Add error message for invalid combination of arguments (if required)

    if (prior_guess is None and isinstance(regularization, float)):
        def inner(
            SS_T:      np.ndarray,
            VS_T:      np.ndarray
        ):
            weights = np.linalg.solve(
			    SS_T + regularization * np.eye(SS_T.shape[0]),
			    VS_T
			    )
            return weights

    elif(isinstance(prior_guess, np.ndarray) and isinstance(regularization, float)):
        def inner(
            SS_T:      np.ndarray,
            VS_T:      np.ndarray
        ):
            name = "prior_guess"
            utils.check_shape(prior_guess.shape,
							  (SS_T.shape[0], VS_T.shape[1]), name)
            weights = np.linalg.solve(
			    SS_T + regularization * np.eye(SS_T.shape[0]),
			    VS_T + regularization * np.eye(SS_T.shape[0]) @ prior_guess
			    )
            return weights    

    elif(prior_guess is None and isinstance(regularization, np.ndarray)):
        def inner(
            SS_T:      np.ndarray,
            VS_T:      np.ndarray
        ):
            name = "regularization"
            utils.check_shape(regularization.shape, SS_T.shape, name)
            weights = np.linalg.solve(
			    SS_T + regularization,
			    VS_T
			    )
            return weights
    
    elif(isinstance(prior_guess, np.ndarray) and isinstance(regularization, np.ndarray)):
        def inner(
            SS_T:      np.ndarray,
            VS_T:      np.ndarray
        ):
            name = "regularization"
            utils.check_shape(regularization.shape, SS_T.shape, name)
            name = "prior guess"
            utils.check_shape(prior_guess.shape,
							  (regularization.shape[0], VS_T.shape[1]), name)
            weights = np.linalg.solve(
			    SS_T + regularization,
			    VS_T + regularization @ prior_guess
			    )
            return weights
        
    return inner