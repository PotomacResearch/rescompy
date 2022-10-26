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
def default(
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
def batched_ridge(
    regularization:    Union[float, np.ndarray] = 1e-6,
    prior_guess:       Optional[np.ndarray] = None,
    ):
    """The Default Batched Ridge Regression function.
    
    Performs a ridge regression fit.
    
    Args:
    regularization: The regularization strength in the ridge regression formula.
                    If passed as a float, this value will regularize all features.
                    To regularize each feature differently, an array of
					regularization strengths may be passed.
	prior_guess (np.ndarray): An initial guess for the output weights.

    Returns:
        A function that performs a ridge regression
        that takes         
            RR_T (np.ndarray): The reservoir state-only information matrix
            YR_T (np.ndarray): The targets and reservoir state information matrix
        and returns
            W (np.ndarray): The fitted weights
    """

    # Add error message for invalid combination of arguments (if required)

    if (prior_guess is None and isinstance(regularization, float)):
        def inner(
            RR_T:      np.ndarray,
            YR_T:      np.ndarray
        ):
            weights = np.linalg.solve(
			    RR_T + regularization * np.eye(RR_T.shape[0]),
			    YR_T
			    )
            return weights

    elif(isinstance(prior_guess, np.ndarray) and isinstance(regularization, float)):
        def inner(
            RR_T:      np.ndarray,
            YR_T:      np.ndarray
        ):
            name = "prior_guess"
            utils.check_shape(prior_guess.shape,
							  (RR_T.shape[0], YR_T.shape[1]), name)
            weights = np.linalg.solve(
			    RR_T + regularization * np.eye(RR_T.shape[0]),
			    YR_T + regularization * np.eye(RR_T.shape[0]) @ prior_guess
			    )
            return weights    

    elif(prior_guess is None and isinstance(regularization, np.ndarray)):
        def inner(
            RR_T:      np.ndarray,
            YR_T:      np.ndarray
        ):
            name = "regularization"
            utils.check_shape(regularization.shape, RR_T.shape, name)
            weights = np.linalg.solve(
			    RR_T + regularization,
			    YR_T
			    )
            return weights
    
    elif(isinstance(prior_guess, np.ndarray) and isinstance(regularization, np.ndarray)):
        def inner(
            RR_T:      np.ndarray,
            YR_T:      np.ndarray
        ):
            name = "regularization"
            utils.check_shape(regularization.shape, RR_T.shape, name)
            name = "prior guess"
            utils.check_shape(prior_guess.shape,
							  (regularization.shape[0], YR_T.shape[1]), name)
            weights = np.linalg.solve(
			    RR_T + regularization,
			    YR_T + regularization @ prior_guess
			    )
            return weights
        
    return inner