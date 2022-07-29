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
from typing import Optional
from pydantic import validate_arguments


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