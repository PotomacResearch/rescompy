"""system.py

The benchmark time series submodule for rescompy.

This submodule contains methods for generating common benchmark dynamical
systems.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


import numba
import logging
import numpy as np
from typing import Literal, Union
from pydantic import validate_arguments
from numpy.random import default_rng
from numpy.random._generator import Generator
from ..utils import utils


@numba.jit(nopython=True, fastmath=True)
def _lorenz(sigma, beta, rho, x0, integrate_length, h):
    
    x = np.zeros((integrate_length, 3))
    x[0] = x0
    
    def lorenz_deriv(x):
        
        x_prime = np.zeros((3))
        x_prime[0] = sigma*(x[1] - x[0])
        x_prime[1] = x[0]*(rho - x[2]) - x[1]
        x_prime[2] = x[0]*x[1] - beta*x[2]

        return x_prime
    
    for t in range(integrate_length - 1):
        
        k1 = lorenz_deriv(x[t])
        k2 = lorenz_deriv(x[t] + (h/2)*k1)
        k3 = lorenz_deriv(x[t] + (h/2)*k2)
        k4 = lorenz_deriv(x[t] + h*k3)

        x[t+1] = x[t] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_lorenz(
    sigma:            float                                = 10.0,
    beta:             float                                = 8/3,
    rho:              float                                = 28.0,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    ) -> np.ndarray:
    """The Lorenz-getting function.
    
    This function integrates and returns a solution to the Lorenz system,
    obtained with a Runge-Kutta integration scheme.
    
    Args:
        sigma (float): The first Lorenz parameter.
        beta (float): The second Lorenz parameter.
        rho (float): The third Lorenz parameter.
        x0 (np.ndarray): The initial condition of the Lorenz system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    # Check that arguments are in the correct range.
    if x0 != 'random':
        utils.check_range(x0.shape[0], 'x0.shape[0]', 3, 'eq', True)
    utils.check_range(transient_length, 'transient_length', 0, 'geq',
                      True)
    utils.check_range(return_length, 'return_length', 0, 'g', True)
    utils.check_range(h, 'h', 0, 'g', True)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        
        # Locations and scales are according to observed means and standard
        # deviations of the Lorenz attractor with default parameters.
        x0_0 = rng.normal(loc=-0.036, scale=8.236)
        x0_1 = rng.normal(loc=-0.036, scale=9.162)
        x0_2 = rng.normal(loc=25.104, scale=7.663)
        x0 = np.array([x0_0, x0_1, x0_2])

    # Integrate the Lorenz system and return.    
    x = _lorenz(sigma, beta, rho, x0, transient_length + return_length,
                h)
    return x[transient_length:]


@numba.jit(nopython=True, fastmath=True)
def _mackey_glass(gamma, beta, n, tau_h, x0, integrate_length, h):
    
    x = np.zeros((integrate_length, 1))
    x[:tau_h+1] = x0
    
    def mackey_glass_deriv(x, xtau_h):
        
        return beta*xtau_h / (1 + xtau_h**n) - gamma*x
    
    for t in range(tau_h, integrate_length - 1):
        x[t+1] = x[t] + h*mackey_glass_deriv(x[t], x[t-tau_h])
        
    return x


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_mackey_glass(
    gamma:            float                                = 0.1,
    beta:             float                                = 0.2,
    n:                float                                = 10,
    tau:              int                                  = 17,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.1,
    seed:             Union[int, None, Generator]          = None,
    ) -> np.ndarray:
    """The Mackey-Glass-getting function.
    
    This function integrates and returns a solution to the Mackey-Glass system,
    obtained with an Euler integration scheme.
    
    Args:
        gamma (float): The first Mackey-Glass parameter.
        beta (float): The second Mackey-Glass parameter.
        n (float): The third Mackey-Glass parameter.
        tau (int): The fourth Mackey-Glass parameter (in units of h).
        x0 (np.ndarray): The initial condition of the Mackey-Glass system.
                         Must be an array of floats of shape (tau).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 1).
    """
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)

    # Convert tau to units of h.
    tau_h = int(tau/h)
    if tau != tau_h*h:
        msg = f"tau not divisible by h; rounding tau down to {tau_h*h}."
        logging.warning(msg)    
    
    # Check that arguments are in the correct range.
    if x0 != 'random':
        utils.check_range(x0.shape[0], 'x0.shape[0]', tau_h, 'eq', True)
    utils.check_range(transient_length, 'transient_length', 0, 'geq',
                      True)
    utils.check_range(return_length, 'return_length', 0, 'g', True)
    utils.check_range(transient_length + return_length,
                      'transient_length + return_length', tau_h,
                      'g', True)
    utils.check_range(h, 'h', 0, 'g', True)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        
        # Locations and scales are according to observed mean and standard
        # deviation of the Mackey-Glass attractor with default parameters.
        x0 = rng.normal(loc=1.0, scale=0.5, size=(tau_h+1, 1))

    # Integrate the Lorenz system and return.    
    x = _mackey_glass(gamma, beta, n, tau_h, x0,
                      transient_length + return_length, h)
    return x[transient_length:]


@numba.jit(nopython=True, fastmath=True)
def _duffing(delta, alpha, beta, gamma, omega, x0, t0,
             integrate_length, h):
    
    x = np.zeros((integrate_length, 2))
    x[0] = x0
    
    def duffing_deriv(x, t):
        
        x_prime = np.zeros((2))
        x_prime[0] = x[1]
        x_prime[1] = gamma*np.cos(omega*t) - delta*x[1] - alpha*x[0] \
                         - beta*x[0]**3

        return x_prime
    
    for t in range(integrate_length - 1):
        x[t+1] = x[t] + h*duffing_deriv(x[t], t0+t*h)
        
    return x


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_duffing(
    delta:            float                                = 0.3,
    alpha:            float                                = -1.0,
    beta:             float                                = 1.0,
    gamma:            float                                = 0.55,
    omega:            float                                = 1.2,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    t0:               float                                = 0,
    transient_length: int                                  = 10000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    ) -> np.ndarray:
    """The Duffing-getting function.
    
    This function integrates and returns a solution to the Duffing system,
    obtained with an Euler integration scheme.
    
    Args:
        delta (float): The amount of damping.
        alpha (float): The linear stiffness.
        beta (float): The nonlinearity.
        gamma (float): The driving strength.
        omega (float): The driving frequency.
        x0 (np.ndarray): The initial condition of the Duffing system.
                         Must be an array of floats of shape (2,).
        t0 (float): The initial time of the Duffing system.
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 2).
    """
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    # Check that arguments are in the correct range.
    if x0 != 'random':
        utils.check_range(x0.shape[0], 'x0.shape[0]', 2, 'eq', True)
    utils.check_range(transient_length, 'transient_length', 0, 'geq',
                      True)
    utils.check_range(return_length, 'return_length', 0, 'g', True)
    utils.check_range(h, 'h', 0, 'g', True)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        
        # Since the oscillator is driven, a small perturbation from the origin
        # will suffice for a random initial condition.
        x0_0 = rng.normal(loc=0, scale=0.01)
        x0_1 = rng.normal(loc=0, scale=0.01)
        x0 = np.array([x0_0, x0_1])

    # Integrate the Lorenz system and return.    
    x = _duffing(delta, alpha, beta, gamma, omega, x0, t0,
                 transient_length + return_length, h)
    return x[transient_length:]


@numba.jit(nopython=True, fastmath=True)
def _van_der_pol(mu, x0, integrate_length, h):
    
    x = np.zeros((integrate_length, 2))
    x[0] = x0
    
    def van_der_pol_deriv(x):
        x_prime = np.zeros(2)
        x_prime[0] = x[1]
        x_prime[1] = mu*(1 - (x[0]**2)) * x[1] - x[0]
        return x_prime
    
    for t in range(integrate_length - 1):
        x[t+1] = x[t] + h*van_der_pol_deriv(x[t])
        
    return x


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_van_der_pol(
        mu:                 float                               = 10,
        x0:                 Union[Literal['random'], np.ndarray] = 'random',
        transient_length:   int                                 = 5000,
        return_length:      int                                 = 100000,
        h:                  float                               = 0.01,
        seed:               Union[int, None, Generator]         = None    
        ) -> np.ndarray:
    """The Van-der-Pol-getting function.
    
    This function integrates and returns a solution to the Van der Pol system,
    obtained with an Euler integration scheme.
    
    Args:
        mu (flaot): The damping strength.
        x0 (np.ndarray): The initial condition of the Van der Pol system.
                         Must be an array of floats of shape (2,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 2).
    """
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    # Check that arguments are in the correct range.
    if x0 != 'random':
        utils.check_range(x0.shape[0], 'x0.shape[0]', 2, 'eq', True)
    utils.check_range(transient_length, 'transient_length', 0, 'geq',
                      True)
    utils.check_range(return_length, 'return_length', 0, 'g', True)
    utils.check_range(h, 'h', 0, 'g', True)

    # Create a random intiail condition, if applicable.
    if x0 == 'random':
        
        # Locations and scales are according to observed means and standard
        # deviations of the Van der Pol attractor with default parameters.
        x0_0 = rng.normal(loc=0, scale=1.67)
        x0_1 = rng.normal(loc=0, scale=1.58)
        x0 = np.array([x0_0, x0_1])
    
    # Integrate the system and return.
    x = _van_der_pol(mu, x0, transient_length + return_length, h)
    
    return x[transient_length:]