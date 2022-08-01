"""analysis.py

The analysis sub for rescompy.

Performs various dynamical analyses on rescompy objects.
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
import rescompy
from .utils import utils


@numba.jit(nopython=True, fastmath=True)
def _gram_schmidt(vectors: np.ndarray):
    basis = np.zeros(vectors.shape)
    for i in range(vectors.shape[0]):
        v = vectors[i]
        w = np.copy(v)
        for j in range(i):
            b = basis[j]
            w += -np.dot(v, b)*b
        basis[i] = w/np.linalg.norm(w)
    return basis


@numba.jit(nopython=True, fastmath=True)
def _get_lyapunov_spectrum(
    f:              Callable,
    x0:             np.ndarray,
    d0:             np.ndarray,
    num_exponents:  int,
    initial_pert:   float,
    num_iterations: int,
    ):
    
    x = np.copy(x0)
    d = np.copy(d0)
    
    lambdas = np.zeros((num_exponents, num_iterations))
    for i in range(num_iterations):
        d = _gram_schmidt(d)
        d *= initial_pert
        x_n = f(x)
        for e in range(num_exponents):
            x_i = f(x + d[e])
            d[e] = x_i - x_n
            lambdas[e, i] = np.linalg.norm(x_i - x_n)/initial_pert - 1
        x = x_n
        
    return lambdas


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lyapunov_spectrum(
    esn:            rescompy.ESN,
    train_result:   rescompy.TrainResult,
    num_exponents:  int                   = 1,
    initial_pert:   float                 = 1e-9,
    num_iterations: int                   = 10000,
    ):
    
    # TODO: docstring
    # TODO: comments
    # TODO: range validation
    # TODO: testing
    # NOTE: assumes all of the inputs are mapped as feedback (default mapper)
    # NOTE: this also means no input is assumed.
        
    u = np.zeros((1, esn.input_dimension))
    feature = train_result.feature_function
    @numba.jit(nopython=True, fastmath=True)
    def mapper(inputs, outputs):
        return outputs
    A_data = esn.A.data
    A_indices = esn.A.indices
    A_indptr = esn.A.indptr
    A_shape = esn.A.shape
    B = esn.B
    C = esn.C
    W = train_result.weights
    leakage = esn.leaking_rate
    res_dim, output_dim = W.shape

    @numba.jit(nopython=True, fastmath=True)
    def f(x):
        v = np.zeros((2, output_dim))
        r = np.zeros((2, res_dim))
        
        v[0] = feature(x, None) @ W
        r[0] = x
        states0, _ = rescompy._get_states_autonomous_jit(u, v, r,
                         feature, mapper, A_data, A_indices, A_indptr,
                         A_shape, B, C, W, leakage)
        return states0[-1]
    
    x0 = train_result.states[-1]
    d0 = np.random.normal(size=(num_exponents, res_dim))
    
    lambdas = _get_lyapunov_spectrum(f, x0, d0, num_exponents,
                                     initial_pert, num_iterations)
    
    return np.mean(lambdas, axis=1)


@numba.jit(nopython=True, fastmath=True)
def esn_jacobian(
    r:              np.ndarray,
    feature:        Callable,
    feature_deriv:  Callable,
    A:              np.ndarray,
    A_data:         np.ndarray,
    A_indices:      np.ndarray,
    A_indptr:       np.ndarray,
    A_shape:        tuple,
    B:              np.ndarray,
    C:              np.ndarray,
    W:              np.ndarray,
    leakage:        float,
    ):
    
    inner_term1 = rescompy._mult_vec(A_data, A_indices, A_indptr,
                                     A_shape, r)
    inner_term2 = B @ (feature(r, None) @ W)
    inner_term3 = C
    inner = inner_term1 + inner_term2 + inner_term3
    outter = (A.T + B @ W.T).T # All of these transposes are suspect
    
    return (1/leakage)*(outter*(1/np.cosh(inner)**2) - np.eye(A_shape[0]))


def esn_rhs(
    r:              np.ndarray,
    feature:        Callable,
    A_data:         np.ndarray,
    A_indices:      np.ndarray,
    A_indptr:       np.ndarray,
    A_shape:        tuple,
    B:              np.ndarray,
    C:              np.ndarray,
    W:              np.ndarray,
    leakage:        float,
    ):
    
    inner_term1 = rescompy._mult_vec(A_data, A_indices, A_indptr,
                                     A_shape, r)
    inner_term2 = B @ (feature(r, None) @ W)
    inner_term3 = C
    inner = inner_term1 + inner_term2 + inner_term3
    
    return (1/leakage)*(-r + np.tanh(inner))


def calc_jac(f, r0, size1, size2, delta=1e-8):
    
    jac = np.zeros((size1, size2))
    for i in range(size1):
        pert = np.zeros((size1))
        pert[i] = delta
        diff = (f(r0+pert) - f(r0-pert))/(2*delta)
        jac[i] = diff
        
    return jac