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
def _gram_schmidt(vectors):
    basis = np.zeros(vectors.shape)
    for i in range(vectors.shape[0]):
        v = vectors[i]
        w = np.copy(v)
        for j in range(i):
            b = basis[j]
            w += -np.dot(v, b)*b
        basis[i] = w/np.linalg.norm(w)
    return basis

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lyapunov_spectrum(
    esn:            rescompy.ESN,
    train_result:   rescompy.TrainResult,
    num_exponents:  int                   = 1,
    initial_pert:   float                 = 1e-9,
    tolerance:      float                 = 1e-6,
    tau:            float                 = 1,
    max_iterations: int                   = 10000,
    ):
        
    u = np.zeros((tau, esn.input_dimension))
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
    
    lambdas = []
    states0 = [train_result.states[-1]]
    perturbations = np.random.normal(size=(num_exponents,
                                           esn.size))
    
    states_pert_all = np.zeros((max_iterations, num_exponents+1,
                                tau+1, esn.size))
    error = tolerance
    iterations = 0
    while error >= tolerance and iterations < max_iterations:
        iterations += 1
        perturbations = _gram_schmidt(perturbations)
        perturbations *= initial_pert
        
        
        r = states0[-1]
        v = feature(r, None) @ W
        r = r[None].repeat(tau+1, axis=0)
        v = v[None].repeat(tau+1, axis=0)
        states0, _ = rescompy._get_states_autonomous_jit(u, v, r, feature,
                                                     mapper, A_data,
                                                     A_indices,
                                                     A_indptr, A_shape,
                                                     B, C, W, leakage)
        statespert = np.zeros((num_exponents, tau, esn.size))
        for i in range(num_exponents):
           
            r = r[0] + perturbations[i]
            v = feature(r, None) @ W
            r = r[None].repeat(tau+1, axis=0)
            v = v[None].repeat(tau+1, axis=0)
            statespert[i], _ = rescompy._get_states_autonomous_jit(u, v, r, feature,
                                                         mapper, A_data,
                                                         A_indices,
                                                         A_indptr, A_shape,
                                                         B, C, W, leakage)
            states_pert_all[iterations-1, i+1, 0] = r[0]
        states_pert_all[iterations-1, 0, 0] = r[0] - perturbations[i]
        states_pert_all[iterations-1, 0, 1:] = states0
        states_pert_all[iterations-1, 1:, 1:] = statespert
        lambdas_i = np.zeros((num_exponents))
        for i in range(num_exponents):
            lambdas_i[i] = np.log(np.linalg.norm(states0[-1]
                                                 - statespert[i, -1]))
            lambdas_i[i] += -np.log(np.linalg.norm(perturbations[i]))
            lambdas_i[i] *= 1/(tau-1)
        lambdas.append(lambdas_i)
 
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lyapunov_spectrum_old(
    esn:              rescompy.ESN,
    train_result:     Union[rescompy.TrainResult, np.ndarray],
    ):
        
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
    
    lambda1 = np.zeros((integration_length))
    z = np.zeros((integration_length))
    z0 = np.random.normal(size=(esn.size))
    z0 *= initial_pert/np.sqrt(np.mean(np.square(z0)))
    for i in range(integration_length-1):
        z[i+1] = z[i] + 0.

@numba.jit(nopython=True, fastmath=True)
def _calculate_perturbation(
    feature:        Callable,
    mapper:         Callable,
    A_data:         np.ndarray,
    A_indices:      np.ndarray,
    A_indptr:       np.ndarray,
    A_shape:        tuple,
    B:              np.ndarray,
    C:              np.ndarray,
    W:              np.ndarray,
    leakage:        float,
    state0:         np.ndarray,
    max_iterations: int,
    perturbation:   np.ndarray,
    initial_pert:   float,
    x:              int,
    ):
    
    d = np.zeros((max_iterations), dtype=np.float64)
    u = np.zeros((x, 1), dtype=np.float64)
    states0 = np.zeros((x, A_shape[0]), dtype=np.float64)
    states0[-1] = state0
    for iteration in range(max_iterations):
        perturbation *= initial_pert/np.sqrt(np.mean(np.square(
                                             perturbation)))
        
        r = np.zeros((x, A_shape[0]), dtype=np.float64)
        r[0] = states0[-1]
        v = np.zeros((x, W.shape[1]))
        v[0] = feature(r[0], None) @ W
        
        states0, _ = rescompy._get_states_autonomous_jit(u, v, r, feature,
                                                     mapper, A_data,
                                                     A_indices,
                                                     A_indptr, A_shape,
                                                     B, C, W, leakage)
        r = np.zeros((x, A_shape[0]), dtype=np.float64) + perturbation
        r[0] = states0[-1]
        v = np.zeros((x, W.shape[1]))
        v[0] = feature(r[0], None) @ W
        states1, _ = rescompy._get_states_autonomous_jit(u, v, r, feature,
                                                     mapper, A_data,
                                                     A_indices,
                                                     A_indptr, A_shape,
                                                     B, C, W, leakage)
        d[iteration] = np.log(np.sqrt(np.mean(np.square(states0[-1]
                       - states1[-1])))) - np.log(initial_pert)
        
        d[iteration] *= 1/x
        
    return 1/np.mean(d)


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