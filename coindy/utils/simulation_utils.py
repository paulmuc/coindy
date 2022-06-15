"""
Created on 6 avr. 2021

@author: Paul
"""

import numpy as np
import sympy as sym

from coindy.utils.matrix_utils import remove_vars


def generate_wiener_increment(time_step: float, time_stop: int, n_dof: int) -> np.ndarray:
    """Returns a n_dof-by-(time_stop/time_step) numpy array of Wiener increments

    :param n_dof: Number of degrees of freedom (number of realization of a Wiener increment to produce)
    :param time_stop: Simulation duration in seconds
    :param time_step: Simulation time step in seconds
    :return: Matrix of Wiener increments
    :Example:
        generate_wiener_increment(0.01, 100, 2)
    """
    dt = time_step

    # Matrix for correct stochastic input variance
    del_mat = np.array([[np.sqrt(dt), 0], [(dt ** 1.5) / 2, (dt ** 1.5) / (2 * np.sqrt(3))]])

    # Initialize matrices
    Nt = round(time_stop / dt)
    dW = np.zeros((2, n_dof, Nt))

    for n in range(0, Nt):
        # Wiener terms
        for i in range(0, n_dof):
            tmp = np.matmul(del_mat, np.random.randn(2, 1))
            dW[0, i, n] = tmp[0]
            dW[1, i, n] = tmp[1]
    return dW


def constant_substitution(term, x, constant_map: dict):
    """
    Replaces symbolic variables in a symbolic expressions with instructed corresponding constants

    :param term: Symbolic expression
    :param x: sym.Symbol or tuple of sym.Symbol types representing the symbols one wishes to replace with the correspon-
    ding constants in constants_map
    :param constant_map:
    :return: Term where the symbolic variables have been replaced by the corresponding constants
    """
    sym_var = term.atoms(sym.Symbol)
    sym_var = remove_vars(sym_var, x)
    var_const_values = list()
    for var in sym_var:
        var_const_values.append((var, constant_map[var]))
    term = term.subs(var_const_values)
    return term
