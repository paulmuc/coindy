"""
Created on 6 avr. 2021

@author: Paul
"""

import numpy as np
import sympy as sym

from matrix_utilities import remove_vars


def generate_wiener_increment(time_tuple, n_dof):
    # GENERATEWIENERINCREMENT Generate a time series of dW and dZ increments
    dt = time_tuple[1]
    # Matrix for correct stochastic input variance
    del_mat = np.array([[np.sqrt(dt), 0], [(dt ** 1.5) / 2, (dt ** 1.5) / (2 * np.sqrt(3))]])
    Nt = round(time_tuple[2] / dt)
    dW = np.zeros((2, n_dof, Nt))

    for n in range(0, Nt):
        # Wiener terms
        for i in range(0, n_dof):
            tmp = np.matmul(del_mat, np.random.randn(2, 1))
            dW[0, i, n] = tmp[0]
            dW[1, i, n] = tmp[1]
    return dW


def const_substitution(term, x, const_map):
    sym_var = term.atoms(sym.Symbol)
    sym_var = remove_vars(sym_var, x)
    var_const_values = list()
    for var in sym_var:
        var_const_values.append((var, const_map[var]))
    term = term.subs(var_const_values)
    return term
