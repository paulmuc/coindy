"""
Created on 8 Apr 2021

@author: Paul Mucchielli
"""
import collections

from PyQt5.QtCore import pyqtSlot
import sympy as sym

import progress_worker
import matrix_utils as mu
import ito_utils as it


class SDEModel(progress_worker.ProgressWorker):

    def __init__(self, n_dof, n_rvs, parent=None):
        super(SDEModel, self).__init__()
        self.parent = parent
        self.n_dof = n_dof
        self.n_rvs = n_rvs
        self._equations = {}
        self._deterministic_input = None
        self._states = sym.symbols('x0:%d' % int(2 * n_dof))
        self._sigma = sym.symbols('s0:%d' % int(n_rvs))
        self._t = sym.Symbol('t')
        self.sde_terms = {'a': None, 'B': None, 'L0a': None, 'L0b': None, 'Lja': None, 'Ljb': None,
                          'L1L1b': None}
        self.compute_map = []
        self.variables = {'x': self._states, 't': self._t}
        self._constants = []
        self._latex_equations = {}
        self.computing_steps = [[it.ito_sde_form, self._equations]]

    @pyqtSlot()
    def compute_ito_sde_terms(self):
        progress = 100/7
        increment = 100/7

        if not bool(self._equations):
            raise TypeError('Equations not set')

        x = self._states
        t = self.variables['t']

        a, B = it.ito_sde_form(**self._equations)

        progress = self.check_progress(progress, increment, 'Computing SDE and It\u014d-Taylor terms...')

        computing_steps = [[it.L0, [a, a, B, x, t]],
                           [it.L0, [B, a, B, x, t]],
                           [it.LJ_total, [a, B, x]],
                           [it.LJ_total, [B, B, x]],
                           [it.L1L1_total, [B, x]]]

        results = list()
        for computing_set in computing_steps:
            results.append(computing_set[0](*computing_set[1]))
            progress = self.check_progress(progress, increment)

        self.sde_terms['a'] = a + self._deterministic_input
        self.sde_terms['B'] = B
        self.sde_terms['L0a'] = results[0]
        self.sde_terms['L0b'] = results[1]
        self.sde_terms['Lja'] = results[2]
        self.sde_terms['Ljb'] = results[3]
        self.sde_terms['L1L1b'] = results[4]

        self.finished.emit()

    @property
    def sde_terms(self):
        return self._sde_terms

    @sde_terms.setter
    def sde_terms(self, value):
        self._sde_terms = value

    @property
    def equations(self):
        return self._equations

    @equations.setter
    def equations(self, equations_str):
        for item in equations_str.items():
            self._equations[item[0]] = sym.Matrix(sym.sympify(item[1]))
        symbols_tmp = list()
        for matrix in self._equations.values():
            symbols_tmp.extend(list(matrix.atoms(sym.Symbol)))
        symbols_tmp = mu.unique(symbols_tmp)
        symbols_tmp = mu.remove_vars(symbols_tmp, self._states)
        symbols_tmp = mu.remove_vars(symbols_tmp, self._t)
        self._constants = list(symbols_tmp)
        self._equations['B_init'] = mu.augment_vectors(self._equations['f'][:, 2:], 2 * self.n_dof)
        self._deterministic_input = mu.augment_vector(self._equations['f'][:, 1], 2 * self.n_dof)
        self._equations['f'] = self._equations['f'][:, 0]
        self._equations['x'] = self._states

    @property
    def n_dof(self):
        return self._n_dof

    @property
    def latex_equations(self):
        return self._latex_equations.copy()

    @n_dof.setter
    def n_dof(self, value):
        self._n_dof = value

    @property
    def n_rvs(self):
        return self._n_rvs

    @n_rvs.setter
    def n_rvs(self, value):
        self._n_rvs = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        self._variables = value

    @property
    def constants(self):
        buff = self._constants
        constants = []
        for item in buff:
            constants.append(item.name)
        constants.sort()
        return constants

    @constants.setter
    def constants(self, value):
        self._constants = value

    @property
    def states(self):
        return self._states

    @property
    def t(self):
        return self._t
