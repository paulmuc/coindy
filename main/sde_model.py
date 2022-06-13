"""
Created on 8 Apr 2021

@author: Paul Mucchielli
"""
from PyQt5.QtCore import pyqtSlot
import sympy as sym
from console_output import progress_bar

from ito_taylor_utilities import ito_sde_form, L0, LJ
from simulation_utilities import remove_vars
from matrix_utilities import unique
from progress_worker import ProgressWorker


def _augment_vector(vector, n_dof):
    augmented_vec = sym.zeros(n_dof, 1)
    for i in range(1, n_dof, 2):
        augmented_vec[i] = vector[int((i - 1) / 2)]
    return augmented_vec


def _augment_vectors(vectors, n_dof):
    augmented_vecs = sym.zeros(n_dof, vectors.shape[1])
    for i in range(0, vectors.shape[1]):
        augmented_vecs[:, i] = _augment_vector(vectors[:, i], n_dof)
    return augmented_vecs


class SDEModel(ProgressWorker):

    def __init__(self, n_dof, n_rvs, parent=None):
        super(SDEModel, self).__init__()
        self.parent = parent
        self.n_dof = n_dof
        self.n_rvs = n_rvs
        self._equations = {}
        self._deterministic_input = None
        self._states = sym.symbols('x0:%d' % int(2 * n_dof))
        self._sigma = sym.symbols('s0:%d' % int(n_rvs))
        self._dB = sym.symbols('dB0:%d' % int(n_rvs))
        self._t = sym.Symbol('t')
        self.sde_terms = {'a': None, 'B': None, 'L0a': None, 'L0b': None, 'Lja': None, 'Ljb': None,
                          'L1L1b': None}
        self.compute_map = []
        self.variables = {'x': self._states, 't': self._t}
        self._constants = []
        self._latex_equations = {}

    @pyqtSlot()
    def compute_ito_sde_terms(self):
        progress = 100/4
        increment = 100/4

        if not bool(self._equations):
            raise TypeError('Equations not set')

        a, B = ito_sde_form(self._states, **self._equations)

        if not self.check_signals(progress):
            return

        if self.__class__.show_progress:
            print("Computing SDE and It\u014d-Taylor terms...")
            progress_bar(progress, 100)

        progress = progress + increment

        x = self._states
        t = self.variables['t']

        l0a = L0(a, a, B, x, t)  # First Kolmogorov operator result applied to a
        if not self.check_signals(progress):
            return

        if self.__class__.show_progress:
            progress_bar(progress, 100)

        progress = progress + increment

        l0b = L0(B, a, B, x, t)  # First Kolmogorov operator result applied to b
        if not self.check_signals(progress):
            return

        if self.__class__.show_progress:
            progress_bar(progress, 100)

        progress = progress + increment

        lja = sym.zeros(2 * self.n_dof, self.n_rvs)  # Second Kolmogorov operator result applied to a
        ljb = sym.zeros(2 * self.n_dof, self.n_rvs)  # Second Kolmogorov operator result applied to b
        l1l1b = sym.zeros(2 * self.n_dof, self.n_rvs)
        for i in range(0, self.n_rvs):
            lja[:, i] = LJ(a, B, x, i)
            ljb[:, i] = LJ(B[:, i], B, x, i)
            l1l1b[:, i] = LJ(LJ(B[:, i], B, x, i), B, x, i)

        if not self.check_signals(progress):
            return

        if self.__class__.show_progress:
            progress_bar(progress, 100)
            print("\r")

        progress = progress + increment

        self.sde_terms['a'] = a + self._deterministic_input
        self.sde_terms['B'] = B
        self.sde_terms['L0a'] = l0a
        self.sde_terms['L0b'] = l0b
        self.sde_terms['Lja'] = lja
        self.sde_terms['Ljb'] = ljb
        self.sde_terms['L1L1b'] = l1l1b

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
        symbols_tmp = unique(symbols_tmp)
        symbols_tmp = remove_vars(symbols_tmp, self._states)
        symbols_tmp = remove_vars(symbols_tmp, self._dB)
        symbols_tmp = remove_vars(symbols_tmp, self._t)
        self._constants = list(symbols_tmp)
        self._equations['B_init'] = _augment_vectors(self._equations['F'][:, 2:], 2 * self.n_dof)
        self._deterministic_input = _augment_vector(self._equations['F'][:, 1], 2 * self.n_dof)
        self._equations['F'] = self._equations['F'][:, 0]

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
    def dB(self):
        return self._dB

    @property
    def t(self):
        return self._t
