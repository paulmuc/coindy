"""
Created on 8 Apr 2021

@author: 13721745
"""
import numpy as np
import sympy as sym
from PyQt5.QtCore import pyqtSlot

from simulation_utilities import const_substitution, generate_wiener_increment
from progress_worker import ProgressWorker
from console_output import progress_bar


def euler_maruyama_iteration(terms_struct, X, dt, t, n_dof, sub_dW):
    dW0 = sub_dW[0, :].transpose()

    # Handle exception for overflow
    if any(np.isnan(X)) or any(np.isinf(X)):
        return Exception('Nan or Inf encountered')
    else:
        try:
            a_eval = terms_struct['a'](*X, t)
            B_eval = terms_struct['B'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)).reshape(2 * n_dof, )  # Euler-Maruyama
        except Exception or Warning as error:
            return error


def milstein_iteration(terms_struct, X, dt, t, n_dof, sub_dW):
    dW0 = sub_dW[0, :].transpose()
    dW2_dt = dW0 ** 2 - dt

    # Handle exception for overflow
    if any(np.isnan(X)) or any(np.isinf(X)):
        return Exception('Nan or Inf encountered')
    else:
        try:
            a_eval = terms_struct['a'](*X, t)
            B_eval = terms_struct['B'](*X, t)

            # Kolmogorov terms
            Ljb_eval = terms_struct['Ljb'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)
                    + np.matmul(0.5 * Ljb_eval, dW2_dt)).reshape(2 * n_dof, )  # Milstein 1.0
        except Exception or Warning as error:
            return error


def ito_taylor_iteration(terms_struct, X, dt, t, n_dof, sub_dW):
    dW0 = sub_dW[0, :].transpose()
    dZ = sub_dW[1, :].transpose()
    dW2_dt = dW0 ** 2 - dt
    dWdt_dz = dW0 * dt - dZ
    dW2_dtdW = dW0 ** 2 - dt * dW0

    # Handle exception for overflow
    if any(np.isnan(X)) or any(np.isinf(X)):
        return Exception('Nan or Inf encountered')
    else:
        try:
            a_eval = terms_struct['a'](*X, t)
            B_eval = terms_struct['B'](*X, t)

            # Kolmogorov terms
            L0a_eval = terms_struct['L0a'](*X, t).reshape(2 * n_dof, )
            L0b_eval = terms_struct['L0b'](*X, t).transpose()
            Lja_eval = terms_struct['Lja'](*X, t)
            Ljb_eval = terms_struct['Ljb'](*X, t)
            L1L1b_eval = terms_struct['L1L1b'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)
                    + np.matmul(Lja_eval, dZ)
                    + 0.5 * L0a_eval * dt ** 2
                    + np.matmul(0.5 * Ljb_eval, dW2_dt)
                    + np.matmul(L0b_eval, dWdt_dz)
                    + 1 / 6 * np.matmul(L1L1b_eval, dW2_dtdW)).reshape(2 * n_dof, )  # Ito-Taylor 1.5
        except Exception or Warning as error:
            return error


class SDESimulator(ProgressWorker):

    def __init__(self, sde_model, time, constants, initial_values):
        super(SDESimulator, self).__init__()
        self.sde_model = sde_model
        self.computing_map = {'X': [[], euler_maruyama_iteration, True],
                              'Y': [[], milstein_iteration, True],
                              'Z': [[], ito_taylor_iteration, True]}
        self._results = []
        self.time = time
        self.constants = constants
        self.initial_values = initial_values

    @pyqtSlot()
    def simulate(self):
        self._results.clear()
        if not isinstance(list(self.constants.keys())[0], sym.Symbol):
            buff = list(self.constants.keys())
            for symbol in buff:
                self.constants[sym.sympify(symbol)] = self.constants.pop(symbol)
            del buff
        terms = self.sde_model.sde_terms
        x = self.sde_model.states
        t = self.sde_model.t
        n_dof = self.sde_model.n_dof

        a, B, L0a, L0b, Lja, Ljb, L1L1b = terms.values()

        update_variables = x + (t,)
        a = const_substitution(a, update_variables, self.constants)
        B = const_substitution(B, update_variables, self.constants)
        L0a = const_substitution(L0a, update_variables, self.constants)
        L0b = const_substitution(L0b, update_variables, self.constants)
        Lja = const_substitution(Lja, update_variables, self.constants)
        Ljb = const_substitution(Ljb, update_variables, self.constants)
        L1L1b = const_substitution(L1L1b, update_variables, self.constants)

        a = sym.lambdify(update_variables, a.transpose())
        B = sym.lambdify(update_variables, B)
        L0a = sym.lambdify(update_variables, L0a.transpose())
        L0b = sym.lambdify(update_variables, L0b.transpose())
        Lja = sym.lambdify(update_variables, Lja)
        Ljb = sym.lambdify(update_variables, Ljb)
        L1L1b = sym.lambdify(update_variables, L1L1b)

        terms_struct = {'a': a,
                        'B': B,
                        'L0a': L0a,
                        'L0b': L0b,
                        'Lja': Lja,
                        'Ljb': Ljb,
                        'L1L1b': L1L1b}

        # Simulation
        # Stochastic increment terms
        dt = self.time[0]
        T = self.time[1]
        dw = generate_wiener_increment((0, dt, T), n_dof)
        Nt = int(T / dt)
        t_vector = np.linspace(0, T, num=Nt)
        X = np.zeros((2 * n_dof, Nt))
        X[:, 0] = np.array(list(self.initial_values.values()))
        Y = np.zeros((2 * n_dof, Nt))
        Y[:, 0] = np.array(list(self.initial_values.values()))
        Z = np.zeros((2 * n_dof, Nt))
        Z[:, 0] = np.array(list(self.initial_values.values()))

        self.computing_map['X'][0] = X
        self.computing_map['Y'][0] = Y
        self.computing_map['Z'][0] = Z

        if not self.check_signals(25):
            return

        print("Computing simulation...")
        for n in range(0, Nt - 1):

            sub_dw = dw[:, :, n]

            if n % round(Nt * 0.025) == 0:
                if not self.check_signals(75 * (n + 1) / Nt + 25):
                    return

                if self.__class__.show_progress:
                    progress_bar(75 * (n + 1) / Nt + 25, 100)

            # technique holds the data as to which processing algorithm to use with which data matrix and flag
            for technique in self.computing_map:
                if self.computing_map[technique][2]:

                    # Processing using either Euler-Maruyama 0.5, Milstein 1.0 or Ito-Taylor 1.5
                    output = self.computing_map[technique][1](terms_struct, self.computing_map[technique][0][:, n], dt,
                                                              t_vector[n], n_dof, sub_dw)
                    if isinstance(output, Exception):
                        self.computing_map[technique][2] = False
                    else:
                        self.computing_map[technique][0][:, n + 1] = output

        self._results = [t_vector, np.concatenate((X, Y, Z)),
                         [self.computing_map['X'][2], self.computing_map['Y'][2], self.computing_map['Z'][2]]]
        if not self.check_signals(100):
            return

        if self.__class__.show_progress:
            progress_bar(100, 100)
            print("\r")

        self.finished.emit()

    @property
    def results(self):
        return *self._results,
