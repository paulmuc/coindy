from typing import Union

import collections
import numpy as np
import sympy as sym

from coindy import ProgressWorker
from coindy import matrix_utils as mu, ito_utils as it, simulation_utils as su

__all__ = ['SDEModel', 'ModelError']


class SDEModel(ProgressWorker):

    def __init__(self, n_dof: int, n_rvs: int, algorithm='all'):
        """ Constructor of SDEModel

        Initializes the class
        :param n_dof: Integer indicating the number of degrees of freedom
        :param n_rvs: Integer indicating the number of random variables
        """
        super(SDEModel, self).__init__()
        self._algorithm = None
        self._algorithm_changed = False
        self.n_dof = n_dof
        self.n_rvs = n_rvs
        self._equations = {}
        self._deterministic_input = None
        self._states = sym.symbols('x0:%d' % int(2 * n_dof))
        self._sigma = sym.symbols('s0:%d' % int(n_rvs))
        self._t = sym.Symbol('t')
        self.sde_terms = {}
        self.reference_keys = [{'M', 'C', 'K', 'f'}, {'a', 'B'}]
        self.compute_map = []
        self.variables = {'x': self._states, 't': self._t}
        self._constants = []
        self._latex_equations = {}
        self._simulator = None
        self.algorithm = algorithm

    def compute_ito_sde_terms(self):
        u""" Compute the terms needed for the simulation of the system described in self._equations
         through the Euler-Maruyama, Milstein, It\u014d-Taylor 1.5 updates
        """
        progress = 0

        # If method is called before self.equations
        if not bool(self._equations):
            raise TypeError('Equations not set')

        x = self._states
        t = self.variables['t']

        # Choose between MCK or AB mode
        if len(self._equations.values()) == 6:
            a, B = it.ito_sde_form(**self._equations)
        else:
            a = self._equations['a']
            B = self._equations['B']

        progress = self.check_progress(progress, 25, 'Computing SDE and It\u014d-Taylor terms...')

        self.sde_terms['a'] = a + self._deterministic_input
        self.sde_terms['B'] = B

        # Build function-argument list for each the set algorithm
        if self._algorithm == 'em':
            # If EM, only a and B are needed
            self.check_progress(progress, 75)
            return
        else:
            # If MI, add Ljb
            computing_steps = [[it.LJ_total, [B, B, x]]]
            increment = 75

        if self._algorithm == 'it' or self._algorithm == 'all':
            # If IT1.5 or all, add L0a, Lja, L0b, L1L1b
            computing_steps.insert(0, [it.L0, [a, a, B, x, t]])
            computing_steps.insert(1, [it.L0, [B, a, B, x, t]])
            computing_steps.insert(2, [it.LJ_total, [a, B, x]])
            computing_steps.insert(4, [it.L1L1_total, [B, x]])
            increment = 75 / 5

        # Compute each term
        results = list()
        for computing_set in computing_steps:
            results.append(computing_set[0](*computing_set[1]))
            progress = self.check_progress(progress, increment)

        if self._algorithm == 'mi':
            self.sde_terms['Ljb'] = results[0]
        else:
            self.sde_terms['L0a'] = results[0]
            self.sde_terms['L0b'] = results[1]
            self.sde_terms['Lja'] = results[2]
            self.sde_terms['Ljb'] = results[3]
            self.sde_terms['L1L1b'] = results[4]

        if self._algorithm_changed:
            self._algorithm_changed = False

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
    def equations(self, equations_str: dict):
        """ Processing of equations into a sympy.Matrix format for later computation
        :param equations_str: Dictionary of 2D nested lists representing the equations of motion of a system. The
        equations can be input in the {'M': M, 'C': C, 'K': K, 'f': f} format or in the {'a': a, 'B': B} format
        """
        # Check if the proper keys were given
        if len(equations_str.values()) == 4:
            for key in equations_str.keys():
                if key not in self.reference_keys[0]:
                    raise ValueError('The given dict does not contain the proper keys')
        elif len(equations_str.values()) == 2:
            for key in equations_str.keys():
                if key not in self.reference_keys[1]:
                    raise ValueError('The given dict does not contain the proper keys')
        else:
            raise ValueError('The input must be a dict with 4 or 2 elements')

        # Sympify matrices
        for item in equations_str.items():
            self._equations[item[0]] = sym.Matrix(sym.sympify(item[1]))

        # Collect symbols
        symbols_tmp = list()
        for matrix in self._equations.values():
            symbols_tmp.extend(list(matrix.atoms(sym.Symbol)))

        # Remove duplicates
        symbols_tmp = mu.unique(symbols_tmp)

        # Remove states and time from symbols
        symbols_tmp = mu.remove_vars(symbols_tmp, self._states)
        symbols_tmp = mu.remove_vars(symbols_tmp, self._t)
        self._constants = list(symbols_tmp)
        if len(equations_str.values()) == 4:
            # Augment excitation matrix and extract force vectors
            self._equations['B_init'] = mu.augment_vectors(self._equations['f'][:, 2:], 2 * self.n_dof)
            self._deterministic_input = mu.augment_vector(self._equations['f'][:, 1], 2 * self.n_dof)
            self._equations['f'] = self._equations['f'][:, 0]
        elif len(equations_str.values()) == 2:
            self._deterministic_input = sym.Matrix([[0]] * self._equations['a'].shape[0])

        self._equations['x'] = self._states

    def simulate(self, time, constants, initial_values):
        if self._algorithm_changed:
            raise ModelError('The algorithm setting was changed without re-running compute_ito_sde_form')

        if SDEModel.show_progress:
            SDESimulator.show_progress = True
        self._simulator = SDESimulator(self, time, constants, initial_values, algorithm=self._algorithm)
        self._simulator.simulate()

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

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: Union[list[str], str]):
        # Validation of inputs
        if isinstance(algorithm, list):
            if collections.Counter(algorithm) == collections.Counter(SDESimulator.computing_map):
                algorithm = 'all'
            else:
                for technique in algorithm:
                    if technique not in SDESimulator.computing_map:
                        raise KeyError('Wrong key used, only "em", "mi" or "it" are allowed')
        elif isinstance(algorithm, str):
            if algorithm not in ['em', 'mi', 'it', 'all']:
                raise KeyError('Wrong key used, only "em", "mi", "it" or "all" are allowed')
        else:
            raise TypeError('The input must be a list of strings or a string')
        if self._algorithm != algorithm and self._algorithm is not None:
            self._algorithm_changed = True
        self._algorithm = algorithm

    @property
    def results(self):
        return self._simulator.results


class SDESimulator(ProgressWorker):
    computing_map = ['em', 'mi', 'it']

    def __init__(self, sde_model: SDEModel, time: list, constants: dict, initial_values: dict, algorithm='all'):
        u"""Constructor for SDESimulator

        :param sde_model: SDEModel containing the It\u014d-Taylor terms as computed by compute_ito_sde_terms()
        :param time: list with 2 entries where the first is the sampling period and the second the total time of integration.
        :param constants: Dictionary of constants and corresponding values
        :param initial_values: Dictionary of constants and corresponding values
        :param algorithm: String or list indicating which integration technique to use
        """
        super(SDESimulator, self).__init__()
        self._algorithms = None
        self.sde_model = sde_model
        self.computing_map = {'em':
                                  {'function': _euler_maruyama_iteration, 'failed_flag': True,
                                   'compute_flag': False},
                              'mi':
                                  {'function': _milstein_iteration, 'failed_flag': True,
                                   'compute_flag': False},
                              'it':
                                  {'function': _ito_taylor_iteration, 'failed_flag': True,
                                   'compute_flag': False}}
        self._results = []
        self.time = time
        self.constants = constants
        self.initial_values = initial_values
        self.algorithm = algorithm

    def simulate(self):
        """Performs a simulation of a system under random excitation using the data contained in self.sde_model
        """

        self._results.clear()

        terms = self.sde_model.sde_terms
        x = self.sde_model.states
        t = self.sde_model.t
        n_dof = self.sde_model.n_dof

        update_variables = x + (t,)

        lambda_terms = {}

        for key, term in terms.items():
            # Substitute constants into equations
            terms[key] = su.constant_substitution(term, update_variables, self.constants)

            # Lambdify the terms for use in recursive sim
            if key == 'a' or key == 'L0a' or key == 'L0b':
                # Transpose if term is a vector
                term_to_lambdify = terms[key].transpose()
            else:
                term_to_lambdify = terms[key]

            lambda_terms[key] = sym.lambdify(update_variables, term_to_lambdify)

        # Simulation
        # Stochastic increment terms
        dt = self.time[0]
        T = self.time[1]
        dw = su.generate_wiener_increment(dt, T, n_dof)
        Nt = int(T / dt)
        t_vector = np.linspace(0, T, num=Nt)

        self._results = {'time': t_vector}
        # Initialization of data arrays
        for technique in self.computing_map:
            if self.computing_map[technique]['compute_flag']:
                self.computing_map[technique]['data'] = np.zeros((2 * n_dof, Nt))
                self.computing_map[technique]['data'][:, 0] = np.array(list(self.initial_values.values()))
                # Enter the appropriate data pointer into the results
                self._results[technique] = self.computing_map[technique]['data']

        progress = self.check_progress(0, 25, 'Computing simulation...')

        for n in range(0, Nt - 1):

            sub_dw = dw[:, :, n]

            if n % int(Nt * 0.025) == 0:
                progress = self.check_progress(progress, 75 * 0.025)

            # technique holds the data as to which processing algorithm to use with which data matrix and flag
            for technique in self.computing_map:
                if self.computing_map[technique]['failed_flag'] and self.computing_map[technique]['compute_flag']:

                    # Processing using either Euler-Maruyama 0.5, Milstein 1.0 or Ito-Taylor 1.5
                    output = self.computing_map[technique]['function'](lambda_terms,
                                                                       self.computing_map[technique]['data'][:, n], dt,
                                                                       t_vector[n], n_dof, sub_dw)
                    if isinstance(output, Exception):
                        self.computing_map[technique]['failed_flag'] = False
                    else:
                        self.computing_map[technique]['data'][:, n + 1] = output

            # Add failed_flags (False if corresponding technique has failed)
            self._results['failed_flags'] = {technique: self.computing_map[technique]['failed_flag'] for technique in
                                             self.computing_map.keys() if self.computing_map[technique]['compute_flag']}

    @property
    def results(self):
        return self._results

    @property
    def constants(self):
        return self._constants

    @constants.setter
    def constants(self, constants: dict):
        """Handles the parsing of constants symbols into sympy Symbols

        :param constants: Dictionary of constants and their values
        """
        self._constants = {}
        if not isinstance(list(constants.keys())[0], sym.Symbol):
            buff = list(constants.keys())
            for symbol in buff:
                self._constants[sym.sympify(symbol)] = constants.pop(symbol)
            del buff
        else:
            self._constants = constants

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: Union[list[str], str]):
        # Validation of inputs
        if isinstance(algorithm, list):
            for technique in algorithm:
                if technique in SDESimulator.computing_map:
                    self.computing_map[technique]['compute_flag'] = True
        elif isinstance(algorithm, str):
            if algorithm in ['em', 'mi', 'it', 'all']:
                if algorithm == 'all':
                    for technique in SDESimulator.computing_map:
                        self.computing_map[technique]['compute_flag'] = True
                else:
                    self.computing_map[algorithm]['compute_flag'] = True
        self._algorithm = algorithm


class ModelError(Exception):
    pass


# Private functions
def _euler_maruyama_iteration(terms: dict, X, dt: float, t: float, n_dof: int, sub_dW):
    """ Performs an Euler-Maruyama update

    This function can be used in a recursive manner to compute an update of the It\u014d-Taylor terms contained in
    terms using the Euler-Maruyama update rule

    :param terms: Dictionary containing the SDE terms
    :param X: Response matrix
    :param dt: Sampling period
    :param t: Total time in seconds
    :param n_dof: Number of degrees of freedom
    :param sub_dW: Wiener increments
    :return: Updated X matrix
    """
    dW0 = sub_dW[0, :].transpose()

    # Handle exception for overflow
    if any(np.isnan(X)) or any(np.isinf(X)):
        return Exception('Nan or Inf encountered')
    else:
        try:
            a_eval = terms['a'](*X, t)
            B_eval = terms['B'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)).reshape(2 * n_dof, )  # Euler-Maruyama
        except Exception or Warning as error:
            return error


def _milstein_iteration(terms: dict, X, dt: float, t: float, n_dof: int, sub_dW):
    """ Performs a Milstein update

    This function can be used in a recursive manner to compute an update of the It\u014d-Taylor terms contained in
    terms using the Milstein update rule

    :param terms: Dictionary containing the SDE terms
    :param X: Response matrix
    :param dt: Sampling period
    :param t: Total time in seconds
    :param n_dof: Number of degrees of freedom
    :param sub_dW: Wiener increments
    :return: Updated X matrix
    """
    dW0 = sub_dW[0, :].transpose()
    dW2_dt = dW0 ** 2 - dt

    # Handle exception for overflow
    if any(np.isnan(X)) or any(np.isinf(X)):
        return Exception('Nan or Inf encountered')
    else:
        try:
            a_eval = terms['a'](*X, t)
            B_eval = terms['B'](*X, t)

            # Kolmogorov terms
            Ljb_eval = terms['Ljb'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)
                    + np.matmul(0.5 * Ljb_eval, dW2_dt)).reshape(2 * n_dof, )  # Milstein 1.0
        except Exception or Warning as error:
            return error


def _ito_taylor_iteration(terms, X, dt, t, n_dof, sub_dW):
    """ Performs an It\u014d-Taylor 1.5 update

    This function can be used in a recursive manner to compute an update of the It\u014d-Taylor terms contained in
    terms using the It\u014d-Taylor 1.5 update rule

    :param terms: Dictionary containing the SDE terms
    :param X: Response matrix
    :param dt: Sampling period
    :param t: Total time in seconds
    :param n_dof: Number of degrees of freedom
    :param sub_dW: Wiener increments
    :return: Updated X matrix
    """
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
            a_eval = terms['a'](*X, t)
            B_eval = terms['B'](*X, t)

            # Kolmogorov terms
            L0a_eval = terms['L0a'](*X, t).reshape(2 * n_dof, )
            L0b_eval = terms['L0b'](*X, t).transpose()
            Lja_eval = terms['Lja'](*X, t)
            Ljb_eval = terms['Ljb'](*X, t)
            L1L1b_eval = terms['L1L1b'](*X, t)

            return (X + a_eval * dt + np.matmul(B_eval, dW0)
                    + np.matmul(Lja_eval, dZ)
                    + 0.5 * L0a_eval * dt ** 2
                    + np.matmul(0.5 * Ljb_eval, dW2_dt)
                    + np.matmul(L0b_eval, dWdt_dz)
                    + 1 / 6 * np.matmul(L1L1b_eval, dW2_dtdW)).reshape(2 * n_dof, )  # Ito-Taylor 1.5
        except Exception or Warning as error:
            return error