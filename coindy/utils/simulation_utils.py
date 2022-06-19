import numpy as np
import sympy as sym

import coindy.utils.matrix_utils as mu


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
    :param x: sym.Symbol or tuple of sym.Symbol types representing the symbols one wishes to exclude from replacement
     with the corresponding constants in constants_map
    :param constant_map:
    :return: Term where the symbolic variables have been replaced by the corresponding constants
    """
    sym_var = term.atoms(sym.Symbol)
    sym_var = mu.remove_vars(sym_var, x)
    var_const_values = list()
    for var in sym_var:
        var_const_values.append((var, constant_map[var]))
    term = term.subs(var_const_values)
    return term


def euler_maruyama_iteration(terms: dict, X, dt: float, t: float, n_dof: int, sub_dW):
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


def milstein_iteration(terms: dict, X, dt: float, t: float, n_dof: int, sub_dW):
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


def ito_taylor_iteration(terms, X, dt, t, n_dof, sub_dW):
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
