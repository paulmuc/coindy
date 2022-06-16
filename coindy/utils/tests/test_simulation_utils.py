import sympy as sym

import coindy.utils.simulation_utils as su


def test_generate_wiener_increment():
    time_step = 0.01
    time_stop = 10
    n_dof = 2

    dW = su.generate_wiener_increment(time_step, time_stop, n_dof)

    assert dW.shape == (n_dof, n_dof, 1000)


def test_constant_substitution():
    term = sym.sympify("a*x")
    a = sym.Symbol("a")
    x = sym.Symbol("x")
    constant_map = {a: 2}

    term = su.constant_substitution(term, x, constant_map)

    assert term == 2*x
