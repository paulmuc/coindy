import sympy as sym

import coindy.utils.ito_utils as it

# TODO : Finish test implementations


def test_ito_sde_form():
    x = sym.symbols('x0:%d' % 2)

    equations = {'M': sym.Matrix([[sym.sympify('m')]]),
                 'C': sym.Matrix([[sym.sympify('c')]]),
                 'K': sym.Matrix([[sym.sympify('alpha*x0^2-k')]]),
                 'f': sym.Matrix([[0]]),
                 'B_init': sym.Matrix([[0], [sym.sympify('rho*x0')]]),
                 'x': x}

    a, B = it.ito_sde_form(**equations)
    a_true = sym.Matrix([
        [sym.sympify('x1')],
        [sym.sympify('(-alpha * x0 ** 3 - c * x1 + k * x0) / m')]])
    B_true = sym.Matrix([
        [sym.sympify(0)],
        [sym.sympify("rho * x0 / m")]])
    assert a == a_true and B == B_true
