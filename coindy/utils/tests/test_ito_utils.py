import sympy as sym

import coindy.utils.ito_utils as it


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


def test_dX():
    x = sym.symbols('x0:%d' % 2)
    matrix = sym.Matrix(sym.sympify([["x0+x1"], ["x0*x1"], ["x1**2"]]))

    assert it.dX(matrix, x) == sym.Matrix(sym.sympify([["1", "1"], ["x1", "x0"], ["0", "2*x1"]]))


def test_ddX():
    x = sym.symbols('x0:%d' % 2)
    matrix = sym.Matrix(sym.sympify([["x0+x1"], ["x0*x1**2"]]))
    B = sym.Matrix(sym.sympify([["x0", "1"], ["2", "x1"]]))

    assert it.ddX(matrix, x, B) == sym.Matrix(sym.sympify([["0"],
                                                           ["2 * x0 * x1 ** 2 + 8 * x0 * x1 + 8 * x0 + 4 * x1 ** 2"]]))


def test_L0():
    x = sym.symbols('x0:%d' % 2)
    t = sym.Symbol('t')
    matrix = sym.Matrix(sym.sympify([["x0+x1"], ["x0*x1**2"]]))
    B = sym.Matrix(sym.sympify([["x0", "1"], ["2", "x1"]]))

    assert it.L0(matrix, matrix, B, x, t) == sym.Matrix(sym.sympify([
        ["x0 * x1 ** 2 + x0 + x1"],
        [
            "2 * x0 ** 2 * x1 ** 3 + 1.0 * x0 * x1 ** 2 + 4.0 * x0 * x1 + 4.0 * x0 + x1 ** 2 * (x0 + x1) + 2.0 * x1 ** 2"]]))


def test_LJ():
    x = sym.symbols('x0:%d' % 2)
    matrix = sym.Matrix(sym.sympify([["x0+x1"], ["x0*x1**2"]]))
    B = sym.Matrix(sym.sympify([["x0", "1"], ["2", "x1"]]))

    assert it.LJ(matrix, B, x, 0) == sym.Matrix(sym.sympify([
        ["x0 + 2"],
        ["x0*x1*(x1 + 4)"]]))


def test_LJ_total():
    x = sym.symbols('x0:%d' % 2)
    matrix = sym.Matrix(sym.sympify([["x0+x1"], ["x0*x1**2"]]))
    B = sym.Matrix(sym.sympify([["x0", "1"], ["2", "x1"]]))

    assert it.LJ_total(matrix, B, x) == sym.Matrix(sym.sympify([
        ["x0 + 2", "x1 + 1"],
        ["x0 * x1 * (x1 + 4)", "x1 ** 2 * (2 * x0 + 1)"]]))


def test_L1L1_total():
    x = sym.symbols('x0:%d' % 2)
    B = sym.Matrix(sym.sympify([["x0", "1"], ["2", "x1"]]))

    assert it.L1L1_total(B, x) == sym.Matrix(sym.sympify([
        ["x0", "0"],
        ["0", "x1"]]))
