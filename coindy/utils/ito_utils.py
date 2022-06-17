import sympy as sym
import numpy as np


def ito_sde_form(x, M, C, K, f, B_init):
    u"""
    Derives the It\u014d SDE form of the system formulated as Mx\u0308 + Cx\u0307 + Kx = F + B_init w\u0307 into
     adt + Bdwt
    :param x: Vector of state variables
    :param M: Mass matrix, diagonal elements must be non-zero
    :param C: Damping matrix
    :param K: Stiffness matrix
    :param f: Force vector
    :param B_init: Excitation matrix
    :return: a - Drift vector
             B - Diffusion matrix
    """
    n = M.shape[0]
    M_inverse = sym.simplify(M.inv('ADJ'))

    # Augmenting the inverse of the mass matrix to 2*N
    iMh = sym.zeros(2 * n)
    for i in range(0, 2 * n, 2):
        iMh[i, i] = 1

    for i in range(1, 2 * n, 2):
        for j in range(1, 2 * n, 2):
            iMh[i, j] = M_inverse[int((i - 1) / 2), int((j - 1) / 2)]

    # Calculation of -C*x_dot-K*x+F
    D = -C * sym.Matrix(x[1::2]) - K * sym.Matrix(x[::2]) + f

    # Augmenting D to 2*N
    Dh = sym.zeros(2 * n, 1)
    for i in range(1, 2 * n, 2):
        Dh[i - 1] = x[i]
        Dh[i] = D[int((i - 1) / 2)]

    a = sym.simplify(iMh * Dh)
    B = sym.simplify(iMh * B_init)
    return a, B


def dX(X, x):
    u""" Returns the first derivative of X w.r.t x
    First derivative matrix term of vector X where X is a vector. For each vector entry, the function outputs a row
    vector in the output matrix where each element (i) corresponds to the derivative of X with respect to x(i)
    (This function can be used for checking the correctness of It\u014d-Taylor 1.5 scheme derivations.)

    :param X: Vector of symbolic expressions
    :param x: Vector of symbolic variables
    :return: Matrix of derivatives of each term in X with respect to x
    """

    n = X.shape[0]
    m = len(x)
    dx = sym.zeros(n, m)
    for i in range(0, n):
        for j in range(0, m):
            dx[i, j] = sym.diff(X[i], x[j])
    return sym.simplify(dx)


def ddX(X, x, B):
    u""" Double derivative term of vector X
    Computes the double derivative matrix of vector X based on state vector x and
    augmented matrix of random factors be as calculated for the It\u014d-Taylor 1.5 scheme.
    :param X: Vector of symbolic expressions
    :param x: Vector of symbolic variables
    :param B: Matrix of excitation amplitude
    :return: Matrix of the sum of double derivatives of each term in X with respect to x
    """

    n = len(x)
    ddx = sym.zeros(X.shape[0], X.shape[1])
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, B.shape[1]):
                ddx = ddx + B[i, k] * B[j, k] * sym.diff(X, x[j], x[i])
    return sym.simplify(ddx)


def L0(X, a, B, x, t: sym.Symbol):
    """ Derivation of the first Kolmogorov operator.
    Derives the result of applying the first Kolmogorov operator to X.

    :param X: Vector of symbolic expressions
    :param a: Drift vector
    :param B: Diffusion matrix
    :param x: State vector
    :param t: Time variable
    :return: Vector of symbolic expressions
    """
    n = a.shape[0]
    dim1, dim2 = X.shape
    part1 = sym.diff(X, t)  # Derivative with respect to t

    # First derivative terms
    part2 = sym.zeros(dim1, dim2)
    for k in range(0, n):
        part2 = part2 + a[k] * sym.diff(X, x[k])

    # Second derivative terms
    part3 = sym.zeros(dim1, dim2)
    for p in range(0, n):
        for k in range(0, n):
            for j in range(0, B.shape[1]):
                part3 = part3 + B[k, j] * B[p, j] * sym.diff(X, x[k], x[p])

    # Final formulation
    Lo = part1 + part2 + 0.5 * part3
    return Lo


def LJ(X, B, x, j: int):
    """First derivation of the second Kolmogorov operator.
        Derives the result of applying the first Kolmogorov operator to X to the jth column of B.
    :param X: Vector of symbolic expressions
    :param B: Diffusion matrix
    :param x: State vector
    :param j: Index of computation (which state variable to be used)
    :return: Vector of symbolic expressions
    """

    n = X.shape[0]
    Lj = sym.zeros(n, 1)
    for k in range(0, B.shape[0]):
        Lj = Lj + B[k, j] * sym.diff(X, x[k])

    # Final formulation
    return sym.simplify(Lj)


def LJ_total(X, B, x):
    """First derivation of the second Kolmogorov operator.
        Derives the result of applying the first Kolmogorov operator to X for every column of B
    :param X: Vector of symbolic expressions
    :param B: Diffusion matrix
    :param x: State vector
    :return: Vector of symbolic expressions
    """
    n_dof = len(x)
    n_rvs = B.shape[1]
    x_dim = X.shape[1]
    lj = sym.zeros(n_dof, n_rvs)
    for i in range(0, n_rvs):
        if x_dim == 1:
            vector = X
        else:
            vector = X[:, i]
        lj[:, i] = LJ(vector, B, x, i)
    return lj


def L1L1_total(X, x):
    """Double application of the LJ operator to matrix B (this is the last term of the It\u014d-Taylor
    update

    :param X: Matrix
    :param x: State vector
    :return: Matrix of symbolic expressions
    """
    n_dof = len(x)
    n_rvs = X.shape[1]
    l1l1 = sym.zeros(n_dof, n_rvs)
    for i in range(0, n_rvs):
        l1l1[:, i] = LJ(LJ(X[:, i], X, x, i), X, x, i)
    return l1l1
