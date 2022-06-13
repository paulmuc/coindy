"""
Created on 6 avr. 2021

@author: Paul
"""

import sympy as sym


def ito_sde_form(x, M, C, K, F, B_init):
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
    D = -C * sym.Matrix(x[1::2]) - K * sym.Matrix(x[::2]) + F

    # Augmenting D to 2*N
    Dh = sym.zeros(2 * n, 1)
    for i in range(1, 2 * n, 2):
        Dh[i - 1] = x[i]
        Dh[i] = D[int((i - 1) / 2)]

    a = sym.simplify(iMh * Dh)
    b = sym.simplify(iMh * B_init)
    return a, b


def dX(X, x):
    # DX First derivative matrix term of vector X where X is either a or b in a
    # diffusion equation of the form dYt = adt + bdWt. Then the second term of
    # L0 applied to X is dx*a. This function is used for checking the
    # correctness of Ito-Taylor 1.5 scheme derivations.
    n = X.shape[0]
    dx = sym.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            dx[i, j] = sym.diff(X[i], x[j])
    return sym.simplify(dx)


def ddX(X, x, be):
    # DDX Double derivative term of vector X
    #   ddx = DDX(X,x,be) computed the double derivative matrix of vector X based on state vector x and
    #   augmented matrix of random factors be as calculated for the Ito-Taylor 1.5 scheme.
    n = len(x)
    ddx = sym.zeros(X.shape[0], X.shape[1])
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, be.shape[1]):
                ddx = ddx + be[i, k] * be[j, k] * sym.diff(X, x[j], x[i])
    return sym.simplify(ddx)


def L0(X, a, bt, x, t):
    # L0 Derivation of the first Kolmogorov operator.
    #
    #       L0(X,a,x,bt,'scratch') derives L1L1(X) where X is a symbolic vector with variables
    #       contained in the state symbolic vector x and bt the extended version of b are inputs.
    #       Use the 'scratch' option as fourth input. This is the default setting
    #       L1L1(b,dX,ddX,'base') derives L1L1(X) from previously computed dX, ddX
    #       when option 'base' is input in fourth position.
    #
    #       Inputs: 
    #       - X variable to be processed through the L1L1 operator
    #       - x state vector of symbolic variables x = x_1,...,x_2*N) where N is
    #       number of DOF
    #       - bt extended version of b in dxt = adt+bdWt
    #       - dX First derivative matrix of X
    #       - ddX Second derivative vector of X
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
            for j in range(0, bt.shape[1]):
                part3 = part3 + bt[k, j] * bt[p, j] * sym.diff(X, x[k], x[p])

    # Final formulation
    Lo = part1 + part2 + 0.5 * part3
    return Lo


def LJ(X, b, x, j):
    # LJ First derivation of the second Kolmogorov operator.
    #
    #       LJ(X,b,x,j) provides the second Kolmogorov derivation of X with
    #       respect to the symbolic state vector x. Vector b is the diffusion
    #       vector and j denotes the index of the derivation for example. if
    #       L1(X) is sought, then type LJ(X,b,x,1), if L2(X) is sought, type
    #       LJ(X,b,x,2). The general rule is then LN(X) = LJ(X,b,x,n) where n =
    #       1, ..., length(b).

    n = X.shape[0]
    Lj = sym.zeros(n, 1)
    for k in range(0, b.shape[0]):
        Lj = Lj + b[k, j] * sym.diff(X, x[k])

    # Final formulation
    return sym.simplify(Lj)


def L1L1(*args):
    # L1L1 Double derivation of the second Kolmogorov operator.
    #       l1l1 = L1L1(b,X,x,bt,'scratch') derives L1L1(X) where X is a symbolic vector with variables
    #       contained in the state symbolic vector x and bt the extended version of b are inputs.
    #       Use the 'scratch' option as fourth input. This is the default setting
    #       l1l1 = L1L1(b,dX,ddX,'base') derives L1L1(X) from previously computed dX, ddX
    #       when option 'base' is input in fourth position
    #
    #       Inputs: 
    #       - X variable to be processed through the L1L1 operator
    #       - x state vector of symbolic variables x = x_1,...,x_2*N) where N is
    #       number of DOF
    #       - bt extended version of b in dxt = adt+bdWt
    #       - dX First derivative matrix of X
    #       - ddX Second derivative vector of X
    # All inputs must be symbolic
    n_arg_in = len(args)
    if n_arg_in == 5 or (n_arg_in == 4 and not isinstance(args[3], str)):  # By default, perform operation from scratch
        b = args[0]
        X = args[1]
        x = args[2]
        bt = args[3]
        dXc = dX(X, x)
        ddXc = ddX(X, x, bt)
    elif n_arg_in == 4:  # if dX and ddX provided, calculate from them
        if args[3] == "base":
            b = args[0]
            dXc = args[1]
            ddXc = args[2]
        else:
            TypeError('Wrong option, if using three inputs, use "base"')
    else:
        if n_arg_in < 4:
            TypeError('Not enough input arguments')

    # Final formulation
    return sym.simplify((b * ddXc.transpose() + dXc ** 2) * b).transpose()
