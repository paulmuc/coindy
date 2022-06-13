"""
Created on 6 avr. 2021

@author: Paul Mucchielli
"""

import sympy as sym


def is_in(vector, variable):
    if not isinstance(vector, sym.Symbol):
        for symbol in vector:
            if variable == symbol:
                return symbol
        return None
    else:
        if variable == vector:
            return vector
        return None


def make_diagonal(vector):
    n = max(vector.shape)
    matrix = sym.zeros(n)
    for i in range(0, n):
        matrix[i, i] = vector[i]
    return matrix


def diagonal(matrix):
    n = len(matrix)
    vector = list()
    for i in range(0, n):
        vector.append(matrix[i][i])
    return vector


def sum_of_vectors(matrix):
    n, m = matrix.shape
    vector = sym.zeros(n, 1)
    for i in range(0, m):
        vector = vector + matrix[0:, i]
    return vector


# function to get unique values
def unique(set1):
    # initialize a null list
    unique_set = set()

    # traverse for all elements
    for x in set1:
        # check if exists in unique_list or not
        if x not in unique_set:
            unique_set.add(x)

    return unique_set


def remove_vars(variables, vars_to_remove):
    remove_vars_list = set()
    for var in variables:
        if is_in(vars_to_remove, var):
            remove_vars_list.add(var)
    variables -= remove_vars_list
    return variables


def get_vector(matrix, vec_index):
    n = matrix.shape[0]
    vector_output = []
    for i in range(0, n):
        vector_output.append(matrix[i, vec_index])
    return vector_output
