"""
Created on 6 avr. 2021

@author: Paul Mucchielli
"""
import sympy as sym


def ismember(to_find, to_search):
    """ Returns True if "to_find" is found in "to_search"

    :param to_search: item to be searched
    :param to_find: item to find
    :return: If the item is found, True is returned
    """
    if isinstance(to_find, to_search.__class__):
        return to_find == to_search

    bind = {}
    for i, elt in enumerate(to_search):
        if elt not in bind:
            bind[elt] = i
    try:
        isin = [bind.get(itm, None) for itm in to_find]
    except TypeError:
        isin = bind.get(to_find, None)
    return isin is not None


# def make_diagonal(vector):
#     """
#     Returns a matrix where diagonal elements are that of the input symbolic vector
#
#     :param vector: Vector to diagonalize, one of its dimensions must be 1
#     :return: Square matrix with size equal the length of the vector
#     """
#     n = max(vector.shape)
#     matrix = sym.zeros(n)
#     for i in range(0, n):
#         matrix[i, i] = vector[i]
#     return matrix


# def diagonal(matrix):
#     """
#     Extracts the diagonal of a matrix
#
#     :param matrix: Matrix
#     :return:
#     """
#     n = len(matrix)
#     vector = list()
#     for i in range(0, n):
#         vector.append(matrix[i][i])
#     return vector


# def sum_of_vectors(matrix):
#     n, m = matrix.shape
#     vector = sym.zeros(n, 1)
#     for i in range(0, m):
#         vector = vector + matrix[0:, i]
#     return vector


def unique(symbol_set):
    """
    Returns a set where each entry is unique

    :param symbol_set: Set containing possible duplicates of symbols
    :return: Set with one entry per symbol
    """
    # Initialize a null list
    unique_set = set()

    # Traverse for all elements
    for x in symbol_set:
        # Check if exists in unique_list or not
        if x not in unique_set:
            unique_set.add(x)

    return unique_set


def remove_vars(variables: set, vars_to_remove):
    """
    Removes symbolic variables in "vars_to_remove" from set "variables"
    :param variables: Set of comparable objects
    :param vars_to_remove:
    :return: Original set without variables in vars_to_remove
    """
    remove_vars_list = set()
    for var in variables:
        if ismember(var, vars_to_remove):
            remove_vars_list.add(var)
    variables -= remove_vars_list
    return variables


def augment_vector(vector, n_dof: int):
    """ Augments a vector to 2*n_dof
    :param vector:
    :param n_dof: Number of initial degrees of freedom
    :return:
    """
    augmented_vec = sym.zeros(n_dof, 1)
    for i in range(1, n_dof, 2):
        augmented_vec[i] = vector[int((i - 1) / 2)]
    return augmented_vec


def augment_vectors(vectors, n_dof):
    """ Augments a matrix to 2*n_dof
    :param vectors:
    :param n_dof: Number of initial degrees of freedom
    :return:
    """
    augmented_vectors = sym.zeros(n_dof, vectors.shape[1])
    for i in range(0, vectors.shape[1]):
        augmented_vectors[:, i] = augment_vector(vectors[:, i], n_dof)
    return augmented_vectors


# def get_vector(matrix, vec_index):
#     n = matrix.shape[0]
#     vector_output = []
#     for i in range(0, n):
#         vector_output.append(matrix[i, vec_index])
#     return vector_output
