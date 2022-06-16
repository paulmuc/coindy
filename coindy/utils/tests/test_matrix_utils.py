import pytest
import sympy as sym

from coindy import matrix_utils as mu


@pytest.fixture
def example_symbols_set():
    return {'a', 'a', 'b', 'c'}


def test_is_member(example_symbols_set):
    assert not (mu.is_member('x', example_symbols_set)) and (mu.is_member('a', example_symbols_set))


def test_unique(example_symbols_set):
    assert (mu.unique(example_symbols_set)) == {'a', 'b', 'c'}


def test_remove_vars(example_symbols_set):
    assert (mu.remove_vars(example_symbols_set, 'c')) == {'a', 'a', 'b'}


def test_augment_vector():
    vector = sym.Matrix([[0], [0], [0], [0]])

    assert (mu.augment_vector(vector, 8)) == sym.Matrix([[0], [0], [0], [0], [0], [0], [0], [0]])


def test_augment_vectors():
    matrix = sym.Matrix([[0, 0], [0, 0]])
    assert (mu.augment_vectors(matrix, 4)) == sym.Matrix([[0, 0], [0, 0], [0, 0], [0, 0]])
