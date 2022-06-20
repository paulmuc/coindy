import coindy.sde.sde_classes as sde
import pytest


def test_sde_model_algorithms_single():
    with pytest.raises(KeyError):
        sde_model = sde.SDEModel(2, 2, algorithm='bloup')


def test_sde_model_algorithms_multiple1():
    with pytest.raises(KeyError):
        sde_model = sde.SDEModel(2, 2, algorithm=['all', 'it'])


def test_sde_model_algorithms_multiple2():
    with pytest.raises(KeyError):
        sde_model = sde.SDEModel(2, 2, algorithm=['bloup', 'it'])


def test_sde_model_algorithms_multiple3():
    sde_model = sde.SDEModel(2, 2, algorithm=['em', 'mi', 'it'])
    assert sde_model.algorithm == 'all'