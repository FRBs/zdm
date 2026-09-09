import pytest

from zdm import parameters

def test_init_state():

    state = parameters.State()

    # Fuss a bit
    assert state.analysis.NewGrids 


def test_broken_power_law_parameters():
    state = parameters.State()

    assert state.energy.gamma2 == -2.0
    assert state.energy.gamma3 == -3.0
    assert state.energy.lEb == 40.0
    assert state.energy.lEb2 == 41.0
    assert state.params["gamma2"] == "energy"
    assert state.params["gamma3"] == "energy"
    assert state.params["lEb"] == "energy"
    assert state.params["lEb2"] == "energy"

test_init_state()
