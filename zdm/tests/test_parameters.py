import pytest

from zdm import parameters

def test_init_state():

    state = parameters.State()

    # Fuss a bit
    assert state.analysis.NewGrids 

test_init_state()