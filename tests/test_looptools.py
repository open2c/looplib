from looplib.looptools import *
import pytest

def test_convert_loops_to_sites():
    tuple_loops = [(4,5), (0,6),(2,3)]
    lsites, rsites = convert_loops_to_sites(tuple_loops)
    assert np.all(lsites == [4,0,2])

    array_loops = np.array(tuple_loops)
    lsites, rsites = convert_loops_to_sites(array_loops)
    assert np.all(lsites == [4,0,2])

    with pytest.raises(Exception):
        lsites, rsites = convert_loops_to_sites(array_loops.T)

def test_get_roots():
    nested_loops = [(4,5), (0,6),(2,3)]
    # convert to Nx2 array of loops
    nested_loops = np.array(nested_loops)
    # only the middle loop is a root loop
    assert np.all(np.array([False, True, False]) == get_roots(nested_loops))

def test_FRiP():
    # Tests for FRiP
    lattice_length = 5
    boundary_list = [1, 4]

    lef_A = [1, 1, 4, 1]
    assert FRiP(lattice_length, lef_A, boundary_list) == 1

    lef_B = [3, 0, 2, 3]
    assert FRiP(lattice_length, lef_B, boundary_list) == 0

    lef_C = [1, 1, 3, 3]
    assert FRiP(lattice_length, lef_C, boundary_list) == 0.5
