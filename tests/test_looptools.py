from looplib.looptools import *


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
