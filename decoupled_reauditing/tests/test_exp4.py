import pytest

from decoupled_reauditing.experiments.exp4_pooloverlap import OVERLAPS, simulate_gap


def test_exp4_gap_non_decreasing():
    gaps = [simulate_gap(overlap) for overlap in OVERLAPS]
    assert all(b + 1e-12 >= a for a, b in zip(gaps, gaps[1:]))


def test_exp4_gap_rises_from_disjoint_to_identical():
    gap0 = simulate_gap(0.0)
    gap1 = simulate_gap(1.0)
    assert gap0 < gap1


def test_exp4_disjoint_overlap_has_near_zero_gap():
    assert simulate_gap(0.0) == pytest.approx(0.0)
