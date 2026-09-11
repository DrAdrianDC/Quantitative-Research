from __future__ import annotations

import numpy as np
import pytest

from src.drift import DriftError, ks_statistic, population_stability_index, psi_band


def test_psi_near_zero_when_samples_match():
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, size=2000)
    actual = rng.normal(0, 1, size=2000)
    psi = population_stability_index(expected, actual)
    assert psi < 0.10
    assert psi_band(psi) == "stable"


def test_psi_alerts_when_distribution_shifts():
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, size=2000)
    actual = rng.normal(3, 1, size=2000)
    psi = population_stability_index(expected, actual)
    assert psi >= 0.25
    assert psi_band(psi) == "significant"


def test_ks_detects_shift():
    rng = np.random.default_rng(2)
    ks_same = ks_statistic(rng.normal(0, 1, 1000), rng.normal(0, 1, 1000))
    ks_shift = ks_statistic(rng.normal(0, 1, 1000), rng.normal(2, 1, 1000))
    assert ks_shift > ks_same
    assert ks_shift > 0.3


def test_psi_rejects_tiny_samples():
    with pytest.raises(DriftError):
        population_stability_index(np.arange(5.0), np.arange(5.0))
