"""Automated tests: the measurements must match known answers on synthetic images."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imageq import patch_snr, slanted_edge_mtf  # noqa: E402
from synthetic import render_edge, true_mtf50  # noqa: E402


@pytest.mark.parametrize("sigma", [0.6, 1.0, 1.6])
@pytest.mark.parametrize("noise", [0.0, 0.01, 0.03])
def test_mtf50_matches_theory(sigma, noise):
    res = slanted_edge_mtf(render_edge(sigma_px=sigma, noise_std=noise, rng=np.random.default_rng(1)))
    assert res.mtf50 == pytest.approx(true_mtf50(sigma), rel=0.05)


@pytest.mark.parametrize("angle", [3.0, 5.0, 8.0])
def test_edge_angle_is_measured(angle):
    res = slanted_edge_mtf(render_edge(angle_deg=angle, noise_std=0.01))
    assert res.edge_angle_deg == pytest.approx(angle, abs=0.2)


def test_mtf_is_one_at_zero_frequency():
    res = slanted_edge_mtf(render_edge())
    assert res.mtf[0] == pytest.approx(1.0)


def test_edge_without_tilt_is_rejected():
    with pytest.raises(ValueError):
        slanted_edge_mtf(render_edge(angle_deg=0.0))


def test_dark_to_bright_and_bright_to_dark_give_same_mtf():
    img = render_edge(sigma_px=1.0)
    assert slanted_edge_mtf(img).mtf50 == pytest.approx(slanted_edge_mtf(img[:, ::-1]).mtf50, rel=0.02)


def test_snr_ignores_illumination_gradient():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:64, :64]
    patch = 0.5 * (1 + 0.05 * xx / 64) + rng.normal(0, 0.01, (64, 64))
    assert patch_snr(patch).std == pytest.approx(0.01, rel=0.05)
    assert patch_snr(patch, detrend=False).std > 0.012
