"""Synthetic camera images with a known answer, used for validation and tests."""
import numpy as np
from scipy.ndimage import gaussian_filter


def render_edge(size=128, angle_deg=5.0, sigma_px=1.0, ss=8, dark=0.15, bright=0.85,
                noise_std=0.0, rng=None):
    """Slanted edge: rendered at ss x resolution, blurred (optics), averaged into pixels (sensor)."""
    rng = rng or np.random.default_rng(0)
    n = size * ss
    yy, xx = np.mgrid[:n, :n] / ss
    a = np.tan(np.radians(angle_deg))
    img = np.where(xx - (size / 2 + a * (yy - size / 2)) > 0, bright, dark).astype(float)
    img = gaussian_filter(img, sigma_px * ss)
    img = img.reshape(size, ss, size, ss).mean(axis=(1, 3))
    if noise_std:
        img = img + rng.normal(0, noise_std, img.shape)
    return img


def true_mtf(f, sigma_px):
    """Analytic MTF of Gaussian blur times a square pixel aperture."""
    return np.exp(-2 * np.pi**2 * sigma_px**2 * f**2) * np.abs(np.sinc(f))


def true_mtf50(sigma_px):
    f = np.linspace(0, 1, 20001)
    return float(f[np.argmax(true_mtf(f, sigma_px) < 0.5)])
