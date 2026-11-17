"""Simulate a camera photo of the chart (blur, vignetting, noise) to demonstrate the workflow."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(7)
CHARTS = Path(__file__).resolve().parent / "charts"
chart = np.asarray(Image.open(CHARTS / "test_chart_A4.png").convert("L")).astype(float) / 255.0
img = gaussian_filter(chart, 1.3)                                   # lens blur
h, w = img.shape
yy, xx = np.mgrid[:h, :w]
r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
img = img * (1 - 0.35 * r2)                                         # vignetting
img = img + rng.normal(0, 1, img.shape) * (0.004 + 0.012 * np.sqrt(np.clip(img, 0, 1)))  # noise
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(CHARTS / "demo_photo.png")
print(f"Saved {CHARTS / 'demo_photo.png'}", img.shape)
