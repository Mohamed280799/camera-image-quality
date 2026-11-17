"""Validate the MTF and SNR code on synthetic images with a known answer.

A slanted edge is rendered at 8x resolution, blurred with a Gaussian of known
width (the "lens"), averaged into pixels (the "sensor") and given noise.
The true system MTF is then known analytically:

    MTF(f) = exp(-2 * pi^2 * sigma^2 * f^2) * |sinc(f)|
             (Gaussian blur)                  (square pixel aperture)

so the measured MTF can be checked against it.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imageq import patch_snr, slanted_edge_mtf
from synthetic import render_edge, true_mtf, true_mtf50

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


rows = []
fig, ax = plt.subplots(figsize=(7, 4.2))
for sigma, color in [(0.6, "C0"), (1.0, "C1"), (1.6, "C2")]:
    for noise in [0.0, 0.01, 0.03]:
        res = slanted_edge_mtf(render_edge(sigma_px=sigma, noise_std=noise, rng=rng))
        exp50 = true_mtf50(sigma)
        rows.append([sigma, noise, round(res.edge_angle_deg, 2), round(exp50, 4),
                     round(res.mtf50, 4), round(100 * (res.mtf50 - exp50) / exp50, 1)])
        if noise == 0.01:
            ax.plot(res.freq, res.mtf, color=color, lw=2,
                    label=f"measured, blur sigma = {sigma} px")
            ax.plot(res.freq, true_mtf(res.freq, sigma), "--", color=color, lw=1,
                    label=f"theory, sigma = {sigma} px")
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.axvline(0.5, color="grey", lw=0.8, ls=":")
ax.text(0.505, 0.02, "Nyquist", color="grey", fontsize=8)
ax.set(xlim=(0, 0.8), ylim=(0, 1.05), xlabel="Spatial frequency [cycles/pixel]", ylabel="MTF",
       title="Slanted-edge MTF: measured vs. theory (noise std = 1 % of full scale)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "mtf_validation.png", dpi=150)

with open(OUT / "mtf_validation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["blur_sigma_px", "noise_std", "edge_angle_deg", "mtf50_theory", "mtf50_measured", "error_pct"])
    w.writerows(rows)

print("MTF50 validation (cycles/pixel)")
print(f"{'sigma':>6} {'noise':>6} {'angle':>6} {'theory':>8} {'measured':>9} {'error %':>8}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>6} {r[2]:>6} {r[3]:>8} {r[4]:>9} {r[5]:>8}")

# SNR validation: uniform patches with known noise, plus a slow illumination gradient
print("\nSNR validation (patch 64x64 with a 5 % brightness gradient)")
snr_rows = []
for level in [0.1, 0.3, 0.5, 0.7, 0.9]:
    true_std = 0.002 + 0.01 * np.sqrt(level)  # read noise + shot-noise-like term
    yy, xx = np.mgrid[:64, :64]
    patch = level * (1 + 0.05 * xx / 64) + rng.normal(0, true_std, (64, 64))
    s = patch_snr(patch)
    s_raw = patch_snr(patch, detrend=False)
    snr_rows.append([level, true_std, s.std, s.snr_db, s_raw.std])
    print(f"level {level:.1f}: true std {true_std:.4f}, measured {s.std:.4f} "
          f"(without detrending {s_raw.std:.4f}), SNR {s.snr_db:.1f} dB")

with open(OUT / "snr_validation.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["level", "true_std", "measured_std_detrended", "snr_db", "measured_std_raw"])
    w.writerows([[round(v, 5) for v in r] for r in snr_rows])
