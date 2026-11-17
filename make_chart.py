"""Create a printable A4 test chart: a slanted edge for MTF and grey patches for SNR/noise."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

OUT = Path(__file__).resolve().parent / "charts"
OUT.mkdir(exist_ok=True)

GREYS = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]  # printed reflectance targets (1 = white)


def draw(ax):
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 210, 297, color="white"))

    # Dark square rotated 5 degrees: its left and right sides are near-vertical slanted edges
    cx, cy, half, ang = 105, 195, 45, np.radians(5)
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    ax.add_patch(Polygon(corners @ rot.T + [cx, cy], closed=True, color=str(0.15)))
    ax.text(105, 260, "Slanted edge (5°) – use a vertical side for MTF", ha="center", fontsize=9)

    # Grey patches for SNR and noise
    w, gap, y0 = 28, 4, 60
    x0 = (210 - (len(GREYS) * w + (len(GREYS) - 1) * gap)) / 2
    for i, g in enumerate(GREYS):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y0), w, w, color=str(g), ec="black", lw=0.3))
        ax.text(x + w / 2, y0 - 6, f"{int(g * 100)} %", ha="center", fontsize=8)
    ax.text(105, y0 + w + 8, "Grey patches – SNR and noise vs. signal", ha="center", fontsize=9)
    ax.text(105, 20, "Print at 100 % scale on matte paper. Photograph flat, evenly lit, no zoom.",
            ha="center", fontsize=8, color="0.3")


fig = plt.figure(figsize=(8.27, 11.69))
ax = fig.add_axes((0, 0, 1, 1))
draw(ax)
fig.savefig(OUT / "test_chart_A4.pdf")
fig.savefig(OUT / "test_chart_A4.png", dpi=150)
print("Saved charts/test_chart_A4.pdf and .png")
