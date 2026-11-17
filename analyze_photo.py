"""Measure MTF and SNR from a photo of the test chart.

Easiest way - choose the regions with the mouse:
    python analyze_photo.py photo.jpg --select

    1. Drag a box around ONE near-vertical side of the dark square (no corners), press Enter.
    2. Drag a box inside each grey patch, pressing Enter after each one.
    3. Press q (or close the window) when done.
    The regions are saved to <photo>_rois.json and the analysis runs straight away.

Run again later with the same regions:
    python analyze_photo.py photo.jpg --rois photo_rois.json

Or give the regions by hand (x0 y0 x1 y1 in pixels):
    python analyze_photo.py photo.jpg --edge 1200 900 1330 1050 --patch 600 2000 700 2100

Results are written to results/<photo name>/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from imageq import patch_snr, slanted_edge_mtf, to_gray

PROJECT = Path(__file__).resolve().parent


def load(path):
    img = ImageOps.exif_transpose(Image.open(path))  # respect phone orientation
    return np.asarray(img.convert("RGB")).astype(float) / 255.0


def crop(img, roi):
    x0, y0, x1, y1 = roi
    return img[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)]


def select_rois(gray):
    """Let the user drag rectangles: first the edge, then one per grey patch."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    rois, current = [], {}
    fig, ax = plt.subplots(figsize=(10, 10 * gray.shape[0] / gray.shape[1]))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)

    def title():
        step = "EDGE (one vertical side of the dark square)" if not rois else f"GREY PATCH {len(rois)}"
        ax.set_title(f"Drag a box around the {step}, press Enter. Press q when done.")
        fig.canvas.draw_idle()

    def on_select(press, release):
        current["roi"] = [int(press.xdata), int(press.ydata), int(release.xdata), int(release.ydata)]

    from matplotlib.patches import Rectangle

    def on_key(event):
        if event.key == "enter" and "roi" in current:
            r = current.pop("roi")
            rois.append(r)
            color = "red" if len(rois) == 1 else "lime"
            ax.add_patch(Rectangle((min(r[0], r[2]), min(r[1], r[3])), abs(r[2] - r[0]),
                                   abs(r[3] - r[1]), fill=False, color=color, lw=1.5))
            title()

    selector = RectangleSelector(ax, on_select, useblit=True, interactive=True)  # keep a reference
    fig.canvas.mpl_connect("key_press_event", on_key)
    title()
    plt.show()
    if not rois:
        raise SystemExit("No regions selected.")
    return {"edge": rois[0], "patches": rois[1:]}


def analyse(img, rois, out_dir, image_name):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    gray = to_gray(img)
    summary = {"image": image_name, "image_size_px": [gray.shape[1], gray.shape[0]], "rois": rois}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if rois.get("edge"):
        res = slanted_edge_mtf(crop(img, rois["edge"]))
        summary["mtf"] = {"edge_angle_deg": round(res.edge_angle_deg, 2),
                          "mtf50_cycles_per_pixel": round(res.mtf50, 4),
                          "mtf_at_nyquist": round(res.mtf_nyquist, 4)}
        axes[0].plot(res.esf_x, res.esf)
        axes[0].set(title="Edge spread function (4x oversampled)",
                    xlabel="Distance from edge [px]", ylabel="Signal")
        axes[1].plot(res.freq, res.mtf, lw=2)
        axes[1].axhline(0.5, ls=":", color="grey")
        axes[1].axvline(0.5, ls=":", color="grey")
        axes[1].set(title=f"MTF – MTF50 = {res.mtf50:.3f} cy/px",
                    xlabel="Spatial frequency [cycles/pixel]", ylabel="MTF", xlim=(0, 0.8), ylim=(0, 1.1))
        print(f"Edge angle {res.edge_angle_deg:.2f} deg | MTF50 {res.mtf50:.3f} cy/px | "
              f"MTF at Nyquist {res.mtf_nyquist:.3f}")

    patches = rois.get("patches", [])
    if patches:
        stats = sorted((patch_snr(crop(img, r)) for r in patches), key=lambda s: s.mean)
        summary["patches"] = [{"mean": round(s.mean, 4), "noise_std": round(s.std, 5),
                               "snr_db": round(s.snr_db, 2)} for s in stats]
        axes[2].plot([s.mean for s in stats], [s.snr_db for s in stats], "o-")
        axes[2].set(title="SNR vs. signal level", xlabel="Mean signal (0-1)", ylabel="SNR [dB]")
        for s in stats:
            print(f"Patch mean {s.mean:.3f} | noise std {s.std:.4f} | SNR {s.snr_db:.1f} dB")

    for ax in axes:
        if not ax.lines:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "image_quality_report.png", dpi=150)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'image_quality_report.png'} and {out_dir / 'summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("--select", action="store_true", help="choose the regions with the mouse")
    p.add_argument("--rois", type=Path, help="JSON file with saved regions")
    p.add_argument("--edge", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--patch", nargs=4, type=int, action="append", default=[],
                   metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--out", type=Path, help="output folder (default: results/<photo name>)")
    a = p.parse_args()

    img = load(a.image)
    out_dir = a.out or PROJECT / "results" / a.image.stem

    if a.select:
        rois = select_rois(to_gray(img))
        roi_file = a.image.with_name(a.image.stem + "_rois.json")
        roi_file.write_text(json.dumps(rois, indent=2))
        print(f"Saved regions to {roi_file}")
    elif a.rois:
        rois = json.loads(a.rois.read_text())
    elif a.edge or a.patch:
        rois = {"edge": a.edge, "patches": a.patch}
    else:
        p.error("choose regions with --select, --rois or --edge/--patch")

    analyse(img, rois, out_dir, a.image.name)


if __name__ == "__main__":
    main()
