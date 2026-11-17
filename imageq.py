"""Camera image-quality measurements: slanted-edge MTF, SNR and noise.

The MTF method follows the principle of ISO 12233 (slanted edge):
  1. find the edge position in every row of the region of interest (ROI)
  2. fit a straight line through those positions (the edge is tilted a few degrees)
  3. project every pixel onto the distance from that line and bin at 1/4 pixel,
     which gives a 4x oversampled edge spread function (ESF)
  4. differentiate the ESF to get the line spread function (LSF), apply a Hamming window
  5. the MTF is the normalised magnitude of the Fourier transform of the LSF
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grey image to a float grey image (Rec. 709 luma weights)."""
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    return img


@dataclass
class MTFResult:
    freq: np.ndarray          # spatial frequency [cycles/pixel]
    mtf: np.ndarray           # normalised MTF
    esf_x: np.ndarray         # ESF sample positions [pixels]
    esf: np.ndarray           # oversampled edge spread function
    lsf: np.ndarray           # line spread function
    edge_angle_deg: float     # tilt of the edge
    mtf50: float              # frequency where MTF = 0.5 [cycles/pixel]
    mtf_nyquist: float        # MTF at 0.5 cycles/pixel


def slanted_edge_mtf(roi: np.ndarray, oversample: int = 4) -> MTFResult:
    """Measure the MTF from an ROI that contains one near-vertical slanted edge."""
    roi = to_gray(roi)
    rows, cols = roi.shape

    # 1. Edge position per row: centroid of the absolute horizontal derivative.
    #    Two passes: a rough fit, then a centroid restricted to a window around
    #    the edge so that noise far from the edge does not bias the estimate.
    # Edge finding uses a lightly smoothed copy with the noise floor removed;
    # the ESF itself (step 3) is built from the original, unsmoothed pixels.
    smooth = gaussian_filter1d(roi, 1.5, axis=1)
    deriv = np.abs(np.diff(smooth, axis=1))
    deriv = np.clip(deriv - np.median(deriv, axis=1, keepdims=True), 0, None)
    x = np.arange(cols - 1) + 0.5
    y = np.arange(rows)
    centroids = (deriv * x).sum(axis=1) / deriv.sum(axis=1)
    a, b = np.polyfit(y, centroids, 1)
    half_width = max(6, cols // 8)
    for _ in range(2):
        line = a * y + b
        win = np.abs(x[None, :] - line[:, None]) <= half_width
        w = deriv * win
        centroids = (w * x).sum(axis=1) / w.sum(axis=1)

        # 2. Fit a straight line x = a*y + b through the per-row edge positions
        a, b = np.polyfit(y, centroids, 1)
    angle = np.degrees(np.arctan(a))
    if abs(angle) < 1.0:
        raise ValueError(f"Edge angle {angle:.2f} deg is too small; tilt the edge 2-10 deg")

    # 3. Distance of every pixel from the fitted edge, binned at 1/oversample pixel
    xx, yy = np.meshgrid(np.arange(cols), y)
    dist = (xx - (a * yy + b)) * np.cos(np.arctan(a))
    bins = np.round(dist * oversample).astype(int)
    bins -= bins.min()
    counts = np.bincount(bins.ravel())
    sums = np.bincount(bins.ravel(), weights=roi.ravel())
    valid = counts > 0
    esf = np.interp(np.arange(len(counts)), np.flatnonzero(valid), sums[valid] / counts[valid])
    esf_x = (np.arange(len(esf)) + bins.min()) / oversample

    # Make the ESF rise from dark to bright so the LSF is positive
    if esf[-10:].mean() < esf[:10].mean():
        esf = esf[::-1]

    # 4. LSF = derivative of the ESF, windowed
    lsf = np.gradient(esf)
    lsf *= np.hamming(len(lsf))

    # 5. MTF = |FFT(LSF)|, normalised to 1 at zero frequency
    spectrum = np.abs(np.fft.rfft(lsf))
    mtf = spectrum / spectrum[0]
    freq = np.fft.rfftfreq(len(lsf), d=1.0 / oversample)  # cycles per pixel

    # The derivative by finite difference attenuates high frequencies; correct for it
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sinc(freq / oversample)
    mtf = np.where(corr > 0.1, mtf / corr, mtf)

    keep = freq <= 1.0
    freq, mtf = freq[keep], mtf[keep]
    mtf50 = _crossing(freq, mtf, 0.5)
    mtf_nyq = float(np.interp(0.5, freq, mtf))
    return MTFResult(freq, mtf, esf_x, esf, lsf, angle, mtf50, mtf_nyq)


def _crossing(freq, mtf, level):
    """First frequency where the MTF falls below `level` (linear interpolation)."""
    below = np.flatnonzero(mtf < level)
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(freq[0])
    f0, f1, m0, m1 = freq[i - 1], freq[i], mtf[i - 1], mtf[i]
    return float(f0 + (level - m0) * (f1 - f0) / (m1 - m0))


@dataclass
class PatchStats:
    mean: float
    std: float
    snr: float      # mean / std
    snr_db: float   # 20*log10(SNR)


def patch_snr(patch: np.ndarray, detrend: bool = True) -> PatchStats:
    """Signal-to-noise ratio of a uniform patch.

    With detrend=True a plane is removed first, so slow illumination gradients
    (common in phone photos) are not counted as noise.
    """
    p = to_gray(patch)
    if detrend:
        yy, xx = np.mgrid[: p.shape[0], : p.shape[1]]
        A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(p.size)])
        coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
        resid = p.ravel() - A @ coef
        std = float(resid.std(ddof=1))
    else:
        std = float(p.std(ddof=1))
    mean = float(p.mean())
    snr = mean / std if std > 0 else float("inf")
    return PatchStats(mean, std, snr, 20 * np.log10(snr))
