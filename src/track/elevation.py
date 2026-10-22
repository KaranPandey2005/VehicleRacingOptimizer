"""
Vertical track profile z(s) and optional bank angle φ(s).

z is height along the centerline (same left and right unless a future
bowl). Grade is the path pitch; kappa_v is vertical curvature (dip
compresses, crest unloads). φ is roll about the centerline, default 0.
Spa has no bank map — do not invent one.
"""

from __future__ import annotations
import numpy as np


def cosine_hill(s, length, s_center, width, height):
    """
    C1-smooth bump on a closed loop. z = 0 and z' = 0 at the edges of
    the window, so a flat track plus one hill still closes in height.
    """
    s = np.asarray(s, dtype=float)
    length = float(length)
    width = float(width)
    if width <= 1e-6 or length <= 1e-6:
        return np.zeros_like(s)
    d = (s - float(s_center) + 0.5 * length) % length - 0.5 * length
    z = np.zeros_like(s)
    half = 0.5 * width
    mask = np.abs(d) < half
    z[mask] = 0.5 * float(height) * (1.0 + np.cos(np.pi * d[mask] / half))
    return z


def cosine_bank(s, length, s_center, width, phi_peak):
    """C1-smooth bank-angle window (rad). Same shape as cosine_hill; tests only."""
    return cosine_hill(s, length, s_center, width, phi_peak)


def periodic_height(s, knots_s, knots_z, length):
    """C1 half-cosine interpolation of height knots on a closed lap."""
    from src.optimization.racing_line import _half_cosine_periodic
    return _half_cosine_periodic(s, knots_s, knots_z, length)


# Official 7004 m chainage vs AMSL (m) from the Circuit de Spa-Francorchamps
# public track plan (not a laser survey, not a DEM). Scaled onto the TUM/OSM
# centerline length in official_spa_z.
# https://ser2026.racspa.be/docs/CircuitdeSpa-Francorchamps-Fulltrack.pdf
SPA_OFFICIAL_LENGTH_M = 7004.0
SPA_OFFICIAL_ALT_M = (
    (0.0, 418.0),
    (220.0, 424.0),
    (310.0, 419.0),
    (690.0, 403.0),
    (890.0, 391.0),    # Eau Rouge floor
    (1130.0, 415.0),   # Raidillon
    (1490.0, 446.0),   # Kemmel
    (2200.0, 463.0),
    (2470.0, 470.0),   # Les Combes / Malmedy (high)
    (2870.0, 455.0),
    (3110.0, 438.0),
    (3600.0, 402.0),
    (4280.0, 377.0),
    (4740.0, 375.0),
    (5100.0, 373.0),   # Paul Frère (lap low)
    (5700.0, 388.0),
    (6000.0, 398.0),
    (6540.0, 408.0),
    (7004.0, 418.0),
)


def official_spa_z(s, length):
    """Interpolate published Spa altitudes onto a centerline of given length."""
    length = float(length)
    scale = length / SPA_OFFICIAL_LENGTH_M
    knots_s = np.array([s_m * scale for s_m, _alt in SPA_OFFICIAL_ALT_M], dtype=float)
    knots_z = np.array([alt for _s_m, alt in SPA_OFFICIAL_ALT_M], dtype=float)
    return periodic_height(s, knots_s, knots_z, length)


def path_segment_lengths_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)


def segment_grade(z, ds_xy):
    """sin(θ) and 3D length of each sample → next sample on a closed loop."""
    z = np.asarray(z, dtype=float)
    ds_xy = np.maximum(np.asarray(ds_xy, dtype=float), 1e-9)
    dz = np.roll(z, -1) - z
    ds_3d = np.sqrt(ds_xy * ds_xy + dz * dz)
    ds_3d = np.maximum(ds_3d, 1e-9)
    sin_theta = dz / ds_3d
    return sin_theta, ds_3d


def slope_arrays(z, ds_xy):
    """
    Pointwise sin(θ), cos(θ), and vertical curvature κ_v (1/m).

    κ_v > 0 is a dip (extra normal load). κ_v < 0 is a crest (unload).
    """
    sin_seg, ds_3d = segment_grade(z, ds_xy)
    sin_pt = 0.5 * (sin_seg + np.roll(sin_seg, 1))
    sin_pt = np.clip(sin_pt, -1.0, 1.0)
    cos_pt = np.sqrt(np.maximum(1.0 - sin_pt * sin_pt, 0.0))
    theta = np.arcsin(np.clip(sin_seg, -1.0, 1.0))
    dtheta = theta - np.roll(theta, 1)
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    ds_pt = 0.5 * (ds_3d + np.roll(ds_3d, 1))
    kappa_v = dtheta / np.maximum(ds_pt, 1e-9)
    return sin_pt, cos_pt, kappa_v
