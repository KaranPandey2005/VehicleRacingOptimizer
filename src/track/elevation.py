"""
Vertical track profile z(s). Not banking.

z is height along the centerline (same left and right). Grade is the
path pitch; kappa_v is vertical curvature (dip compresses, crest unloads).
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


def periodic_height(s, knots_s, knots_z, length):
    """C1 half-cosine interpolation of height knots on a closed lap."""
    from src.optimization.racing_line import _half_cosine_periodic
    return _half_cosine_periodic(s, knots_s, knots_z, length)


# Relative height (m) vs fraction of lap. Not a survey: a recognizable
# Eau Rouge valley and Kemmel climb on the TUM/OSM 2D centerline.
# Peak-to-trough is ~40 m, in the public ballpark for Spa.
_SPA_Z_FRACTIONS = (
    (0.00, 40.0),  # La Source plateau
    (0.06, 30.0),
    (0.11, 6.0),   # drop
    (0.14, 0.0),   # Eau Rouge floor
    (0.22, 30.0),  # Raidillon
    (0.30, 38.0),  # Kemmel
    (0.36, 44.0),  # Les Combes (high)
    (0.48, 20.0),  # Rivage
    (0.56, 12.0),  # Pouhon
    (0.70, 26.0),  # Stavelot
    (0.86, 38.0),  # Blanchimont
    (0.97, 40.0),
    (1.00, 40.0),
)


def illustrative_spa_z(s, length):
    length = float(length)
    knots_s = np.array([f * length for f, _z in _SPA_Z_FRACTIONS], dtype=float)
    knots_z = np.array([z for _f, z in _SPA_Z_FRACTIONS], dtype=float)
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
