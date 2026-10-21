"""
Deterministic wet μ(s,n): rubber corridor plus geometry puddles.

Illustrative, not a rain model. Dry and damp return a uniform 1.0 here
(damp still uses Weather.grip_modifier on the tire). Wet does not also
multiply the tire by 0.65 — this map is the wet cut.

No RNG: two evals on the same track and offset are identical.
"""

from __future__ import annotations

import numpy as np

from src.weather.weather import Condition

# Relative to dry asphalt. Product of factors, then clipped.
WET_FILM = 0.52
WET_RUBBER_PEAK = 0.80
RUBBER_SIGMA_M = 1.5
EDGE_START_FRAC = 0.75
EDGE_CUT = 0.25
DIP_CUT = 0.30
DIP_SIGMA_M = 40.0
GRIP_MIN = 0.35
GRIP_MAX = 0.85


def as_grip_array(grip_s, line):
    """Broadcast a scalar, array, or callable(line) to length-n grip_s."""
    n = len(line["s"])
    if grip_s is None:
        return np.ones(n, dtype=float)
    if callable(grip_s):
        raw = grip_s(line)
    else:
        raw = grip_s
    out = np.asarray(raw, dtype=float)
    if out.ndim == 0:
        return np.full(n, float(out))
    return np.broadcast_to(out, (n,)).astype(float).copy()


def rubber_reference_offset(track):
    """
    Track-owned dry rubber stripe: generic shaped outside-apex-outside.

    vehicle=None so this is a surface property, not this car's apex bias.
    """
    cached = getattr(track, "_rubber_offset", None)
    if cached is not None and np.shape(cached)[0] == len(track.s):
        return np.asarray(cached, dtype=float)
    from src.optimization.racing_line import shaped_racing_line
    line = shaped_racing_line(track, vehicle=None)
    n_ref = np.asarray(line["offset"], dtype=float)
    try:
        track._rubber_offset = n_ref
    except Exception:
        pass
    return n_ref


def _dip_weight(track):
    """1 at strict local z minima (valleys), Gaussian along s. Flat → 0."""
    z = np.asarray(getattr(track, "z", np.zeros_like(track.s)), dtype=float)
    n = z.size
    if n < 3 or float(np.ptp(z)) < 0.25:
        return np.zeros(n, dtype=float)
    prev = np.roll(z, 1)
    nxt = np.roll(z, -1)
    is_min = (z < prev - 1e-9) & (z < nxt - 1e-9)
    thresh = float(np.min(z) + 0.35 * np.ptp(z))
    mins = np.where(is_min & (z <= thresh))[0]
    if mins.size == 0:
        return np.zeros(n, dtype=float)
    s = np.asarray(track.s, dtype=float)
    length = float(track.length)
    sigma = DIP_SIGMA_M
    w = np.zeros(n, dtype=float)
    for i in mins:
        d = (s - s[i] + 0.5 * length) % length - 0.5 * length
        w = np.maximum(w, np.exp(-0.5 * (d / sigma) ** 2))
    return w


def spatial_grip_scale(track, offset, weather):
    """
    Per-sample grip relative to dry (1.0). Length of track.s.

    Wet: film * rubber Gaussian * edge gutters * dip pools off-rubber.
    Dry / damp / None: ones.
    """
    n = len(track.s)
    if weather is None or getattr(weather, "condition", Condition.DRY) != Condition.WET:
        return np.ones(n, dtype=float)
    if offset is None:
        n_lat = np.zeros(n, dtype=float)
    else:
        n_lat = np.asarray(offset, dtype=float)
        if n_lat.shape[0] != n:
            n_lat = np.zeros(n, dtype=float)

    n_ref = rubber_reference_offset(track)
    rubber = np.exp(-0.5 * ((n_lat - n_ref) / RUBBER_SIGMA_M) ** 2)
    half = 0.5 * float(track.width)
    edge_start = EDGE_START_FRAC * half
    edge = np.clip(
        (np.abs(n_lat) - edge_start) / max(0.25 * half, 1e-6),
        0.0, 1.0,
    )
    dip = _dip_weight(track)
    off_rubber = 1.0 - rubber
    boost = WET_RUBBER_PEAK / WET_FILM - 1.0
    g = (
        WET_FILM
        * (1.0 + boost * rubber)
        * (1.0 - EDGE_CUT * edge)
        * (1.0 - DIP_CUT * dip * off_rubber)
    )
    return np.clip(g, GRIP_MIN, GRIP_MAX)


def grip_along_line(track, weather):
    """Callable for qss_with_thermal / the optimizer: resample μ at line n(s)."""
    def _fn(line):
        return spatial_grip_scale(
            track, line.get("offset") if line is not None else None, weather,
        )
    return _fn
