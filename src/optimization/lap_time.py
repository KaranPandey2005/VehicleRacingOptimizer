"""
Lap time integration and summary telemetry derived from a solved speed profile.

Lap time:
    t = integral of ds / v(s) around the closed loop, approximated with the
    trapezoidal rule on 1/v (more accurate than using ds/v_avg with the
    arithmetic mean of endpoint speeds, since time-per-distance is convex in v).

Longitudinal acceleration (for telemetry / G-analysis):
    a_lon[i] approximated from finite differences of v^2 over ds:
        a_lon = (v[i+1]^2 - v[i]^2) / (2*ds)

Lateral acceleration:
    a_lat = v^2 * curvature
"""

import numpy as np


def compute_lap_time(profile):
    """Integrate ds/v around the closed loop using the trapezoidal rule on 1/v."""
    v = profile["v"]
    ds = profile["ds"]
    inv_v = 1.0 / np.maximum(v, 0.1)  # guard against div-by-zero
    # trapezoidal: for each segment i -> i+1 (wrapping), average of 1/v at endpoints * ds
    inv_v_next = np.roll(inv_v, -1)
    dt = 0.5 * (inv_v + inv_v_next) * ds
    return float(np.sum(dt))


def compute_telemetry(profile):
    """Derive acceleration channels and summary stats from a solved speed profile."""
    v = profile["v"]
    ds = profile["ds"]
    curvature = profile["curvature"]

    v_next = np.roll(v, -1)
    a_lon = (v_next ** 2 - v ** 2) / (2.0 * np.maximum(ds, 1e-6))
    a_lat = v ** 2 * curvature

    return {
        "a_lon": a_lon,
        "a_lat": a_lat,
        "max_speed": float(np.max(v)),
        "min_speed": float(np.min(v)),
        "max_speed_kmh": float(np.max(v) * 3.6),
        "peak_lon_accel_g": float(np.max(a_lon) / 9.81),
        "peak_lon_decel_g": float(-np.min(a_lon) / 9.81),
        "peak_lat_accel_g": float(np.max(np.abs(a_lat)) / 9.81),
    }
