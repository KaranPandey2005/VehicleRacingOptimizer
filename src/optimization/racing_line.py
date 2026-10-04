"""
Racing line + quasi-steady-state speed solver.

Racing line
-----------
geometric_racing_line: the track centerline. This is still the Simulator
default, because it is the only line that has been shown to be no slower
than "do nothing."

shaped_racing_line (Phase 4, experimental): a curvature-based
outside-apex-outside offset, with a vehicle-specific apex shift from
power-to-weight and grip. It is smooth, stays in bounds, and differs by
car — but on the V1 rounded rectangle it is *slower* than the centerline.

Why: a real racing line is only faster if it *widens* the effective
corner radius (using the straights to pre-turn so the tightest point is
gentler than the road itself). This heuristic only adds a lateral wobble
on top of the existing curve, which concentrates curvature at the apex
instead of relieving it. Getting genuine radius-widening is Phase 5 (`line_mode="optimized"`):
CasADi chooses n(s) to cut lap time, then this speed solver re-scores
the path. Do not treat line_mode="shaped" as an improved lap time.

Speed profile solver (forward-backward pass)
---------------------------------------------
This is the standard quasi-steady-state (point-mass) lap simulation method
used by tools like OptimumLap and many published papers on lap time
simulation:

  1. Compute the pure cornering speed limit v_corner(s) at every point from
     track curvature and vehicle/tire grip (dynamics.corner_speed_limit).
  2. FORWARD PASS: starting from the global slowest point (so we never start
     "faster than possible"), march forward integrating the traction-limited
     acceleration limit, clipping to v_corner at each point:
         v[i+1] = min(v_corner[i+1], sqrt(v[i]^2 + 2*a_accel(v[i])*ds))
  3. BACKWARD PASS: march backward from the same start point integrating the
     braking-limited deceleration limit, clipping again:
         v[i] = min(v[i], sqrt(v[i+1]^2 + 2*a_brake(v[i+1])*ds))
     This enforces that the car can actually brake in time for upcoming
     corners -- a corner speed limit alone isn't enough, since a fast car
     must start braking earlier.
  4. Repeat forward+backward a few passes around the closed loop so the
     solution is self-consistent at the seam (start/end point).

This does not yet use full nonlinear trajectory optimization for the
*path* by default. Phase 5 (CasADi, `line_mode="optimized"`) solves for
an offset n(s) that can widen corners; the speed along that path is still
this forward-backward pass.
"""

import numpy as np
from src.physics.dynamics import corner_speed_limit, max_traction_accel, max_braking_decel
from src.vehicle.aero import RHO_AIR_SEA_LEVEL


def geometric_racing_line(track):
    """V1 placeholder racing line: the track centerline itself."""
    return {"s": track.s.copy(), "x": track.x.copy(), "y": track.y.copy(),
            "curvature": track.curvature.copy(),
            "offset": np.zeros_like(track.s)}


def apex_bias_m(vehicle) -> float:
    """
    Along-track apex shift in meters.

    Positive = later apex (high power-to-weight, straighten the exit).
    Negative = earlier apex (low power / high relative grip).
    """
    if vehicle is None:
        return 0.0
    pwr = vehicle.max_power / max(vehicle.mass, 1.0)  # W/kg
    pwr_ref = 170_000.0 / 1275.0  # subaru_brz.json: 170 kW / 1275 kg
    mu_ref = 1.15
    bias = 0.12 * (pwr - pwr_ref) - 10.0 * (vehicle.tire_mu - mu_ref)
    return float(np.clip(bias, -20.0, 20.0))


def _half_cosine_periodic(s, knots_s, knots_n, length):
    """C1-smooth interpolation of (s, n) knots on a closed loop."""
    s = np.mod(np.asarray(s, dtype=float), length)
    ks = np.mod(np.asarray(knots_s, dtype=float), length)
    kn = np.asarray(knots_n, dtype=float)
    order = np.argsort(ks, kind="mergesort")
    ks, kn = ks[order], kn[order]
    ks_ext = np.concatenate([ks - length, ks, ks + length])
    kn_ext = np.concatenate([kn, kn, kn])
    idx = np.searchsorted(ks_ext, s, side="right") - 1
    idx = np.clip(idx, 0, len(ks_ext) - 2)
    s0, s1 = ks_ext[idx], ks_ext[idx + 1]
    n0, n1 = kn_ext[idx], kn_ext[idx + 1]
    span = np.maximum(s1 - s0, 1e-12)
    t = np.clip((s - s0) / span, 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * t))
    return n0 + w * (n1 - n0)


def _path_segment_lengths(x, y):
    """Closed-loop segment lengths along the actual (x, y) polyline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)


def _periodic_first_second(y, ds_seg):
    """Periodic first/second derivatives of y wrt arc length.

    Spa's last sample sits ~5 cm from the first (s = length). Using that tiny
    wrap gap in a central difference makes a fake κ spike at start/finish.
    """
    y = np.asarray(y, dtype=float)
    ds = float(np.median(np.asarray(ds_seg, dtype=float)))
    ds = max(ds, 1e-6)
    n_s = (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * ds)
    n_ss = (np.roll(y, -1) - 2.0 * y + np.roll(y, 1)) / (ds ** 2)
    return n_s, n_ss


def line_from_offset(track, offset, edge_margin=0.5):
    """Build a racing-line dict by offsetting the centerline along its left normal."""
    offset = np.asarray(offset, dtype=float)
    max_n = max(track.width / 2.0 - edge_margin, 0.05 * track.width / 2.0)
    offset = np.clip(offset, -max_n, max_n)
    nx = -np.sin(track.heading)
    ny = np.cos(track.heading)
    x = track.x + offset * nx
    y = track.y + offset * ny
    # Frenet offset curvature: matches track.curvature exactly when n ≡ 0,
    # so QSS does not invent apex spikes from polyline heading.
    k0 = np.asarray(track.curvature, dtype=float)
    ds_seg = _path_segment_lengths(track.x, track.y)
    n_s, n_ss = _periodic_first_second(offset, ds_seg)
    k0_s, _ = _periodic_first_second(k0, ds_seg)
    A = 1.0 - offset * k0
    cross = A ** 2 * k0 + A * n_ss + 2.0 * n_s ** 2 * k0 + offset * n_s * k0_s
    speed = np.hypot(A, n_s)
    curvature = cross / np.maximum(speed ** 3, 1e-9)
    heading = np.unwrap(np.arctan2(
        A * np.sin(track.heading) + n_s * np.cos(track.heading),
        A * np.cos(track.heading) - n_s * np.sin(track.heading),
    ))
    return {
        "s": track.s.copy(),
        "x": x,
        "y": y,
        "curvature": curvature,
        "heading": heading,
        "offset": offset,
    }


def shaped_racing_line(track, vehicle=None, n_out_frac=0.70, n_in_frac=0.75,
                       transition_m=40.0, smooth_sigma_m=8.0, edge_margin=0.6):
    """
    Experimental outside-apex-outside offset of the centerline.

    Not faster than geometric_racing_line on the default track — see module
    docstring. Opt in via Simulator(..., line_mode="shaped").
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    s = track.s
    n = len(s)
    length = float(track.length)
    ds = float(np.median(np.diff(s))) if n > 1 else 1.0
    k = track.curvature
    k_abs = np.abs(k)

    half_w = track.width / 2.0
    max_n = max(half_w - edge_margin, 0.05 * half_w)
    n_out = n_out_frac * max_n
    n_in = n_in_frac * max_n
    bias = apex_bias_m(vehicle)

    peak_height = max(0.015, 0.30 * np.percentile(k_abs, 85))
    min_sep = max(int(12.0 / max(ds, 0.5)), 5)
    peaks, _ = find_peaks(k_abs, height=peak_height, distance=min_sep)

    if peaks.size == 0:
        line = geometric_racing_line(track)
        line["heading"] = track.heading.copy()
        return line

    knots_s = []
    knots_n = []
    trans = float(transition_m)
    # Shrink the transition if two corners are closer than 2*trans.
    if peaks.size >= 2:
        peak_s = np.sort(s[peaks])
        gaps = np.diff(np.concatenate([peak_s, peak_s[:1] + length]))
        trans = min(trans, 0.40 * float(np.min(gaps)))
        trans = max(trans, 8.0)

    for idx in peaks:
        sign = np.sign(k[idx])
        if sign == 0.0:
            sign = 1.0
        s_apex = float(s[idx])
        knots_s.extend([s_apex - trans, s_apex + bias, s_apex + trans])
        knots_n.extend([-sign * n_out, sign * n_in, -sign * n_out])

    offset = _half_cosine_periodic(s, knots_s, knots_n, length)

    sigma_samples = max(smooth_sigma_m / max(ds, 0.5), 1.0)
    offset = gaussian_filter1d(offset, sigma=sigma_samples, mode="wrap")
    return line_from_offset(track, offset, edge_margin=edge_margin)


def compute_speed_profile(track, vehicle, tire, line=None, safety_factor=1.0,
                          n_passes=3, rho=RHO_AIR_SEA_LEVEL, u_max=None):
    """
    Compute a vehicle-specific speed profile v(s) along a racing line.

    safety_factor / u_max
        Fraction of the friction circle the solver may use (Phase 7).
        Attack = 1.0. This is NOT a scale on speed: corner speed goes as
        sqrt(u_max). Combined-slip couples remaining long force to lateral
        demand so you cannot trail-brake or accelerate at u = 1 in a corner.
    """
    if line is None:
        line = geometric_racing_line(track)
    u_max = safety_factor if u_max is None else u_max

    s = line["s"]
    n = len(s)
    # Use the actual polyline spacing. Centerline `s` over-states the length
    # of a line that cuts a corner, which biased QSS against a real racing line.
    ds = _path_segment_lengths(line["x"], line["y"])
    curvature = line["curvature"]

    v_corner = np.array([
        corner_speed_limit(k, vehicle, tire, rho=rho, u_max=u_max)
        for k in curvature
    ])

    start_idx = int(np.argmin(v_corner))
    v = v_corner.copy()

    for _ in range(n_passes):
        order = (np.arange(n) + start_idx) % n
        for j in range(1, n):
            i_prev = order[j - 1]
            i_cur = order[j]
            seg_len = ds[i_prev]
            a_max = max_traction_accel(
                vehicle, tire, v[i_prev], rho=rho,
                curvature=curvature[i_prev], u_max=u_max,
            )
            v_reachable = np.sqrt(max(v[i_prev] ** 2 + 2 * a_max * seg_len, 0.0))
            v[i_cur] = min(v_corner[i_cur], v_reachable)

        for j in range(n - 1, -1, -1):
            i_cur = order[j]
            if i_cur == start_idx:
                continue
            i_next = order[(j + 1) % n]
            seg_len = ds[i_cur]
            a_max = max_braking_decel(
                vehicle, tire, v[i_next], rho=rho,
                curvature=curvature[i_next], u_max=u_max,
            )
            v_reachable = np.sqrt(max(v[i_next] ** 2 + 2 * a_max * seg_len, 0.0))
            v[i_cur] = min(v[i_cur], v_reachable)

    return {
        "s": s,
        "x": line["x"],
        "y": line["y"],
        "curvature": curvature,
        "v_corner_limit": v_corner,
        "v": v,
        "ds": ds,
        "offset": np.asarray(line.get("offset", np.zeros_like(s)), dtype=float),
        "u_max": float(u_max),
    }
