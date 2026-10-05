"""
Phase 5: CasADi racing-line optimizer.

The path is a classic racing-line *shape* through each detected corner
(outside on entry, inside at the apex, outside on exit), with C1 half-cosine
blends. Every curvature peak is a decision corner. Amplitudes are chosen by
a greedy QSS search (keep a change only if the lap gets faster), then a
CasADi polish whenever CasADi is available.

The centerline is only kept if the solved line is not faster.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.optimization.racing_line import (
    geometric_racing_line, line_from_offset, compute_speed_profile,
    _half_cosine_periodic, apex_bias_m,
)
from src.optimization.lap_time import compute_lap_time
from src.optimization.safety import safety_metrics, objective_cost
from src.vehicle.aero import RHO_AIR_SEA_LEVEL


def casadi_available() -> bool:
    try:
        import casadi  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_corners(track, max_corners=None):
    """Return apex stations, turn sign (+ left), and transition length (m)."""
    s = track.s
    k = track.curvature
    length = float(track.length)
    ds = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    k_abs = np.abs(k)
    height = max(0.006, 0.15 * np.percentile(k_abs, 80))
    min_sep = max(int(35.0 / max(ds, 0.5)), 5)
    n = len(k_abs)
    pad = min_sep + 2
    k_ext = np.concatenate([k_abs[-pad:], k_abs, k_abs[:pad]])
    peaks_ext, props = find_peaks(k_ext, height=height, distance=min_sep)
    peaks = peaks_ext - pad
    keep = (peaks >= 0) & (peaks < n)
    peaks = peaks[keep]
    heights = props["peak_heights"][keep]
    # Deduplicate wrap-around detections of the same apex.
    if peaks.size:
        order = np.argsort(peaks)
        peaks, heights = peaks[order], heights[order]
        uniq_p, uniq_h = [int(peaks[0])], [float(heights[0])]
        for p, h in zip(peaks[1:], heights[1:]):
            if p - uniq_p[-1] < min_sep:
                if h > uniq_h[-1]:
                    uniq_p[-1], uniq_h[-1] = int(p), float(h)
            else:
                uniq_p.append(int(p))
                uniq_h.append(float(h))
        if len(uniq_p) > 1 and (uniq_p[0] + n - uniq_p[-1]) < min_sep:
            if uniq_h[0] >= uniq_h[-1]:
                uniq_p.pop()
                uniq_h.pop()
            else:
                uniq_p.pop(0)
                uniq_h.pop(0)
        peaks = np.array(uniq_p)
        heights = np.array(uniq_h)
    else:
        return []

    if max_corners is not None and peaks.size > max_corners:
        order = np.argsort(heights)[::-1][:max_corners]
        peaks = np.sort(peaks[order])

    peak_s = s[peaks].astype(float)
    signs = np.sign(k[peaks])
    signs[signs == 0.0] = 1.0

    if peak_s.size == 1:
        gaps = np.array([length])
    else:
        gaps = np.diff(np.concatenate([peak_s, peak_s[:1] + length]))
    # Split each gap between the previous exit and the next entry so knots
    # never interleave (that was folding Spa's 27 peaks into a tighter line).
    share = np.minimum(np.clip(0.38 * gaps, 12.0, 380.0), 0.40 * gaps)

    corners = []
    n_c = len(peaks)
    for i, idx in enumerate(peaks):
        tr_in = float(share[(i - 1) % n_c])
        tr_out = float(share[i])
        corners.append({
            "s": float(s[idx]),
            "sign": float(signs[i]),
            "trans": 0.5 * (tr_in + tr_out),
            "trans_in": tr_in,
            "trans_out": tr_out,
        })
    return corners


def _offset_from_z(track, z, corners, apex_shift=0.0):
    """z = [n_out_0, n_in_0, n_out_1, n_in_1, ...] magnitudes in meters."""
    z = np.asarray(z, dtype=float).flatten()
    knots_s = []
    knots_n = []
    for i, c in enumerate(corners):
        n_out = float(z[2 * i])
        n_in = float(z[2 * i + 1])
        sgn = c["sign"]
        sa = c["s"] + float(apex_shift)
        tr_in = float(c.get("trans_in", c["trans"]))
        tr_out = float(c.get("trans_out", c["trans"]))
        knots_s.extend([sa - tr_in, sa, sa + tr_out])
        knots_n.extend([-sgn * n_out, sgn * n_in, -sgn * n_out])
    offset = _half_cosine_periodic(track.s, knots_s, knots_n, track.length)
    ds = float(np.median(np.diff(track.s))) if len(track.s) > 1 else 1.0
    sigma = max(6.0 / max(ds, 0.5), 1.0)
    return gaussian_filter1d(offset, sigma=sigma, mode="wrap")


def _line_from_z(track, z, corners, edge_margin, apex_shift=0.0):
    n = _offset_from_z(track, z, corners, apex_shift=apex_shift)
    return line_from_offset(track, n, edge_margin=edge_margin)


def _score_profile(track, vehicle, tire, profile, driving_mode, rho):
    t = compute_lap_time(profile)
    if driving_mode is None or (
        driving_mode.w_risk == 0.0 and driving_mode.w_tire == 0.0
    ):
        return t, t
    metrics = safety_metrics(profile, track, vehicle, tire, rho=rho)
    return objective_cost(t, metrics, driving_mode), t


def _qss_lap(track, vehicle, tire, safety_factor, z, corners, edge_margin,
             apex_shift=0.0, rho=RHO_AIR_SEA_LEVEL, driving_mode=None):
    line = _line_from_z(track, z, corners, edge_margin, apex_shift=apex_shift)
    profile = compute_speed_profile(
        track, vehicle, tire, line=line, safety_factor=safety_factor, rho=rho,
    )
    cost, t = _score_profile(track, vehicle, tire, profile, driving_mode, rho)
    return cost, t, line


def _keep_callback(owner, name, cb):
    # CasADi Callbacks are destroyed if Python drops the last reference
    # before IPOPT finishes (and we reuse them after solve for eval).
    setattr(owner, name, cb)


def _solve_qss(track, vehicle, tire, safety_factor, n_max, edge_margin,
               corners, apex_shift, z0, verbose=False, rho=RHO_AIR_SEA_LEVEL,
               driving_mode=None):
    import casadi as ca

    n_z = 2 * len(corners)
    z0 = np.clip(np.asarray(z0, dtype=float).flatten(), 0.0, n_max)

    class QssLap(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct(
                "qss_lap",
                {"enable_fd": True, "enable_forward": False, "enable_reverse": False,
                 "fd_method": "forward",
                 "fd_options": {"h": 5e-2}},
            )

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, _i):
            return ca.Sparsity.dense(n_z, 1)

        def get_sparsity_out(self, _i):
            return ca.Sparsity.dense(1, 1)

        def eval(self, args):
            z = np.array(args[0]).flatten()
            cost, _t, _line = _qss_lap(
                track, vehicle, tire, safety_factor, z, corners, edge_margin,
                apex_shift, rho=rho, driving_mode=driving_mode,
            )
            return [float(cost)]

    cb = QssLap()
    _keep_callback(_solve_qss, "_cb", cb)

    opti = ca.Opti()
    z = opti.variable(n_z)
    opti.minimize(cb(z))
    opti.subject_to(opti.bounded(0.0, z, n_max))
    opti.set_initial(z, z0)

    opts = {
        "ipopt.print_level": 3 if verbose else 0,
        "print_time": verbose,
        "ipopt.max_iter": 25 if n_z <= 12 else 8,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 4,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.limited_memory_max_history": 8,
        "ipopt.sb": "yes",
    }
    opti.solver("ipopt", opts)
    try:
        sol = opti.solve()
        z_val = np.array(sol.value(z)).flatten()
    except Exception:
        z_val = np.array(opti.debug.value(z)).flatten()
        if verbose:
            print("IPOPT (QSS) did not declare success:",
                  opti.stats().get("return_status"),
                  "using last iterate")
    return np.clip(z_val, 0.0, n_max)


def _greedy_corners_qss(track, vehicle, tire, safety_factor, n_max, edge_margin,
                        corners, apex_shift, verbose=False, rho=RHO_AIR_SEA_LEVEL,
                        driving_mode=None):
    """
    Visit every detected corner (strongest |κ| first). Try a small grid of
    outside/inside amplitudes and keep a change only if the objective improves.

    Attack (w_risk = w_tire = 0) is pure lap time. Race/Safe can refuse a
    wider line if wall risk / tire stress outweigh the seconds saved.
    """
    n_z = 2 * len(corners)
    z = np.zeros(n_z)
    best_cost, _t, _ = _qss_lap(
        track, vehicle, tire, safety_factor, z, corners, edge_margin, apex_shift,
        rho=rho, driving_mode=driving_mode,
    )
    strength = []
    for c in corners:
        idx = int(np.argmin(np.abs(track.s - c["s"])))
        strength.append(abs(float(track.curvature[idx])))
    order = np.argsort(strength)[::-1]
    grid = n_max * np.array([0.0, 0.40, 0.75, 1.00])
    if verbose:
        print(f"Phase 5: greedy QSS over {len(corners)} corners "
              f"({len(grid) ** 2} trials each).")
    for i in order:
        pair = (float(z[2 * i]), float(z[2 * i + 1]))
        local = best_cost
        for a in grid:
            for b in grid:
                z[2 * i] = a
                z[2 * i + 1] = b
                cost, _t, _ = _qss_lap(
                    track, vehicle, tire, safety_factor, z, corners,
                    edge_margin, apex_shift, rho=rho, driving_mode=driving_mode,
                )
                if cost < local - 1e-4:
                    local = cost
                    pair = (float(a), float(b))
        z[2 * i], z[2 * i + 1] = pair
        if verbose and local < best_cost - 1e-4:
            print(f"  s={corners[i]['s']:.0f} m  n_out={pair[0]:.2f}  "
                  f"n_in={pair[1]:.2f}  J {best_cost:.2f} -> {local:.2f}")
        best_cost = local
    return z


def optimized_racing_line(track, vehicle, tire, safety_factor=1.0,
                          edge_margin=0.5, verbose=False, rho=RHO_AIR_SEA_LEVEL,
                          driving_mode=None):
    """
    Racing line under point-mass QSS physics. Attack minimizes lap time;
    Race/Safe minimize J = w_t*t + w_r*risk + w_s*tire_stress.
    """
    n_max = max(track.width / 2.0 - edge_margin, 0.05 * track.width / 2.0)
    center = geometric_racing_line(track)
    center["solver_ok"] = False
    center["used_fallback"] = True
    t_cen = compute_lap_time(compute_speed_profile(
        track, vehicle, tire, line=center, safety_factor=safety_factor, rho=rho,
    ))
    center["centerline_lap_time"] = t_cen
    center["fine_lap_time"] = t_cen

    corners = _detect_corners(track)
    if not corners:
        if verbose:
            print("Phase 5: no corners detected; using centerline.")
        return center
    center["n_corners"] = len(corners)

    apex_shift = apex_bias_m(vehicle)
    n_z = 2 * len(corners)
    if verbose:
        print(f"Phase 5: optimizing {len(corners)} corners "
              f"({n_z} decision variables).")

    try:
        z = _greedy_corners_qss(
            track, vehicle, tire, safety_factor, n_max, edge_margin,
            corners, apex_shift, verbose=verbose, rho=rho,
            driving_mode=driving_mode,
        )
        cost_g, t_g, _ = _qss_lap(
            track, vehicle, tire, safety_factor, z, corners, edge_margin,
            apex_shift, rho=rho, driving_mode=driving_mode,
        )
        if casadi_available():
            z_p = _solve_qss(
                track, vehicle, tire, safety_factor, n_max, edge_margin,
                corners, apex_shift, z, verbose=verbose, rho=rho,
                driving_mode=driving_mode,
            )
            cost_p, t_p, _ = _qss_lap(
                track, vehicle, tire, safety_factor, z_p, corners, edge_margin,
                apex_shift, rho=rho, driving_mode=driving_mode,
            )
            if cost_p <= cost_g + 1e-3:
                z = z_p
                t_g = t_p
    except Exception as exc:
        if verbose:
            print(f"Phase 5 NLP failed ({exc}); using centerline.")
        return center

    cost_opt, t_opt, line = _qss_lap(
        track, vehicle, tire, safety_factor, z, corners, edge_margin, apex_shift,
        rho=rho, driving_mode=driving_mode,
    )
    line["solver_ok"] = True
    line["used_fallback"] = False
    line["fine_lap_time"] = t_opt
    line["centerline_lap_time"] = t_cen
    line["objective_cost"] = cost_opt
    line["n_corners"] = len(corners)
    line["coeffs"] = z

    prof_c = compute_speed_profile(
        track, vehicle, tire, line=center, safety_factor=safety_factor, rho=rho,
    )
    cost_cen, _ = _score_profile(
        track, vehicle, tire, prof_c, driving_mode, rho,
    )
    if cost_opt > cost_cen + 0.02:
        print(
            f"Phase 5: no better objective than centerline "
            f"(J {cost_opt:.2f} vs {cost_cen:.2f}); keeping centerline."
        )
        center["solver_ok"] = True
        center["used_fallback"] = True
        return center

    return line
