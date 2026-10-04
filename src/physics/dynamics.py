"""
Core longitudinal/lateral dynamics limits used by the racing-line speed solver.

Phase 10a: front/rear normal loads (static + aero + long transfer) and
drivetrain-specific drive force from the torque curve.

    N_f, N_r  = static split + Cop*downforce ± m a_x h / L
    F_max,i   = u_max * mu(N_i) * N_i     (per axle, load sensitive)
    F_y,i     = (N_i / N) * m v^2 |kappa|  (lateral split by load)
    leftover  = sqrt(F_max,i^2 - F_y,i^2)

Drive uses only the driven axle(s). Brakes use brake_bias_front vs leftover.

Cornering with lambda = 0 still matches v = sqrt(mu g r) at 1g, no aero.
"""

import numpy as np
from src.vehicle.aero import drag_force, downforce, RHO_AIR_SEA_LEVEL
from src.vehicle.powertrain import max_engine_drive_force

G = 9.81  # m/s^2


def normal_load(vehicle, v, rho=RHO_AIR_SEA_LEVEL):
    """Total vertical load on the car (N): static weight + aero downforce."""
    return vehicle.mass * G + downforce(vehicle, v, rho=rho)


def reference_load(vehicle):
    """Static weight used as total N_ref (N)."""
    return vehicle.mass * G


def static_axle_refs(vehicle):
    w = float(np.clip(vehicle.weight_dist_front, 0.05, 0.95))
    n = vehicle.mass * G
    return w * n, (1.0 - w) * n


def axle_loads(vehicle, v, a_lon=0.0, rho=RHO_AIR_SEA_LEVEL):
    """
    Front and rear normal loads (N).

    a_lon > 0 is acceleration (load to the rear). a_lon < 0 is braking.
    """
    n_static = vehicle.mass * G
    w = vehicle.weight_dist_front
    if w < 0.05:
        w = 0.05
    elif w > 0.95:
        w = 0.95
    cop = vehicle.center_of_pressure
    if cop < 0.05:
        cop = 0.05
    elif cop > 0.95:
        cop = 0.95
    L = vehicle.wheelbase if vehicle.wheelbase > 0.5 else 0.5
    h = vehicle.cg_height if vehicle.cg_height > 0.05 else 0.05
    aero_k = 0.5 * rho * vehicle.downforce_coefficient * vehicle.frontal_area

    if isinstance(v, np.ndarray) and v.ndim > 0:
        a_lon = np.asarray(a_lon, dtype=float)
        n_aero = aero_k * v * v
        n_f = w * n_static + cop * n_aero
        n_r = (1.0 - w) * n_static + (1.0 - cop) * n_aero
        transfer = vehicle.mass * a_lon * h / L
        return np.maximum(n_f - transfer, 0.0), np.maximum(n_r + transfer, 0.0)

    v = float(v)
    a_lon = float(a_lon)
    n_aero = aero_k * v * v
    transfer = vehicle.mass * a_lon * h / L
    n_f = w * n_static + cop * n_aero - transfer
    n_r = (1.0 - w) * n_static + (1.0 - cop) * n_aero + transfer
    if n_f < 0.0:
        n_f = 0.0
    if n_r < 0.0:
        n_r = 0.0
    return n_f, n_r


def _axle_fmax(tire, n, n_ref, u_max):
    if n <= 0.0:
        return 0.0
    mu0 = tire.mu * tire.grip_modifier
    lam = tire.load_sensitivity
    if abs(lam) < 1e-12:
        return u_max * mu0 * n
    mu = mu0 * (n_ref / n) ** lam if n_ref > 1.0 else mu0
    return u_max * mu * n


def axle_grip(vehicle, tire, v, a_lon=0.0, rho=RHO_AIR_SEA_LEVEL, u_max=1.0):
    """Per-axle friction-circle radii (N) at the current load."""
    n_f, n_r = axle_loads(vehicle, v, a_lon=a_lon, rho=rho)
    nref_f, nref_r = static_axle_refs(vehicle)
    u_max = float(u_max)
    if isinstance(n_f, np.ndarray) and n_f.ndim > 0:
        mu0 = tire.effective_mu()
        lam = tire.load_sensitivity
        n_f = np.maximum(n_f, 0.0)
        n_r = np.maximum(n_r, 0.0)
        if abs(lam) < 1e-12:
            return u_max * mu0 * n_f, u_max * mu0 * n_r, n_f, n_r
        mu_f = mu0 * (nref_f / np.maximum(n_f, 1.0)) ** lam
        mu_r = mu0 * (nref_r / np.maximum(n_r, 1.0)) ** lam
        return u_max * mu_f * n_f, u_max * mu_r * n_r, n_f, n_r
    return (
        _axle_fmax(tire, n_f, nref_f, u_max),
        _axle_fmax(tire, n_r, nref_r, u_max),
        n_f,
        n_r,
    )


def grip_force(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL, u_max=1.0, a_lon=0.0):
    """Total friction-circle radius (N), sum of axles."""
    f_f, f_r, _nf, _nr = axle_grip(vehicle, tire, v, a_lon=a_lon, rho=rho, u_max=u_max)
    return f_f + f_r


def _driven_fx_cap(vehicle, fx_f, fx_r, f_eng):
    kind = str(vehicle.drivetrain).upper()
    if kind == "FWD":
        return min(f_eng, fx_f)
    if kind == "AWD":
        split = float(np.clip(vehicle.awd_torque_front, 0.0, 1.0))
        return min(f_eng * split, fx_f) + min(f_eng * (1.0 - split), fx_r)
    return min(f_eng, fx_r)


def _leftover_axles(vehicle, tire, v, a_lon, curvature, rho, u_max):
    f_f, f_r, n_f, n_r = axle_grip(
        vehicle, tire, v, a_lon=a_lon, rho=rho, u_max=u_max,
    )
    f_lat = vehicle.mass * v * v * abs(curvature)
    n_tot = n_f + n_r
    if n_tot < 1.0:
        n_tot = 1.0
    fy_f = f_lat * n_f / n_tot
    fy_r = f_lat * n_r / n_tot
    def leftover(fmax, fy):
        rem = fmax * fmax - fy * fy
        return rem ** 0.5 if rem > 0.0 else 0.0
    return leftover(f_f, fy_f), leftover(f_r, fy_r)


def _corner_speed_constant_mu(k, vehicle, tire, rho, u_max):
    """Closed form when load sensitivity is off (linear in v^2)."""
    mu = float(u_max) * tire.effective_mu()
    m = vehicle.mass
    aero_term = mu * 0.5 * rho * vehicle.downforce_coefficient * vehicle.frontal_area
    denom = m * k - aero_term
    if denom <= 1e-9:
        return vehicle.top_speed
    v_sq = (mu * m * G) / denom
    if v_sq <= 0:
        return vehicle.top_speed
    return float(min(np.sqrt(v_sq), vehicle.top_speed))


def corner_speed_limit(curvature, vehicle, tire, rho=RHO_AIR_SEA_LEVEL,
                       u_max=1.0):
    """
    Maximum speed (m/s) at which the car can hold a corner of given curvature
    (1/m) at utilization u_max of the friction circle (pure lateral, a_x = 0).
    """
    k = abs(curvature)
    if k < 1e-9:
        return vehicle.top_speed

    u_max = float(u_max)
    if abs(tire.load_sensitivity) < 1e-12:
        return _corner_speed_constant_mu(k, vehicle, tire, rho, u_max)

    m = vehicle.mass

    def residual(vs):
        vs = max(vs, 0.0)
        v = np.sqrt(vs)
        f_f, f_r, _nf, _nr = axle_grip(
            vehicle, tire, v, a_lon=0.0, rho=rho, u_max=u_max,
        )
        return m * vs * k - (f_f + f_r)

    vs_hi = vehicle.top_speed ** 2
    r0 = residual(0.0)
    r1 = residual(vs_hi)
    if r0 >= 0:
        return 0.0
    if r1 < 0:
        return vehicle.top_speed
    lo, hi = 0.0, vs_hi
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(0.5 * (lo + hi)))


def max_traction_accel(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL,
                       curvature=0.0, u_max=1.0):
    """Maximum forward acceleration (m/s^2) available at speed v (v > 0)."""
    f_eng = max_engine_drive_force(vehicle, v)
    f_drag = drag_force(vehicle, v, rho=rho)
    a = 0.0
    for _ in range(3):
        fx_f, fx_r = _leftover_axles(
            vehicle, tire, v, a, curvature, rho, u_max,
        )
        f_drive = _driven_fx_cap(vehicle, fx_f, fx_r, f_eng)
        a_new = max(f_drive - f_drag, 0.0) / vehicle.mass
        if abs(a_new - a) < 1e-4:
            return a_new
        a = 0.5 * a + 0.5 * a_new
    return a


def max_braking_decel(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL,
                      curvature=0.0, u_max=1.0):
    """Maximum braking deceleration magnitude (m/s^2) available at speed v."""
    f_drag = drag_force(vehicle, v, rho=rho)
    bias = float(np.clip(vehicle.brake_bias_front, 0.05, 0.95))
    f_hyd_f = bias * vehicle.max_brake_force
    f_hyd_r = (1.0 - bias) * vehicle.max_brake_force
    a = 0.0
    for _ in range(3):
        fx_f, fx_r = _leftover_axles(
            vehicle, tire, v, -a, curvature, rho, u_max,
        )
        f_brake = min(f_hyd_f, fx_f) + min(f_hyd_r, fx_r)
        a_new = (f_brake + f_drag) / vehicle.mass
        if abs(a_new - a) < 1e-4:
            return a_new
        a = 0.5 * a + 0.5 * a_new
    return a
