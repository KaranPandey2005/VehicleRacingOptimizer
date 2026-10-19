"""
Core longitudinal/lateral dynamics limits used by the racing-line speed solver.

Phase 10a: front/rear normal loads (static + aero + long transfer) and
drivetrain-specific drive force from the torque curve.

Phase 10b: Pacejka peaks D_x, D_y on a bicycle. Lateral demand is split by
lever arms (yaw-moment balance), not by instantaneous load. Combined slip
is an ellipse so leftover drive/brake can use D_x ≠ D_y.

    N_f, N_r  = static split * cos(θ) + m v² κ_v + Cop*downforce ± m a_x h / L
    Grade gravity (-g sin θ) is applied in the QSS kinematic update, not here.
    D_y,i     = u_max * μ_y(N_i) * N_i
    D_x,i     = u_max * μ_x(N_i) * N_i
    F_y,f     = (b/L) * m v^2 |kappa|
    F_y,r     = (a/L) * m v^2 |kappa|
    leftover  = D_x * sqrt(1 - (F_y / D_y)^2)

Drive uses only the driven axle(s). Brakes use brake_bias_front vs leftover.

Cornering with lambda = 0 and no aero still matches v = sqrt(mu g r) at 1g.
"""

import numpy as np
from src.vehicle.aero import drag_force, downforce, RHO_AIR_SEA_LEVEL
from src.vehicle.bicycle import lever_arms, lateral_axle_forces
from src.vehicle.powertrain import max_engine_drive_force
from src.vehicle.tires import combined_slip_fx

G = 9.81  # m/s^2


def _broadcast_grade(cos_theta, kappa_v, like):
    if isinstance(like, np.ndarray) and like.ndim > 0:
        shape = like.shape
        cos_theta = np.broadcast_to(np.asarray(cos_theta, dtype=float), shape)
        kappa_v = np.broadcast_to(np.asarray(kappa_v, dtype=float), shape)
        return cos_theta, kappa_v
    return float(np.asarray(cos_theta).reshape(-1)[0]), float(np.asarray(kappa_v).reshape(-1)[0])


def normal_load(vehicle, v, rho=RHO_AIR_SEA_LEVEL, cos_theta=1.0, kappa_v=0.0):
    """Total vertical load (N): mg cos(θ) + aero + m v² κ_v."""
    cos_theta, kappa_v = _broadcast_grade(cos_theta, kappa_v, v)
    return (vehicle.mass * G * cos_theta
            + downforce(vehicle, v, rho=rho)
            + vehicle.mass * v * v * kappa_v)


def reference_load(vehicle):
    """Static weight used as total N_ref (N)."""
    return vehicle.mass * G


def static_axle_refs(vehicle):
    w = float(np.clip(vehicle.weight_dist_front, 0.05, 0.95))
    n = vehicle.mass * G
    return w * n, (1.0 - w) * n


def axle_loads(vehicle, v, a_lon=0.0, rho=RHO_AIR_SEA_LEVEL,
               cos_theta=1.0, kappa_v=0.0):
    """
    Front and rear normal loads (N).

    a_lon > 0 is acceleration (load to the rear). a_lon < 0 is braking.
    Grade: N uses mg cos(θ) plus m v² κ_v split like static weight.
    Path gravity g sin(θ) is not a tire force — QSS adds it to kinematics.
    """
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
        cos_theta, kappa_v = _broadcast_grade(cos_theta, kappa_v, v)
        n_static = vehicle.mass * G * cos_theta
        n_vert = vehicle.mass * v * v * kappa_v
        n_aero = aero_k * v * v
        n_f = w * (n_static + n_vert) + cop * n_aero
        n_r = (1.0 - w) * (n_static + n_vert) + (1.0 - cop) * n_aero
        transfer = vehicle.mass * a_lon * h / L
        return np.maximum(n_f - transfer, 0.0), np.maximum(n_r + transfer, 0.0)

    v = float(v)
    a_lon = float(a_lon)
    cos_theta, kappa_v = _broadcast_grade(cos_theta, kappa_v, v)
    n_static = vehicle.mass * G * cos_theta
    n_vert = vehicle.mass * v * v * kappa_v
    n_aero = aero_k * v * v
    transfer = vehicle.mass * a_lon * h / L
    n_f = w * (n_static + n_vert) + cop * n_aero - transfer
    n_r = (1.0 - w) * (n_static + n_vert) + (1.0 - cop) * n_aero + transfer
    if n_f < 0.0:
        n_f = 0.0
    if n_r < 0.0:
        n_r = 0.0
    return n_f, n_r


def _peak_at_load(mu0, n, n_ref, lam, u_max):
    if isinstance(n, np.ndarray) and n.ndim > 0:
        n = np.maximum(n, 0.0)
        if abs(lam) < 1e-12:
            return u_max * mu0 * n
        mu = mu0 * (n_ref / np.maximum(n, 1.0)) ** lam
        return u_max * mu * n
    n = float(n)
    if n <= 0.0:
        return 0.0
    if abs(lam) < 1e-12:
        return u_max * mu0 * n
    mu = mu0 * (n_ref / n) ** lam if n_ref > 1.0 else mu0
    return u_max * mu * n


def axle_peaks(vehicle, tire, v, a_lon=0.0, rho=RHO_AIR_SEA_LEVEL, u_max=1.0,
               cos_theta=1.0, kappa_v=0.0):
    """Per-axle Pacejka peaks Dx, Dy (N) and normal loads at the current load."""
    n_f, n_r = axle_loads(
        vehicle, v, a_lon=a_lon, rho=rho, cos_theta=cos_theta, kappa_v=kappa_v,
    )
    nref_f, nref_r = static_axle_refs(vehicle)
    u_max = float(u_max)
    lam = tire.load_sensitivity
    mu_y = tire.effective_mu()
    mu_x = tire.effective_mu_x()
    dy_f = _peak_at_load(mu_y, n_f, nref_f, lam, u_max)
    dy_r = _peak_at_load(mu_y, n_r, nref_r, lam, u_max)
    dx_f = _peak_at_load(mu_x, n_f, nref_f, lam, u_max)
    dx_r = _peak_at_load(mu_x, n_r, nref_r, lam, u_max)
    return dx_f, dy_f, dx_r, dy_r, n_f, n_r


def axle_grip(vehicle, tire, v, a_lon=0.0, rho=RHO_AIR_SEA_LEVEL, u_max=1.0,
              cos_theta=1.0, kappa_v=0.0):
    """Per-axle peak lateral force D_y (N) at the current load."""
    _dx_f, dy_f, _dx_r, dy_r, n_f, n_r = axle_peaks(
        vehicle, tire, v, a_lon=a_lon, rho=rho, u_max=u_max,
        cos_theta=cos_theta, kappa_v=kappa_v,
    )
    return dy_f, dy_r, n_f, n_r


def grip_force(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL, u_max=1.0, a_lon=0.0,
               cos_theta=1.0, kappa_v=0.0):
    """Sum of axle D_y (N). Bicycle cornering is limited by the first axle."""
    f_f, f_r, _nf, _nr = axle_grip(
        vehicle, tire, v, a_lon=a_lon, rho=rho, u_max=u_max,
        cos_theta=cos_theta, kappa_v=kappa_v,
    )
    return f_f + f_r


def _driven_fx_cap(vehicle, fx_f, fx_r, f_eng):
    kind = str(vehicle.drivetrain).upper()
    if kind == "FWD":
        return min(f_eng, fx_f)
    if kind == "AWD":
        split = float(np.clip(vehicle.awd_torque_front, 0.0, 1.0))
        return min(f_eng * split, fx_f) + min(f_eng * (1.0 - split), fx_r)
    return min(f_eng, fx_r)


def _leftover_axles(vehicle, tire, v, a_lon, curvature, rho, u_max,
                    cos_theta=1.0, kappa_v=0.0):
    dx_f, dy_f, dx_r, dy_r, _nf, _nr = axle_peaks(
        vehicle, tire, v, a_lon=a_lon, rho=rho, u_max=u_max,
        cos_theta=cos_theta, kappa_v=kappa_v,
    )
    f_lat = vehicle.mass * v * v * abs(curvature)
    fy_f, fy_r = lateral_axle_forces(vehicle, f_lat)
    return combined_slip_fx(dx_f, fy_f, dy_f), combined_slip_fx(dx_r, fy_r, dy_r)


def _corner_speed_constant_mu(k, vehicle, tire, rho, u_max, cos_theta=1.0,
                             kappa_v=0.0):
    """Closed form when load sensitivity is off and there is no aero (linear in v^2)."""
    mu = float(u_max) * tire.effective_mu()
    m = vehicle.mass
    aero_term = mu * 0.5 * rho * vehicle.downforce_coefficient * vehicle.frontal_area
    vert_term = mu * m * float(kappa_v)
    denom = m * k - aero_term - vert_term
    if denom <= 1e-9:
        return vehicle.top_speed
    v_sq = (mu * m * G * float(cos_theta)) / denom
    if v_sq <= 0:
        return vehicle.top_speed
    return float(min(np.sqrt(v_sq), vehicle.top_speed))


def corner_speed_limit(curvature, vehicle, tire, rho=RHO_AIR_SEA_LEVEL,
                       u_max=1.0, cos_theta=1.0, kappa_v=0.0):
    """
    Maximum speed (m/s) at which the car can hold a corner of given curvature
    (1/m) at utilization u_max (pure lateral, a_x = 0).

    Bicycle: the slower of the two axles to D_y. With no aero, constant mu,
    and a flat road both axles saturate together at v = sqrt(mu g r).
    Grade uses N = mg cos(θ) + m v² κ_v (crest unloads, dip compresses).
    """
    k = abs(curvature)
    if k < 1e-9:
        return vehicle.top_speed

    u_max = float(u_max)
    cos_theta = float(cos_theta)
    kappa_v = float(kappa_v)
    no_aero = abs(float(vehicle.downforce_coefficient)) < 1e-12
    if no_aero and abs(tire.load_sensitivity) < 1e-12:
        return _corner_speed_constant_mu(
            k, vehicle, tire, rho, u_max, cos_theta=cos_theta, kappa_v=kappa_v,
        )

    m = vehicle.mass

    def residual(vs):
        vs = max(vs, 0.0)
        v = np.sqrt(vs)
        dy_f, dy_r, _nf, _nr = axle_grip(
            vehicle, tire, v, a_lon=0.0, rho=rho, u_max=u_max,
            cos_theta=cos_theta, kappa_v=kappa_v,
        )
        fy_f, fy_r = lateral_axle_forces(vehicle, m * vs * k)
        return max(fy_f - dy_f, fy_r - dy_r)

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
                       curvature=0.0, u_max=1.0, cos_theta=1.0, kappa_v=0.0):
    """Tire+aero forward accel (m/s^2). QSS adds -g sin(θ) along the path."""
    f_eng = max_engine_drive_force(vehicle, v)
    f_drag = drag_force(vehicle, v, rho=rho)
    a = 0.0
    for _ in range(3):
        fx_f, fx_r = _leftover_axles(
            vehicle, tire, v, a, curvature, rho, u_max,
            cos_theta=cos_theta, kappa_v=kappa_v,
        )
        f_drive = _driven_fx_cap(vehicle, fx_f, fx_r, f_eng)
        a_new = max(f_drive - f_drag, 0.0) / vehicle.mass
        if abs(a_new - a) < 1e-4:
            return a_new
        a = 0.5 * a + 0.5 * a_new
    return a


def max_braking_decel(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL,
                      curvature=0.0, u_max=1.0, cos_theta=1.0, kappa_v=0.0):
    """Tire+aero braking magnitude (m/s^2). QSS adds -g sin(θ) along the path."""
    f_drag = drag_force(vehicle, v, rho=rho)
    bias = float(np.clip(vehicle.brake_bias_front, 0.05, 0.95))
    f_hyd_f = bias * vehicle.max_brake_force
    f_hyd_r = (1.0 - bias) * vehicle.max_brake_force
    a = 0.0
    for _ in range(3):
        fx_f, fx_r = _leftover_axles(
            vehicle, tire, v, -a, curvature, rho, u_max,
            cos_theta=cos_theta, kappa_v=kappa_v,
        )
        f_brake = min(f_hyd_f, fx_f) + min(f_hyd_r, fx_r)
        a_new = (f_brake + f_drag) / vehicle.mass
        if abs(a_new - a) < 1e-4:
            return a_new
        a = 0.5 * a + 0.5 * a_new
    return a
