"""
Core longitudinal/lateral dynamics limits used by the racing-line speed solver.

Normal load:
    N(v) = m*g + F_downforce(v)

Friction-circle grip (Phase 7), with optional load sensitivity:
    mu(N)    = mu0 * (N_ref / N)^lambda
    F_max(v) = mu(N) * N
    N_ref    = m*g

Combined slip: Fx and Fy share F_max. After spending F_y = m v^2 |kappa| on
the corner, remaining drive/brake force is sqrt(F_max^2 - F_y^2). Driving
modes cap the circle at u_max * F_max (Attack = 1, Race < 1, Safe smaller).

Cornering speed at curvature kappa, lambda = 0, u_max = 1, no downforce:
    v_max = sqrt(mu0 * g / |kappa|)
which is the closed-form check used in tests.
"""

import numpy as np
from src.vehicle.aero import drag_force, downforce, RHO_AIR_SEA_LEVEL
from src.vehicle.tires import combined_slip_fx

G = 9.81  # m/s^2


def normal_load(vehicle, v, rho=RHO_AIR_SEA_LEVEL):
    """Total vertical load on the car (N): static weight + aero downforce."""
    return vehicle.mass * G + downforce(vehicle, v, rho=rho)


def reference_load(vehicle):
    """Static weight used as N_ref for load sensitivity (N)."""
    return vehicle.mass * G


def grip_force(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL, u_max=1.0):
    """Friction-circle radius (N) at speed v, including load sensitivity."""
    n = normal_load(vehicle, v, rho=rho)
    return float(u_max) * tire.max_force(n, n_ref=reference_load(vehicle))


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
    (1/m) at utilization u_max of the friction circle (pure lateral).
    """
    k = abs(curvature)
    if k < 1e-9:
        return vehicle.top_speed

    u_max = float(u_max)
    if abs(tire.load_sensitivity) < 1e-12:
        return _corner_speed_constant_mu(k, vehicle, tire, rho, u_max)

    m = vehicle.mass
    n_ref = reference_load(vehicle)
    c = 0.5 * rho * vehicle.downforce_coefficient * vehicle.frontal_area
    mu0 = u_max * tire.effective_mu()

    def residual(vs):
        vs = max(vs, 0.0)
        n = n_ref + c * vs
        mu = mu0 * (n_ref / max(n, 1.0)) ** tire.load_sensitivity
        return m * vs * k - mu * max(n, 0.0)

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
    v_safe = max(v, 1.0)
    f_power_limited = vehicle.max_power / v_safe
    f_drive = min(f_power_limited, vehicle.max_traction_force)

    f_max = grip_force(vehicle, tire, v, rho=rho, u_max=u_max)
    f_lat = vehicle.mass * v * v * abs(curvature)
    f_drive = min(f_drive, combined_slip_fx(f_max, f_lat))

    f_drag = drag_force(vehicle, v, rho=rho)
    f_net = f_drive - f_drag
    return max(f_net, 0.0) / vehicle.mass


def max_braking_decel(vehicle, tire, v, rho=RHO_AIR_SEA_LEVEL,
                      curvature=0.0, u_max=1.0):
    """Maximum braking deceleration magnitude (m/s^2) available at speed v."""
    f_max = grip_force(vehicle, tire, v, rho=rho, u_max=u_max)
    f_lat = vehicle.mass * v * v * abs(curvature)
    f_brake = min(combined_slip_fx(f_max, f_lat), vehicle.max_brake_force)
    f_drag = drag_force(vehicle, v, rho=rho)
    f_net = f_brake + f_drag
    return f_net / vehicle.mass
