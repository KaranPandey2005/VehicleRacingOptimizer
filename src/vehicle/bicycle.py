"""
Quasi-steady bicycle (two-axle) kinematics for the QSS lap solver.

Lateral force is split by the static lever arms so yaw moment is zero:

    a = (1 - w) L     CG to front axle
    b = w L           CG to rear axle
    Fy_f = (b / L) * m * ay
    Fy_r = (a / L) * m * ay

w is weight_dist_front. This is not a roll model: left/right load transfer
is lumped into each axle. The first axle to hit D_y limits the turn
(understeer if the front saturates first).

Steer and slip are reconstructed from Pacejka after the speed profile
exists. They do not feed back into the QSS limits.
"""

from __future__ import annotations

import numpy as np

from src.vehicle.tires import invert_magic_formula


def lever_arms(vehicle):
    """a (CG→front), b (CG→rear), wheelbase L. All meters."""
    L = float(vehicle.wheelbase) if vehicle.wheelbase > 0.5 else 0.5
    w = float(vehicle.weight_dist_front)
    if w < 0.05:
        w = 0.05
    elif w > 0.95:
        w = 0.95
    a = (1.0 - w) * L
    b = w * L
    return a, b, L


def lateral_axle_forces(vehicle, f_lat):
    """
    Front/rear lateral force (N) for a total lateral demand f_lat (N).

    Sign of f_lat is preserved (left-positive if ay = v^2 κ uses that).
    """
    a, b, L = lever_arms(vehicle)
    return f_lat * (b / L), f_lat * (a / L)


def split_longitudinal_force(vehicle, f_x):
    """
    Split a net longitudinal force across axles for telemetry.

    Drive uses the drivetrain; braking uses brake_bias_front. This is not
    the QSS leftover (that lives in dynamics._leftover_axles).
    """
    fx = np.asarray(f_x, dtype=float)
    kind = str(vehicle.drivetrain).upper()
    bias = float(np.clip(vehicle.brake_bias_front, 0.05, 0.95))
    awd = float(np.clip(vehicle.awd_torque_front, 0.0, 1.0))
    scalar = np.isscalar(f_x) or (isinstance(f_x, np.ndarray) and f_x.ndim == 0)
    fx = np.atleast_1d(fx)
    fx_f = np.zeros_like(fx)
    fx_r = np.zeros_like(fx)
    accel = fx >= 0.0
    brake = ~accel
    if kind == "FWD":
        fx_f[accel] = fx[accel]
    elif kind == "AWD":
        fx_f[accel] = awd * fx[accel]
        fx_r[accel] = (1.0 - awd) * fx[accel]
    else:
        fx_r[accel] = fx[accel]
    fx_f[brake] = bias * fx[brake]
    fx_r[brake] = (1.0 - bias) * fx[brake]
    if scalar:
        return float(fx_f[0]), float(fx_r[0])
    return fx_f, fx_r


def reconstruct_slips(tire, fy_f, fy_r, fx_f, fx_r, dy_f, dy_r, dx_f, dx_r):
    """Slip angle α (rad) and slip ratio κ from axle forces vs Pacejka D."""
    alpha_f = invert_magic_formula(fy_f, tire.By, tire.Cy, dy_f, tire.Ey, x_hi=0.40)
    alpha_r = invert_magic_formula(fy_r, tire.By, tire.Cy, dy_r, tire.Ey, x_hi=0.40)
    kappa_f = invert_magic_formula(fx_f, tire.Bx, tire.Cx, dx_f, tire.Ex, x_hi=0.30)
    kappa_r = invert_magic_formula(fx_r, tire.Bx, tire.Cx, dx_r, tire.Ex, x_hi=0.30)
    return alpha_f, alpha_r, kappa_f, kappa_r


def steer_and_beta(vehicle, kappa, alpha_f, alpha_r):
    """
    Path-tracking bicycle: r/v = κ, small-angle ISO.

        α_f = δ − β − a κ
        α_r = −β + b κ
    """
    a, b, _L = lever_arms(vehicle)
    beta = b * kappa - alpha_r
    delta = alpha_f + beta + a * kappa
    return delta, beta
