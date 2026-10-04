"""
Phase 10a powertrain: torque curve × gears × final drive.

Wheel force at speed v is the best gear that keeps the engine between idle
and redline:

    rpm  = (v / r) * gear * final * 60 / (2π)
    F    = T(rpm) * gear * final * eta / r

If JSON omits a curve or ratios, they are synthesized from max_power and
top_speed so older vehicle files still run.

This is still quasi-steady-state. It is not a shift-time model, clutch
model, or engine inertia. Pacejka / slip states are later.
"""

from __future__ import annotations

import math
import numpy as np


def tire_radius_m(vehicle) -> float:
    return max(float(getattr(vehicle, "tire_radius", 0.31)), 0.15)


def _synthesize_torque_curve(vehicle):
    """
    Simple peak-torque then peak-power curve from max_power and redline.
    Shape is illustrative, not a mapped dyno.
    """
    rpm_red = max(float(vehicle.rpm_redline), 3000.0)
    rpm_idle = min(float(vehicle.rpm_idle), 0.4 * rpm_red)
    rpm_tq = 0.52 * rpm_red
    rpm_pwr = 0.90 * rpm_red
    omega_pwr = rpm_pwr * 2.0 * math.pi / 60.0
    t_pwr = float(vehicle.max_power) / max(omega_pwr, 1.0)
    t_peak = 1.18 * t_pwr
    return [
        [rpm_idle, 0.72 * t_peak],
        [rpm_tq, t_peak],
        [rpm_pwr, t_pwr],
        [rpm_red, 0.88 * t_pwr],
    ]


def torque_curve_points(vehicle):
    curve = getattr(vehicle, "torque_curve", None) or []
    if len(curve) >= 2:
        pts = np.asarray(curve, dtype=float)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            order = np.argsort(pts[:, 0])
            return pts[order, 0], pts[order, 1]
    pts = np.asarray(_synthesize_torque_curve(vehicle), dtype=float)
    return pts[:, 0], pts[:, 1]


def engine_torque_nm(vehicle, rpm: float) -> float:
    rpms, tqs = torque_curve_points(vehicle)
    rpm = float(np.clip(rpm, rpms[0], rpms[-1]))
    return float(np.interp(rpm, rpms, tqs))


def gear_ratios(vehicle):
    gears = getattr(vehicle, "gear_ratios", None) or []
    gears = [float(g) for g in gears if float(g) > 0.0]
    if gears:
        return sorted(gears, reverse=True)
    # Geometric 6-speed: 1st pulls from ~idle at 6 m/s, top gear near
    # redline at top_speed.
    r = tire_radius_m(vehicle)
    fd = max(float(vehicle.final_drive), 1.0)
    rpm_red = max(float(vehicle.rpm_redline), 3000.0)
    rpm_idle = max(float(vehicle.rpm_idle), 800.0)
    v_top = max(float(vehicle.top_speed), 20.0)
    g_top = (rpm_red * 2.0 * math.pi / 60.0) * r / (v_top * fd)
    g_1 = (rpm_idle * 2.0 * math.pi / 60.0) * r / (6.0 * fd)
    g_1 = max(g_1, 1.8 * g_top)
    n = 6
    ratio = (g_top / g_1) ** (1.0 / (n - 1))
    return [g_1 * (ratio ** i) for i in range(n)]


def engine_rpm(vehicle, v, gear: float) -> float:
    omega_w = max(float(v), 0.1) / tire_radius_m(vehicle)
    return omega_w * float(gear) * float(vehicle.final_drive) * 60.0 / (2.0 * math.pi)


def max_engine_drive_force(vehicle, v: float) -> float:
    """Best-gear tractive force (N) before tire grip is applied."""
    v_safe = max(float(v), 1.0)
    r = tire_radius_m(vehicle)
    eta = float(np.clip(vehicle.drivetrain_efficiency, 0.5, 1.0))
    fd = max(float(vehicle.final_drive), 1.0)
    rpm_idle = float(vehicle.rpm_idle)
    rpm_red = float(vehicle.rpm_redline)
    gears = gear_ratios(vehicle)
    best = 0.0
    for g in gears:
        rpm = engine_rpm(vehicle, v_safe, g)
        if rpm < 0.75 * rpm_idle:
            continue
        if rpm > rpm_red * 1.03:
            continue
        tq = engine_torque_nm(vehicle, min(rpm, rpm_red))
        force = tq * g * fd * eta / r
        if force > best:
            best = force
    if best <= 0.0:
        # Crawl in 1st or overspeed in top: still allow P/v.
        g = gears[0] if v_safe < 0.4 * float(vehicle.top_speed) else gears[-1]
        rpm = min(engine_rpm(vehicle, v_safe, g), rpm_red)
        best = engine_torque_nm(vehicle, rpm) * g * fd * eta / r
    best = min(best, float(vehicle.max_power) / v_safe)
    best = min(best, float(vehicle.max_traction_force))
    return max(best, 0.0)
