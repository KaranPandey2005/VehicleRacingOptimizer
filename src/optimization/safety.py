"""
Phase 7 safety model.

Driving modes are no longer a uniform scale on speed. Each mode is:

    J = w_time * lap_time + w_risk * wall_risk + w_tire * tire_stress

plus a hard friction-circle cap u_max and a larger edge margin for Safer
modes (the optimizer is not allowed as close to the walls).

    wall_risk    = mean( ( |n| / (width/2) )^2 )     in [0, 1]
    tire_stress  = mean( u^2 )                       in [0, ~u_max^2]

Attack minimizes lap time. Race/Safe trade seconds for lower u and more
distance from the painted edge.

This is still a point-mass QSS model. It is not a driver model, a crash
probability, or a thermal tire-life model.
"""

from dataclasses import dataclass
import numpy as np

from src.physics.dynamics import axle_grip
from src.vehicle.aero import RHO_AIR_SEA_LEVEL
from src.vehicle.tires import friction_utilization


@dataclass(frozen=True)
class DrivingMode:
    name: str
    u_max: float
    edge_margin: float
    w_time: float = 1.0
    w_risk: float = 0.0
    w_tire: float = 0.0


DRIVING_MODES = {
    "attack": DrivingMode("attack", u_max=1.00, edge_margin=0.50,
                          w_time=1.0, w_risk=0.0, w_tire=0.0),
    # Weights are in seconds-per-unit so J is in seconds and comparable
    # to lap time. Race still wants the fast line; Safe pays more to stay in.
    "race": DrivingMode("race", u_max=0.92, edge_margin=0.90,
                        w_time=1.0, w_risk=4.0, w_tire=6.0),
    "safe": DrivingMode("safe", u_max=0.75, edge_margin=1.60,
                        w_time=1.0, w_risk=12.0, w_tire=10.0),
}


def forces_along_profile(profile, vehicle, tire, rho=RHO_AIR_SEA_LEVEL):
    """Fx, Fy, F_max, u along a solved speed profile (point-mass)."""
    from src.optimization.lap_time import compute_telemetry

    tel = compute_telemetry(profile)
    v = np.asarray(profile["v"], dtype=float)
    fx = vehicle.mass * tel["a_lon"]
    fy = vehicle.mass * tel["a_lat"]
    f_f, f_r, n_f, n_r = axle_grip(
        vehicle, tire, v, a_lon=tel["a_lon"], rho=rho, u_max=1.0,
    )
    f_max = f_f + f_r
    n_load = n_f + n_r
    u = friction_utilization(fx, fy, f_max)
    return {
        "a_lon": tel["a_lon"],
        "a_lat": tel["a_lat"],
        "f_x": fx,
        "f_y": fy,
        "f_max": f_max,
        "u": u,
        "normal_load": n_load,
    }


def wall_risk(profile, track):
    """0 on the centerline, 1 if the path sits on a track edge."""
    offset = np.asarray(profile.get("offset", np.zeros_like(profile["s"])),
                        dtype=float)
    half = max(float(track.width) / 2.0, 1e-6)
    frac = np.clip(np.abs(offset) / half, 0.0, 1.0)
    return float(np.mean(frac ** 2))


def safety_metrics(profile, track, vehicle, tire, rho=RHO_AIR_SEA_LEVEL):
    forces = forces_along_profile(profile, vehicle, tire, rho=rho)
    u = forces["u"]
    risk = wall_risk(profile, track)
    stress = float(np.mean(u ** 2))
    return {
        **forces,
        "mean_utilization": float(np.mean(u)),
        "peak_utilization": float(np.max(u)),
        "wall_risk": risk,
        "tire_stress": stress,
    }


def objective_cost(lap_time, metrics, mode: DrivingMode) -> float:
    return (
        mode.w_time * float(lap_time)
        + mode.w_risk * float(metrics["wall_risk"])
        + mode.w_tire * float(metrics["tire_stress"])
    )
