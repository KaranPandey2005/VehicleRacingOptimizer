"""
Simulator: orchestrates track + vehicle + tire + weather into a single
run() that produces a speed profile, lap time, and summary telemetry.

Phase 7 driving modes use a Pacejka-ellipse cap u_max plus the weighted
objective J = w_time * lap_time + w_risk * wall_risk + w_tire * tire_stress.
Attack is still minimum-time. Race/Safe leave grip and wall margin.

Phase 6 weather: dry/damp scale tire grip uniformly; wet uses a spatial
μ(s,n) (rubber + puddles in wet_line.py) instead of the 0.65 tire scalar.
Air density comes from ambient temperature. Phase 10c uses ambient/track
temp for lumped tire heat. Track z(s) is used by QSS (grade and vertical
curvature); it is not banking.
"""

from dataclasses import dataclass, replace
import numpy as np
from src.optimization.racing_line import (
    geometric_racing_line, shaped_racing_line,
)
from src.optimization.trajectory import optimized_racing_line
from src.optimization.lap_time import compute_lap_time, compute_telemetry
from src.optimization.safety import DRIVING_MODES, safety_metrics, objective_cost
from src.vehicle.tires import TireModel
from src.vehicle.thermal import ThermalModel, qss_with_thermal
from src.weather.weather import Weather, Condition
from src.weather.wet_line import grip_along_line

# Back-compat alias: old code treated this as a uniform speed scale.
DRIVING_MODE_SAFETY_FACTOR = {name: m.u_max for name, m in DRIVING_MODES.items()}


def _reported_grip(tire, profile):
    """Mean surface scale along the driven line, times the tire's own modifier."""
    tire_scale = tire.effective_mu() / max(tire.mu, 1e-12)
    g = profile.get("grip_s")
    if g is None:
        return float(tire_scale)
    return float(np.mean(np.asarray(g, dtype=float))) * float(tire_scale)


@dataclass
class SimulationResult:
    profile: dict
    telemetry: dict
    lap_time: float
    mode: str
    vehicle_name: str
    track_name: str
    line_mode: str = "centerline"
    used_fallback: bool = False
    weather: str = "dry"
    grip_modifier: float = 1.0
    u_max: float = 1.0
    objective: float = 0.0

    def summary(self) -> str:
        t = self.telemetry
        line_note = self.line_mode
        if self.line_mode == "optimized" and self.used_fallback:
            line_note = "optimized → centerline fallback"
        return (
            f"--- {self.vehicle_name} on {self.track_name} "
            f"[{self.mode.upper()} / {line_note} / {self.weather.upper()}] ---\n"
            f"Lap time:        {self.lap_time:6.2f} s\n"
            f"Objective J:     {self.objective:6.2f} s-eq\n"
            f"Max speed:       {t['max_speed_kmh']:6.1f} km/h\n"
            f"Min corner spd:  {t['min_speed']*3.6:6.1f} km/h\n"
            f"Peak accel:      {t['peak_lon_accel_g']:5.2f} g\n"
            f"Peak braking:    {t['peak_lon_decel_g']:5.2f} g\n"
            f"Peak lateral:    {t['peak_lat_accel_g']:5.2f} g\n"
            f"Util. mean/peak: {t['mean_utilization']:5.2f} / {t['peak_utilization']:5.2f}"
            f"  (cap {self.u_max:.2f})\n"
            f"Peak slip α:     f {t['peak_slip_front_deg']:4.1f}° / "
            f"r {t['peak_slip_rear_deg']:4.1f}°\n"
            f"Peak tire T:     f {t.get('peak_t_front_c', 0.0):5.1f}°C / "
            f"r {t.get('peak_t_rear_c', 0.0):5.1f}°C\n"
            f"Wear index:      {t.get('wear_index', 0.0):6.4f}\n"
            f"Wall risk:       {t['wall_risk']:5.3f}   tire stress: {t['tire_stress']:5.3f}\n"
        )


class Simulator:
    def __init__(self, track, vehicle, tire: TireModel = None, mode: str = "attack",
                 line_mode: str = "centerline", weather: Weather = None):
        self.track = track
        self.vehicle = vehicle
        self.tire = tire or TireModel.from_vehicle(vehicle)
        self.thermal = ThermalModel.from_vehicle(vehicle)
        self.weather = weather or Weather()
        if mode not in DRIVING_MODES:
            raise ValueError(f"Unknown mode '{mode}', expected one of {list(DRIVING_MODES)}")
        if line_mode not in ("centerline", "shaped", "optimized"):
            raise ValueError("line_mode must be 'centerline', 'shaped', or 'optimized'")
        self.mode = mode
        # Default stays centerline so existing V1/V3 numbers stay comparable.
        # Use line_mode="optimized" for the Phase 5 CasADi line.
        self.line_mode = line_mode

    def _effective_tire(self) -> TireModel:
        # Dry/damp: uniform weather scale on the tire. Wet: spatial map
        # (wet_line.py) — do not also multiply the tire by 0.65.
        if self.weather.condition == Condition.WET:
            return replace(self.tire, grip_modifier=self.tire.grip_modifier)
        return replace(
            self.tire,
            grip_modifier=self.tire.grip_modifier * self.weather.grip_modifier(),
        )

    def run(self) -> SimulationResult:
        tire = self._effective_tire()
        rho = self.weather.air_density()
        mode_cfg = DRIVING_MODES[self.mode]
        grip_s = grip_along_line(self.track, self.weather)
        if self.line_mode == "shaped":
            line = shaped_racing_line(self.track, self.vehicle)
        elif self.line_mode == "optimized":
            line = optimized_racing_line(
                self.track, self.vehicle, tire,
                safety_factor=mode_cfg.u_max,
                edge_margin=mode_cfg.edge_margin,
                rho=rho,
                driving_mode=mode_cfg,
                thermal=self.thermal,
                t_amb_c=float(self.weather.ambient_temp_c),
                t0_c=float(self.weather.track_temp_c),
                grip_s=grip_s,
            )
        else:
            line = geometric_racing_line(self.track)
        t_amb = float(self.weather.ambient_temp_c)
        t0 = float(self.weather.track_temp_c)
        profile, _th = qss_with_thermal(
            self.track, self.vehicle, tire, self.thermal, line,
            mode_cfg.u_max, rho, t_amb, t0, grip_s=grip_s,
        )
        lap_time = compute_lap_time(profile)
        telemetry = compute_telemetry(profile)
        metrics = safety_metrics(
            profile, self.track, self.vehicle, tire, rho=rho,
        )
        thermal = self.thermal.integrate(
            profile["ds"], profile["v"],
            metrics["u_front"], metrics["u_rear"],
            metrics["n_front"], metrics["n_rear"],
            t_amb, t0, t0,
        )
        telemetry.update({
            "z": profile.get("z", np.zeros_like(profile["s"])),
            "sin_theta": profile.get("sin_theta", np.zeros_like(profile["s"])),
            "kappa_v": profile.get("kappa_v", np.zeros_like(profile["s"])),
            "u": metrics["u"],
            "u_front": metrics["u_front"],
            "u_rear": metrics["u_rear"],
            "f_x": metrics["f_x"],
            "f_y": metrics["f_y"],
            "f_max": metrics["f_max"],
            "alpha_f": metrics["alpha_f"],
            "alpha_r": metrics["alpha_r"],
            "steer": metrics["steer"],
            "beta": metrics["beta"],
            "mean_utilization": metrics["mean_utilization"],
            "peak_utilization": metrics["peak_utilization"],
            "peak_slip_front_deg": metrics["peak_slip_front_deg"],
            "peak_slip_rear_deg": metrics["peak_slip_rear_deg"],
            "wall_risk": metrics["wall_risk"],
            "tire_stress": metrics["tire_stress"],
            "t_front": thermal["t_front"],
            "t_rear": thermal["t_rear"],
            "wear_index": thermal["wear_index"],
            "peak_t_front_c": thermal["peak_t_front_c"],
            "peak_t_rear_c": thermal["peak_t_rear_c"],
            "mean_t_front_c": thermal["mean_t_front_c"],
            "mean_t_rear_c": thermal["mean_t_rear_c"],
        })
        objective = objective_cost(lap_time, metrics, mode_cfg)
        return SimulationResult(
            profile=profile,
            telemetry=telemetry,
            lap_time=lap_time,
            mode=self.mode,
            vehicle_name=self.vehicle.name,
            track_name=self.track.name,
            line_mode=self.line_mode,
            used_fallback=bool(line.get("used_fallback", False)),
            weather=self.weather.condition.value,
            grip_modifier=_reported_grip(tire, profile),
            u_max=mode_cfg.u_max,
            objective=objective,
        )
