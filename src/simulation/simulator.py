"""
Simulator: orchestrates track + vehicle + tire + weather into a single
run() that produces a speed profile, lap time, and summary telemetry.

Phase 7 driving modes use a friction-circle cap u_max plus the weighted
objective J = w_time * lap_time + w_risk * wall_risk + w_tire * tire_stress.
Attack is still minimum-time. Race/Safe leave grip and wall margin.

Phase 6 weather scales tire grip (dry/damp/wet) and air density from ambient
temperature. It does not model standing water, a wet line, or tire temp.
"""

from dataclasses import dataclass, replace
from src.optimization.racing_line import (
    geometric_racing_line, shaped_racing_line, compute_speed_profile,
)
from src.optimization.trajectory import optimized_racing_line
from src.optimization.lap_time import compute_lap_time, compute_telemetry
from src.optimization.safety import DRIVING_MODES, safety_metrics, objective_cost
from src.vehicle.tires import TireModel
from src.weather.weather import Weather

# Back-compat alias: old code treated this as a uniform speed scale.
DRIVING_MODE_SAFETY_FACTOR = {name: m.u_max for name, m in DRIVING_MODES.items()}


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
            f"Wall risk:       {t['wall_risk']:5.3f}   tire stress: {t['tire_stress']:5.3f}\n"
        )


class Simulator:
    def __init__(self, track, vehicle, tire: TireModel = None, mode: str = "attack",
                 line_mode: str = "centerline", weather: Weather = None):
        self.track = track
        self.vehicle = vehicle
        self.tire = tire or TireModel(mu=vehicle.tire_mu)
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
        # Weather grip stacks on the tire's own modifier (dry weather = ×1).
        return replace(
            self.tire,
            grip_modifier=self.tire.grip_modifier * self.weather.grip_modifier(),
        )

    def run(self) -> SimulationResult:
        tire = self._effective_tire()
        rho = self.weather.air_density()
        mode_cfg = DRIVING_MODES[self.mode]
        if self.line_mode == "shaped":
            line = shaped_racing_line(self.track, self.vehicle)
        elif self.line_mode == "optimized":
            line = optimized_racing_line(
                self.track, self.vehicle, tire,
                safety_factor=mode_cfg.u_max,
                edge_margin=mode_cfg.edge_margin,
                rho=rho,
                driving_mode=mode_cfg,
            )
        else:
            line = geometric_racing_line(self.track)
        profile = compute_speed_profile(
            self.track, self.vehicle, tire, line=line,
            safety_factor=mode_cfg.u_max, rho=rho,
        )
        lap_time = compute_lap_time(profile)
        telemetry = compute_telemetry(profile)
        metrics = safety_metrics(
            profile, self.track, self.vehicle, tire, rho=rho,
        )
        telemetry.update({
            "u": metrics["u"],
            "f_x": metrics["f_x"],
            "f_y": metrics["f_y"],
            "f_max": metrics["f_max"],
            "mean_utilization": metrics["mean_utilization"],
            "peak_utilization": metrics["peak_utilization"],
            "wall_risk": metrics["wall_risk"],
            "tire_stress": metrics["tire_stress"],
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
            grip_modifier=tire.effective_mu() / max(tire.mu, 1e-12),
            u_max=mode_cfg.u_max,
            objective=objective,
        )
