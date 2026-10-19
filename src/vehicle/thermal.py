"""
Phase 10c: lumped front/rear tire temperature and a per-lap wear index.

Not a 3-zone carcass, not a fitted tire, not a stint of degrading laps.
One closed lap, after the QSS speed profile exists:

    C dT/dt = k_heat * u^2 * N * v  −  (h0 + hv v) (T − T_amb)
    dT/ds   = (1/v) dT/dt
    μ(T)    = μ0 * clip(1 − 0.3 ((T − T_opt)/T_width)^2, mu_floor, 1)
    wear    = ∫ k_wear * u^2 * N * v * (1 + 0.5 * hot) dt

QSS cannot carry T(s) inside the first speed solve. `qss_with_thermal`
does: solve at T0 → integrate T → re-solve with μ(T(s)). Simulator and
the greedy/CasADi racing-line cost both use that helper.

Coefficients are illustrative. Weather.track_temp_c is the start temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_THERMAL_KEYS = (
    "mass_kg", "cp", "k_heat", "h0", "hv",
    "t_opt_c", "t_width_c", "mu_floor", "k_wear",
)


@dataclass
class ThermalModel:
    mass_kg: float = 8.0          # lumped rubber+carcass per axle (kg)
    cp: float = 1500.0            # J / (kg K)
    k_heat: float = 0.18          # generation scale (illustrative)
    h0: float = 40.0              # W/K, still-air cooling
    hv: float = 3.0               # W/K per (m/s)
    t_opt_c: float = 80.0         # peak-grip temperature
    t_width_c: float = 40.0       # |ΔT| where grip is ~0.70 before the floor
    mu_floor: float = 0.70
    k_wear: float = 5.0e-9        # per-lap index scale

    def heat_capacity(self) -> float:
        return max(float(self.mass_kg), 0.5) * max(float(self.cp), 100.0)

    def mu_factor(self, t_c):
        """Grip scale in [mu_floor, 1] from carcass temperature (°C)."""
        t = np.asarray(t_c, dtype=float)
        width = max(float(self.t_width_c), 1.0)
        x = (t - float(self.t_opt_c)) / width
        factor = 1.0 - 0.3 * x * x
        out = np.clip(factor, float(self.mu_floor), 1.0)
        if np.ndim(out) == 0:
            return float(out)
        return out

    def grip_scale(self, t_front, t_rear):
        """Weaker of the two axles — bicycle is limited by the first to fade."""
        return np.minimum(self.mu_factor(t_front), self.mu_factor(t_rear))

    def integrate(self, ds, v, u_front, u_rear, n_front, n_rear,
                  t_amb_c, t0_front_c, t0_rear_c, wraps=2):
        """
        March T along the closed lap. `wraps` extra loops start from the
        previous finish so T(s) is closer to a periodic hot lap.
        Wear is the last loop only (one lap).
        """
        ds = np.asarray(ds, dtype=float)
        v = np.maximum(np.asarray(v, dtype=float), 0.5)
        u_f = np.asarray(u_front, dtype=float)
        u_r = np.asarray(u_rear, dtype=float)
        n_f = np.maximum(np.asarray(n_front, dtype=float), 0.0)
        n_r = np.maximum(np.asarray(n_rear, dtype=float), 0.0)
        C = self.heat_capacity()
        t_amb = float(t_amb_c)
        width = max(float(self.t_width_c), 1.0)
        t_opt = float(self.t_opt_c)

        def march(tf0, tr0):
            t_f = np.empty(v.shape, dtype=float)
            t_r = np.empty(v.shape, dtype=float)
            tf, tr = float(tf0), float(tr0)
            wear = 0.0
            for i in range(v.size):
                t_f[i] = tf
                t_r[i] = tr
                dt = ds[i] / v[i]
                h = float(self.h0) + float(self.hv) * v[i]
                q_f = float(self.k_heat) * (u_f[i] ** 2) * n_f[i] * v[i]
                q_r = float(self.k_heat) * (u_r[i] ** 2) * n_r[i] * v[i]
                tf = tf + (q_f - h * (tf - t_amb)) / C * dt
                tr = tr + (q_r - h * (tr - t_amb)) / C * dt
                tf = float(np.clip(tf, -20.0, 220.0))
                tr = float(np.clip(tr, -20.0, 220.0))
                hot_f = 1.0 + 0.5 * max(0.0, (tf - t_opt) / width)
                hot_r = 1.0 + 0.5 * max(0.0, (tr - t_opt) / width)
                wear += float(self.k_wear) * dt * (
                    (u_f[i] ** 2) * n_f[i] * v[i] * hot_f
                    + (u_r[i] ** 2) * n_r[i] * v[i] * hot_r
                )
            return t_f, t_r, wear, tf, tr

        tf, tr = float(t0_front_c), float(t0_rear_c)
        t_f = t_r = None
        wear = 0.0
        n_wrap = max(int(wraps), 1)
        for _ in range(n_wrap):
            t_f, t_r, wear, tf, tr = march(tf, tr)
        return {
            "t_front": t_f,
            "t_rear": t_r,
            "wear_index": float(wear),
            "peak_t_front_c": float(np.max(t_f)),
            "peak_t_rear_c": float(np.max(t_r)),
            "mean_t_front_c": float(np.mean(t_f)),
            "mean_t_rear_c": float(np.mean(t_r)),
        }

    @classmethod
    def from_vehicle(cls, vehicle) -> "ThermalModel":
        kwargs = {}
        raw = getattr(vehicle, "thermal", None)
        if isinstance(raw, dict):
            for key in _THERMAL_KEYS:
                if key in raw and raw[key] is not None:
                    kwargs[key] = float(raw[key])
        return cls(**kwargs)


def qss_with_thermal(track, vehicle, tire, thermal, line, safety_factor, rho,
                     t_amb_c, t0_c):
    """
    Two-pass QSS: cold/track-temp grip, then μ(T(s)) from a heat march.

    Returns (profile, thermal_dict) for the second pass. The racing-line
    optimizer scores this profile; Simulator re-integrates T on it for
    telemetry.
    """
    from src.optimization.racing_line import compute_speed_profile
    from src.optimization.safety import safety_metrics

    t0 = float(t0_c)
    t_amb = float(t_amb_c)
    profile = compute_speed_profile(
        track, vehicle, tire, line=line,
        safety_factor=safety_factor, rho=rho,
        u_scale=thermal.mu_factor(t0),
    )
    metrics = safety_metrics(
        profile, track, vehicle, tire, rho=rho,
    )
    th = thermal.integrate(
        profile["ds"], profile["v"],
        metrics["u_front"], metrics["u_rear"],
        metrics["n_front"], metrics["n_rear"],
        t_amb, t0, t0,
    )
    profile = compute_speed_profile(
        track, vehicle, tire, line=line,
        safety_factor=safety_factor, rho=rho,
        u_scale=thermal.grip_scale(th["t_front"], th["t_rear"]),
    )
    return profile, th
