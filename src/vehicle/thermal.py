"""
Phase 10c: 3-zone tire temperature (surface / carcass / rim) and wear.

Not a fitted tire, not a 3D carcass mesh. Per axle, after the QSS speed
profile exists:

    Q          = k_heat * u^2 * N * v          (surface only)
    C_s dTs/dt = Q − (h0 + hv v)(Ts − T_amb) − k_sc (Ts − Tc)
    C_c dTc/dt = k_sc (Ts − Tc) − k_cr (Tc − Tr)
    C_r dTr/dt = k_cr (Tc − Tr) − h_rim (Tr − T_amb)
    μ(T)       = μ0 * clip(1 − 0.3 ((Ts − T_opt)/T_width)^2, mu_floor, 1)
    W          = ∫ k_wear * u^2 * N * v * (1 + 0.5 * hot) dt
    μ_wear     = clip(1 − W, mu_floor, 1)

Grip uses surface temperature. CasADi still scores one flying lap
(two-pass QSS). A stint is Simulator(n_laps>1) on that frozen line.

Coefficients are illustrative. Weather.track_temp_c starts all three zones.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_THERMAL_KEYS = (
    "mass_kg", "cp", "k_heat", "h0", "hv",
    "t_opt_c", "t_width_c", "mu_floor", "k_wear",
    "k_sc", "k_cr", "h_rim",
    "mass_frac_surface", "mass_frac_rim",
)


def _as_zone_start(t0):
    """Scalar → (Ts, Tc, Tr); length-3 stays a 3-tuple."""
    if t0 is None:
        raise ValueError("zone start temperature is required")
    if np.ndim(t0) == 0:
        t = float(t0)
        return t, t, t
    arr = np.asarray(t0, dtype=float).flatten()
    if arr.size == 1:
        t = float(arr[0])
        return t, t, t
    if arr.size < 3:
        t = float(arr[0])
        return t, t, t
    return float(arr[0]), float(arr[1]), float(arr[2])


@dataclass
class ThermalModel:
    mass_kg: float = 8.0          # total rubber+carcass+rim per axle (kg)
    cp: float = 1500.0            # J / (kg K)
    k_heat: float = 0.04          # surface generation; sized so Ts stays near T_opt
    h0: float = 40.0              # W/K, surface still-air cooling
    hv: float = 3.0               # W/K per (m/s), surface
    t_opt_c: float = 80.0         # peak-grip surface temperature
    t_width_c: float = 40.0       # |ΔT| where grip is ~0.70 before the floor
    mu_floor: float = 0.70
    k_wear: float = 5.0e-8        # W ~ few % grip drop per Attack rectangle lap
    k_sc: float = 22.0            # W/K surface ↔ carcass
    k_cr: float = 12.0            # W/K carcass ↔ rim
    h_rim: float = 6.0            # W/K rim → air
    mass_frac_surface: float = 0.20
    mass_frac_rim: float = 0.25

    def heat_capacity(self) -> float:
        return max(float(self.mass_kg), 0.5) * max(float(self.cp), 100.0)

    def _zone_caps(self):
        m = max(float(self.mass_kg), 0.5)
        fs = float(np.clip(self.mass_frac_surface, 0.05, 0.50))
        fr = float(np.clip(self.mass_frac_rim, 0.05, 0.50))
        if fs + fr > 0.90:
            fr = 0.90 - fs
        fc = 1.0 - fs - fr
        cp = max(float(self.cp), 100.0)
        return max(fs * m, 0.15) * cp, max(fc * m, 0.3) * cp, max(fr * m, 0.2) * cp

    def mu_factor(self, t_c):
        """Grip scale in [mu_floor, 1] from surface temperature (°C)."""
        t = np.asarray(t_c, dtype=float)
        width = max(float(self.t_width_c), 1.0)
        x = (t - float(self.t_opt_c)) / width
        factor = 1.0 - 0.3 * x * x
        out = np.clip(factor, float(self.mu_floor), 1.0)
        if np.ndim(out) == 0:
            return float(out)
        return out

    def grip_scale(self, t_front, t_rear):
        """Weaker of the two surfaces — bicycle is limited by the first to fade."""
        return np.minimum(self.mu_factor(t_front), self.mu_factor(t_rear))

    def wear_scale(self, wear):
        """μ_wear = clip(1 − W, mu_floor, 1). W is cumulative dimensionless wear."""
        w = float(np.clip(wear, 0.0, 10.0))
        return float(np.clip(1.0 - w, float(self.mu_floor), 1.0))

    def _step_axle(self, ts, tc, tr, q, v, dt, t_amb, cs, cc, cr):
        h = float(self.h0) + float(self.hv) * v
        k_sc = max(float(self.k_sc), 0.1)
        k_cr = max(float(self.k_cr), 0.1)
        h_rim = max(float(self.h_rim), 0.1)
        dts = (q - h * (ts - t_amb) - k_sc * (ts - tc)) / cs * dt
        dtc = (k_sc * (ts - tc) - k_cr * (tc - tr)) / cc * dt
        dtr = (k_cr * (tc - tr) - h_rim * (tr - t_amb)) / cr * dt
        ts = float(np.clip(ts + dts, -20.0, 220.0))
        tc = float(np.clip(tc + dtc, -20.0, 220.0))
        tr = float(np.clip(tr + dtr, -20.0, 220.0))
        return ts, tc, tr

    def integrate(self, ds, v, u_front, u_rear, n_front, n_rear,
                  t_amb_c, t0_front_c, t0_rear_c, wraps=2):
        """
        March 3-zone T along the closed lap. `wraps` extra loops start from
        the previous finish so T(s) is closer to a periodic hot lap.
        Wear is the last loop only (one lap).
        t0_* may be a scalar (all zones) or (Ts, Tc, Tr).
        """
        ds = np.asarray(ds, dtype=float)
        v = np.maximum(np.asarray(v, dtype=float), 0.5)
        u_f = np.asarray(u_front, dtype=float)
        u_r = np.asarray(u_rear, dtype=float)
        n_f = np.maximum(np.asarray(n_front, dtype=float), 0.0)
        n_r = np.maximum(np.asarray(n_rear, dtype=float), 0.0)
        t_amb = float(t_amb_c)
        width = max(float(self.t_width_c), 1.0)
        t_opt = float(self.t_opt_c)
        cs, cc, cr = self._zone_caps()
        k_heat = float(self.k_heat)
        k_wear = float(self.k_wear)

        def march(zf, zr):
            t_fs = np.empty(v.shape, dtype=float)
            t_fc = np.empty(v.shape, dtype=float)
            t_fr = np.empty(v.shape, dtype=float)
            t_rs = np.empty(v.shape, dtype=float)
            t_rc = np.empty(v.shape, dtype=float)
            t_rr = np.empty(v.shape, dtype=float)
            tsf, tcf, trf = zf
            tsr, tcr, trr = zr
            wear = 0.0
            for i in range(v.size):
                t_fs[i], t_fc[i], t_fr[i] = tsf, tcf, trf
                t_rs[i], t_rc[i], t_rr[i] = tsr, tcr, trr
                dt = ds[i] / v[i]
                q_f = k_heat * (u_f[i] ** 2) * n_f[i] * v[i]
                q_r = k_heat * (u_r[i] ** 2) * n_r[i] * v[i]
                tsf, tcf, trf = self._step_axle(
                    tsf, tcf, trf, q_f, v[i], dt, t_amb, cs, cc, cr,
                )
                tsr, tcr, trr = self._step_axle(
                    tsr, tcr, trr, q_r, v[i], dt, t_amb, cs, cc, cr,
                )
                hot_f = 1.0 + 0.5 * max(0.0, (tsf - t_opt) / width)
                hot_r = 1.0 + 0.5 * max(0.0, (tsr - t_opt) / width)
                wear += k_wear * dt * (
                    (u_f[i] ** 2) * n_f[i] * v[i] * hot_f
                    + (u_r[i] ** 2) * n_r[i] * v[i] * hot_r
                )
            return (
                t_fs, t_fc, t_fr, t_rs, t_rc, t_rr, wear,
                (tsf, tcf, trf), (tsr, tcr, trr),
            )

        zf = _as_zone_start(t0_front_c)
        zr = _as_zone_start(t0_rear_c)
        t_fs = t_fc = t_fr = t_rs = t_rc = t_rr = None
        wear = 0.0
        n_wrap = max(int(wraps), 1)
        for _ in range(n_wrap):
            (t_fs, t_fc, t_fr, t_rs, t_rc, t_rr, wear, zf, zr) = march(zf, zr)
        return {
            "t_front": t_fs,
            "t_rear": t_rs,
            "t_front_carcass": t_fc,
            "t_rear_carcass": t_rc,
            "t_front_rim": t_fr,
            "t_rear_rim": t_rr,
            "wear_index": float(wear),
            "peak_t_front_c": float(np.max(t_fs)),
            "peak_t_rear_c": float(np.max(t_rs)),
            "mean_t_front_c": float(np.mean(t_fs)),
            "mean_t_rear_c": float(np.mean(t_rs)),
            "peak_t_front_carcass_c": float(np.max(t_fc)),
            "peak_t_rear_carcass_c": float(np.max(t_rc)),
            "peak_t_front_rim_c": float(np.max(t_fr)),
            "peak_t_rear_rim_c": float(np.max(t_rr)),
            "end_front": zf,
            "end_rear": zr,
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
                     t_amb_c, t0_c, grip_s=None, wear=0.0,
                     t0_front=None, t0_rear=None, wraps=2):
    """
    Two-pass QSS: grip at the start state, then μ(T_surface(s)) from a
    3-zone heat march. Optional wear scales μ (stint). Stacked with wet grip_s.

    Returns (profile, thermal_dict) for the second pass. The racing-line
    optimizer scores this profile; Simulator may march further laps.
    """
    from src.optimization.racing_line import compute_speed_profile
    from src.optimization.safety import safety_metrics
    from src.weather.wet_line import as_grip_array

    t0 = float(t0_c)
    t_amb = float(t_amb_c)
    g = as_grip_array(grip_s, line)
    wscale = thermal.wear_scale(wear)
    zf = t0 if t0_front is None else t0_front
    zr = t0 if t0_rear is None else t0_rear
    ts_f, _tc_f, _tr_f = _as_zone_start(zf)
    ts_r, _tc_r, _tr_r = _as_zone_start(zr)
    profile = compute_speed_profile(
        track, vehicle, tire, line=line,
        safety_factor=safety_factor, rho=rho,
        u_scale=thermal.grip_scale(ts_f, ts_r) * g * wscale,
    )
    metrics = safety_metrics(
        profile, track, vehicle, tire, rho=rho,
    )
    th = thermal.integrate(
        profile["ds"], profile["v"],
        metrics["u_front"], metrics["u_rear"],
        metrics["n_front"], metrics["n_rear"],
        t_amb, zf, zr, wraps=wraps,
    )
    profile = compute_speed_profile(
        track, vehicle, tire, line=line,
        safety_factor=safety_factor, rho=rho,
        u_scale=thermal.grip_scale(th["t_front"], th["t_rear"]) * g * wscale,
    )
    profile["grip_s"] = g
    return profile, th
