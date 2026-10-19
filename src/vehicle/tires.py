"""
Tire model — Phase 10b: axle-equivalent Pacejka Magic Formula on a bicycle.

Pure slip (sine MF, Pacejka '94 teaching form):

    F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))

x is slip ratio κ (longitudinal) or slip angle α in radians (lateral).
D is the peak force at the current load; QSS uses D, not the B/C/E shape.

    D_y(N) = μ_y(N) * N
    D_x(N) = μ_x(N) * N
    μ(N)   = μ0 * grip_modifier * (N_ref / N)^lambda

Combined slip is an ellipse so Dx and Dy may differ:

    (Fx / Dx)^2 + (Fy / Dy)^2  <=  u_max^2

B, C, E reconstruct slip after the lap (telemetry). They do not change the
QSS speed limit, which is peak-limited.

Deferred: compound, camber, inflation, left/right transfer.
Temperature and wear live in thermal.py (Phase 10c), not in this MF.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Axle-equivalent teaching defaults (not a fitted tire).
_DEFAULT_BX = 12.0
_DEFAULT_CX = 1.65
_DEFAULT_EX = 0.0
_DEFAULT_BY = 10.0
_DEFAULT_CY = 1.30
_DEFAULT_EY = -0.5

_PACEJKA_KEYS = ("Bx", "Cx", "Ex", "By", "Cy", "Ey", "mu_x", "load_sensitivity")


@dataclass
class TireModel:
    mu: float = 1.2          # dry peak lateral mu at N = N_ref
    grip_modifier: float = 1.0  # surface condition (1.0 = dry baseline)
    # Fraction of mu lost as load rises. 0 = constant mu (V1). ~0.1–0.2 is a
    # typical lumped passenger-car/slick placeholder, not a fitted tire.
    load_sensitivity: float = 0.15
    # Pacejka shape (axle-equivalent). D comes from mu / mu_x at load.
    Bx: float = _DEFAULT_BX
    Cx: float = _DEFAULT_CX
    Ex: float = _DEFAULT_EX
    By: float = _DEFAULT_BY
    Cy: float = _DEFAULT_CY
    Ey: float = _DEFAULT_EY
    # Peak longitudinal mu at N_ref. None / omitted → same as lateral mu.
    mu_x: float = None

    def effective_mu(self) -> float:
        """Peak lateral mu at the reference load (before load sensitivity)."""
        return self.mu * self.grip_modifier

    def effective_mu_x(self) -> float:
        mx = self.mu if self.mu_x is None else float(self.mu_x)
        return mx * self.grip_modifier

    def mu_at_load(self, normal_load: float, n_ref: float) -> float:
        n = max(float(normal_load), 1.0)
        n_ref = max(float(n_ref), 1.0)
        return self.effective_mu() * (n_ref / n) ** self.load_sensitivity

    def mu_x_at_load(self, normal_load: float, n_ref: float) -> float:
        n = max(float(normal_load), 1.0)
        n_ref = max(float(n_ref), 1.0)
        return self.effective_mu_x() * (n_ref / n) ** self.load_sensitivity

    def max_force(self, normal_load: float, n_ref: float = None) -> float:
        """
        Peak lateral force D_y (N) at a given vertical load.

        If n_ref is omitted, load sensitivity is skipped so existing
        `max_force(N) = mu * N` call sites stay exact.
        """
        return self._peak(normal_load, n_ref, self.effective_mu())

    def max_fx(self, normal_load: float, n_ref: float = None) -> float:
        """Peak longitudinal force D_x (N) at a given vertical load."""
        return self._peak(normal_load, n_ref, self.effective_mu_x())

    def _peak(self, normal_load, n_ref, mu0) -> float:
        n = max(float(normal_load), 0.0)
        if n_ref is None or abs(self.load_sensitivity) < 1e-12:
            return mu0 * n
        if n <= 0.0:
            return 0.0
        mu = mu0 * (max(float(n_ref), 1.0) / n) ** self.load_sensitivity
        return mu * n

    def fy_pure(self, alpha, normal_load, n_ref=None):
        """Lateral force (N) vs slip angle α (rad) at this load."""
        d = self.max_force(normal_load, n_ref=n_ref)
        return magic_formula(alpha, self.By, self.Cy, d, self.Ey)

    def fx_pure(self, kappa, normal_load, n_ref=None):
        """Longitudinal force (N) vs slip ratio κ at this load."""
        d = self.max_fx(normal_load, n_ref=n_ref)
        return magic_formula(kappa, self.Bx, self.Cx, d, self.Ex)

    @classmethod
    def from_vehicle(cls, vehicle) -> "TireModel":
        kwargs = {"mu": float(vehicle.tire_mu)}
        raw = getattr(vehicle, "pacejka", None)
        if isinstance(raw, dict):
            for key in _PACEJKA_KEYS:
                if key in raw and raw[key] is not None:
                    kwargs[key] = float(raw[key])
        return cls(**kwargs)


def magic_formula(x, B, C, D, E):
    """Pacejka sine MF. x is slip (rad or ratio); D is the peak force."""
    x = np.asarray(x, dtype=float)
    bx = float(B) * x
    y = D * np.sin(C * np.arctan(bx - E * (bx - np.arctan(bx))))
    if np.isscalar(x) or x.ndim == 0:
        return float(y)
    return y


def invert_magic_formula(force, B, C, D, E, x_hi=0.40):
    """
    Smallest non-negative slip x (then re-signed) with MF(x) ≈ |force|.

    Saturates at x_hi if |force| exceeds the curve peak (QSS reconstruction).
    """
    force = np.asarray(force, dtype=float)
    d = np.asarray(D, dtype=float)
    scalar = force.ndim == 0 and d.ndim == 0
    force = np.atleast_1d(force.astype(float))
    d = np.abs(np.broadcast_to(np.atleast_1d(d.astype(float)), force.shape))
    sign = np.sign(force)
    sign = np.where(sign == 0.0, 1.0, sign)
    mag = np.abs(force)
    lo = np.zeros_like(mag)
    hi = np.full_like(mag, float(x_hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fmid = np.abs(magic_formula(mid, B, C, d, E))
        go_up = fmid < mag
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    x = 0.5 * (lo + hi)
    x = np.where(d < 1.0, 0.0, x)
    out = sign * x
    if scalar:
        return float(out[0])
    return out


def combined_slip_fx(f_x_peak, f_lat, f_y_peak=None):
    """
    Longitudinal force still available after spending f_lat on cornering.

    Ellipse: (Fx/Dx)^2 + (Fy/Dy)^2 <= 1. Two-argument calls keep the
    original friction-circle leftover.
    """
    dx = np.maximum(f_x_peak, 0.0)
    dy = dx if f_y_peak is None else np.maximum(f_y_peak, 0.0)
    fy = np.abs(f_lat)
    dy_safe = np.maximum(dy, 1e-12)
    ratio = fy / dy_safe
    leftover = dx * np.sqrt(np.maximum(1.0 - ratio * ratio, 0.0))
    leftover = np.where(np.asarray(dy) <= 0.0, 0.0, leftover)
    if np.ndim(leftover) == 0:
        return float(leftover)
    return leftover


def combined_slip_fy(f_y_peak, f_x, f_x_peak=None):
    """
    Lateral force still available after spending f_x on drive/brake.

    Inverse of combined_slip_fx on the same ellipse.
    """
    dy = np.maximum(f_y_peak, 0.0)
    dx = dy if f_x_peak is None else np.maximum(f_x_peak, 0.0)
    fx = np.abs(f_x)
    dx_safe = np.maximum(dx, 1e-12)
    ratio = fx / dx_safe
    leftover = dy * np.sqrt(np.maximum(1.0 - ratio * ratio, 0.0))
    leftover = np.where(np.asarray(dx) <= 0.0, 0.0, leftover)
    if np.ndim(leftover) == 0:
        return float(leftover)
    return leftover


def friction_utilization(f_x, f_y, f_x_peak, f_y_peak=None):
    """u = hypot(Fx/Dx, Fy/Dy). 1.0 is the combined-slip limit."""
    dx = np.maximum(f_x_peak, 1.0)
    dy = dx if f_y_peak is None else np.maximum(f_y_peak, 1.0)
    return np.hypot(np.asarray(f_x) / dx, np.asarray(f_y) / dy)
