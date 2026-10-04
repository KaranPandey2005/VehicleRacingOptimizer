"""
Tire model — Phase 7: friction-circle grip with optional load sensitivity.

    F_max(N) = mu(N) * N
    mu(N)    = mu0 * grip_modifier * (N_ref / N)^lambda

lambda = 0 recovers a constant-mu tire. N_ref is the vehicle's static weight
(m*g), so 1g, zero-aero corners still match v = sqrt(mu0 * g * r).

Combined slip (GG circle): longitudinal and lateral force share one budget

    F_x^2 + F_y^2  <=  (u_max * F_max)^2

Deferred (not in this phase — needs slip states / a bicycle or similar):
  - Pacejka / Magic Formula (Fy vs slip angle, Fx vs slip ratio)
  - Temperature, wear, compound, camber, inflation
  - Front/rear split and longitudinal/lateral load transfer
  - Explicit combined-slip *curves* (this file only uses the circular envelope)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TireModel:
    mu: float = 1.2          # dry peak grip at N = N_ref (dimensionless)
    grip_modifier: float = 1.0  # surface condition (1.0 = dry baseline)
    # Fraction of mu lost as load rises. 0 = constant mu (V1). ~0.1–0.2 is a
    # typical lumped passenger-car/slick placeholder, not a fitted tire.
    load_sensitivity: float = 0.15

    def effective_mu(self) -> float:
        """Peak mu at the reference load (before load sensitivity)."""
        return self.mu * self.grip_modifier

    def mu_at_load(self, normal_load: float, n_ref: float) -> float:
        n = max(float(normal_load), 1.0)
        n_ref = max(float(n_ref), 1.0)
        return self.effective_mu() * (n_ref / n) ** self.load_sensitivity

    def max_force(self, normal_load: float, n_ref: float = None) -> float:
        """
        Maximum friction-circle radius (N) at a given vertical load.

        If n_ref is omitted, load sensitivity is skipped so existing
        `max_force(N) = mu * N` call sites stay exact.
        """
        n = max(float(normal_load), 0.0)
        if n_ref is None or abs(self.load_sensitivity) < 1e-12:
            return self.effective_mu() * n
        if n <= 0.0:
            return 0.0
        return self.mu_at_load(n, n_ref) * n


def combined_slip_fx(f_max, f_lat):
    """Longitudinal force still available after spending f_lat on cornering."""
    f_max = np.maximum(f_max, 0.0)
    f_lat = np.abs(f_lat)
    leftover = f_max * f_max - f_lat * f_lat
    if np.isscalar(leftover):
        return float(np.sqrt(leftover)) if leftover > 0.0 else 0.0
    return np.sqrt(np.maximum(leftover, 0.0))


def friction_utilization(f_x, f_y, f_max):
    """u = hypot(Fx, Fy) / F_max. 1.0 is the friction-circle limit."""
    f_max = np.maximum(f_max, 1.0)
    return np.hypot(f_x, f_y) / f_max
