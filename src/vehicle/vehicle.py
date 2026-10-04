"""
Vehicle model.

Holds configurable physical parameters for a vehicle and derives simple,
physically-motivated forces from them (engine traction force vs. speed,
braking force limit). Values are NOT claimed to represent any real vehicle
unless explicitly validated against manufacturer/test data -- see README.

V1 simplifications (documented, still later work):
  - Engine modeled as a single max-power figure with a traction-limited floor
    at low speed, rather than a full torque curve x gear ratio x efficiency
    chain.
  - Braking modeled as a single max deceleration-capable force, not per-axle
    brake bias / ABS dynamics.
  - Tires: friction-circle + load sensitivity (Phase 7). Pacejka, temperature,
    wear, camber, and axle split are not modeled.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict


@dataclass
class Vehicle:
    name: str = "Subaru BRZ (2022+, unvalidated ballpark)"

    # Mass & geometry (default = BRZ ballpark; see data/vehicles/)
    mass: float = 1275.0             # kg
    wheelbase: float = 2.575         # m
    track_width: float = 1.52        # m
    cg_height: float = 0.50          # m
    weight_dist_front: float = 0.53  # fraction of mass on front axle (0-1)

    # Powertrain (simplified V1: single max-power figure)
    max_power: float = 170_000.0     # W
    max_traction_force: float = 9500.0  # N, limited by grip/drivetrain at low speed
    drivetrain: str = "RWD"          # "RWD" | "FWD" | "AWD" (informational in V1)

    # Braking
    max_brake_force: float = 15000.0  # N, total, straight-line, before grip limit

    # Aerodynamics
    frontal_area: float = 2.0        # m^2
    drag_coefficient: float = 0.28   # C_D
    downforce_coefficient: float = 0.0  # C_L (effective lift coefficient, negative-lift convention handled as positive "downforce")
    center_of_pressure: float = 0.5  # fraction of wheelbase from front axle (informational V1)

    # Tire / grip (paired with TireModel in tires.py; mu stored here for convenience)
    tire_mu: float = 1.15            # dry grip coefficient, dimensionless

    # Performance caps
    top_speed: float = 63.0          # m/s, hard cap used when aero corner-speed limit is unbounded

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        return (f"Vehicle('{self.name}', mass={self.mass:.0f}kg, "
                f"power={self.max_power/1000:.0f}kW, Cd={self.drag_coefficient}, "
                f"Cl={self.downforce_coefficient}, mu={self.tire_mu})")
