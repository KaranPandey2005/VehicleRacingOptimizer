"""
Vehicle model.

Holds configurable physical parameters for a vehicle and derives simple,
physically-motivated forces from them. Values are NOT claimed to represent
any real vehicle unless explicitly validated — see README.

Phase 10a: optional torque curve, gear ratios, and brake bias. Empty curve
or ratios are synthesized from max_power / top_speed. Drivetrain (RWD/FWD/AWD)
now limits which axle can put down drive force.

Still later: Pacejka, thermal/wear, elevation, a true wet line.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field


@dataclass
class Vehicle:
    name: str = "Subaru BRZ (2022+, unvalidated ballpark)"

    # Mass & geometry (default = BRZ ballpark; see data/vehicles/)
    mass: float = 1275.0             # kg
    wheelbase: float = 2.575         # m
    track_width: float = 1.52        # m
    cg_height: float = 0.50          # m
    weight_dist_front: float = 0.53  # fraction of mass on front axle (0-1)

    # Powertrain
    max_power: float = 170_000.0     # W, also used as a hard power cap
    max_traction_force: float = 9500.0  # N, clutch/diff/drivetrain ceiling
    drivetrain: str = "RWD"          # "RWD" | "FWD" | "AWD"
    drivetrain_efficiency: float = 0.90
    final_drive: float = 4.10
    rpm_idle: float = 1200.0
    rpm_redline: float = 7500.0
    tire_radius: float = 0.312        # m, driven-wheel rolling radius
    # [[rpm, Nm], ...]  empty → synthesized from max_power
    torque_curve: list = field(default_factory=list)
    # descending or any order; empty → synthesized 6-speed
    gear_ratios: list = field(default_factory=list)
    awd_torque_front: float = 0.40   # used only when drivetrain == AWD

    # Braking
    max_brake_force: float = 15000.0  # N, total hydraulic capacity
    brake_bias_front: float = 0.62    # fraction of brake force requested at front

    # Aerodynamics
    frontal_area: float = 2.0        # m^2
    drag_coefficient: float = 0.28   # C_D
    downforce_coefficient: float = 0.0  # C_L as positive downforce
    center_of_pressure: float = 0.5  # fraction of wheelbase from front axle

    # Tire / grip (paired with TireModel in tires.py)
    tire_mu: float = 1.15            # dry grip coefficient, dimensionless

    # Performance caps
    top_speed: float = 63.0          # m/s

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self):
        return (f"Vehicle('{self.name}', mass={self.mass:.0f}kg, "
                f"power={self.max_power/1000:.0f}kW, Cd={self.drag_coefficient}, "
                f"Cl={self.downforce_coefficient}, mu={self.tire_mu}, "
                f"{self.drivetrain})")
