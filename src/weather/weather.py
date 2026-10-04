"""
Weather / track-condition model (Phase 6).

Wired into Simulator as a uniform grip scale (and optional air density from
ambient temperature). Placeholders are physically motivated in *direction*
only — not measured for a real tire or circuit.

A later upgrade can add rain intensity, a wet racing line, and tire
temperature. This module does not track thermal state.
"""

from dataclasses import dataclass
from enum import Enum

from src.vehicle.aero import RHO_AIR_SEA_LEVEL


class Condition(Enum):
    DRY = "dry"
    DAMP = "damp"
    WET = "wet"


# Illustrative grip multipliers vs dry (mu_effective = mu_dry * modifier).
GRIP_MODIFIER = {
    Condition.DRY: 1.00,
    Condition.DAMP: 0.85,
    Condition.WET: 0.65,
}

# ISA reference used by aero.py (sea level, 15 C → 1.225 kg/m^3).
_T_REF_K = 288.15


@dataclass
class Weather:
    condition: Condition = Condition.DRY
    ambient_temp_c: float = 15.0
    track_temp_c: float = 30.0

    def grip_modifier(self) -> float:
        return GRIP_MODIFIER[self.condition]

    def air_density(self) -> float:
        """
        Sea-level ideal-gas density vs the 15 C aero reference.

        Pressure and humidity are not modeled. Default 15 C returns exactly
        RHO_AIR_SEA_LEVEL so existing dry laps stay comparable.
        """
        t_k = 273.15 + float(self.ambient_temp_c)
        t_k = max(t_k, 200.0)
        return RHO_AIR_SEA_LEVEL * (_T_REF_K / t_k)
