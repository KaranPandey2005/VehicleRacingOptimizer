"""
Aerodynamics model (V1).

Standard quadratic-in-speed relations:

    F_drag      = 0.5 * rho * C_D * A * v^2      (opposes motion)
    F_downforce = 0.5 * rho * C_L * A * v^2      (adds to normal/tire load)

rho (air density) defaults to sea-level, 15 C: 1.225 kg/m^3. This is a
simplification -- true air density varies with altitude, temperature and
humidity; Phase 6 weather can override rho via Simulator.

These relations are standard aerodynamics (not vehicle-specific empirical
constants), so they are safe to use directly; what's vehicle-specific is
C_D, C_L and frontal area A, which come from the Vehicle config and are
NOT validated against a real car unless stated.
"""

RHO_AIR_SEA_LEVEL = 1.225  # kg/m^3


def drag_force(vehicle, v, rho=RHO_AIR_SEA_LEVEL):
    """Aerodynamic drag force (N), opposing motion. v in m/s, always >= 0 here."""
    return 0.5 * rho * vehicle.drag_coefficient * vehicle.frontal_area * v ** 2


def downforce(vehicle, v, rho=RHO_AIR_SEA_LEVEL):
    """Aerodynamic downforce (N), added to tire normal load. v in m/s."""
    return 0.5 * rho * vehicle.downforce_coefficient * vehicle.frontal_area * v ** 2
