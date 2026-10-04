# Vehicle-Specific Racing Line & Lap-Time Optimization Simulator

A physics-based simulator that computes a vehicle-specific speed profile and
lap time for a given track, vehicle, tire grip, and driving mode. The central
engineering goal of this project is that the result should depend on **which
car** you simulate, not just the track shape.

> **Status:** Phases 1–8 plus Phase 10a physics (torque curve + axle split).
> Nothing here is validated against a real vehicle or track unless explicitly
> stated — treat all numbers as illustrative engineering-simulation output,
> not ground truth.

## What it does

- Generates a closed-loop, mathematically exact 2D track (rounded rectangle
  with 4 independently-sized corners) from analytic geometry — no numerical
  drift, verified to close to ~1e-13 m — and can load Spa from a centerline CSV.
- Models a configurable vehicle (mass, powertrain, drivetrain, drag, downforce,
  brakes, tire grip) loaded from JSON. Drive force comes from a torque curve
  × gears; FWD/RWD/AWD and brake bias use front/rear load transfer.
- Computes speed-dependent tire grip (per-axle friction circle with combined
  slip and load sensitivity) and aerodynamic downforce/drag.
- Solves a vehicle-specific speed profile along a line using the standard
  **quasi-steady-state forward-backward pass**.
- Optimizes a vehicle-specific racing line (`line_mode="optimized"`).
- Driving modes (Attack / Race / Safe) cap friction-circle utilization and
  minimize `J = w_t*lap_time + w_r*wall_risk + w_s*tire_stress`.
- Integrates lap time and derives telemetry (speed, g's, friction utilization).
- Plots the track, a speed-colored racing line, and telemetry channels.
- Desktop GUI (`python3 -m app.gui`) to pick car / track / mode / weather / line
  and run the same Simulator without the CLI.

**Deferred:** Pacejka tires, thermal/wear, elevation, a true wet line,
validation against real laps (Phase 9). Next physics: Pacejka / bicycle.

## Quick start

```bash
cd VehicleRacingOptimizer
pip install -r requirements.txt
python3 -m app.main                 # CLI demo, writes plots to ./output/
python3 -m app.gui                  # Phase 8 desktop GUI
pytest                              # test suite
```

`app/main.py` builds the default track, runs three production-car ballpark
vehicles (Subaru BRZ, Honda Fit, FIA F3) through Maximum Attack mode, prints a
summary, and also compares Attack/Race/Safe modes for the BRZ. All plots are
saved as PNGs in `output/`. Parameters are published-spec ballparks, not
validated Spa setups.

## Architecture

```
VehicleRacingOptimizer/
├── src/
│   ├── track/
│   │   ├── geometry.py      # analytic straight/arc segment builder (turtle graphics)
│   │   └── track.py         # Track class: centerline, boundaries, curvature queries
│   ├── vehicle/
│   │   ├── vehicle.py       # Vehicle config dataclass, JSON load/save
│   │   ├── powertrain.py    # torque curve × gears (Phase 10a)
│   │   ├── tires.py         # friction-circle + load sensitivity (Phase 7)
│   │   └── aero.py          # drag / downforce (quadratic-in-speed)
│   ├── weather/
│   │   └── weather.py       # dry/damp/wet grip + air density
│   ├── physics/
│   │   └── dynamics.py      # corner-speed limit, traction accel limit, braking decel limit
│   ├── optimization/
│   │   ├── racing_line.py   # geometric / shaped line + QSS speed solver
│   │   ├── trajectory.py    # Phase 5 CasADi / greedy optimized line
│   │   ├── safety.py        # Phase 7 driving modes, wall risk, tire stress
│   │   └── lap_time.py      # lap time integration + telemetry (g's)
│   ├── simulation/
│   │   └── simulator.py     # Simulator: ties track+vehicle+tire+mode together
│   └── visualization/
│       └── track_plot.py    # matplotlib: track layout, speed map, telemetry
├── data/
│   ├── tracks/               # (reserved for imported/saved track JSON)
│   └── vehicles/              # subaru_brz.json, honda_fit.json, fia_f3.json
├── tests/                     # pytest suite (also runnable via tests/_run_all.py without pytest)
└── app/
    ├── main.py                # CLI demo entry point
    ├── gui.py                 # Phase 8 PySide6 GUI
    └── session.py             # shared car/track/run setup for CLI + GUI
```

### Why this structure

Each physical concern (track geometry, vehicle, tires, aero, weather,
dynamics limits, optimization, simulation orchestration, visualization) is
its own module with a single responsibility. `Simulator` wires a run;
`app/gui.py` is only a front end on that.

## The physics, briefly

**Track geometry** (`track/geometry.py`): tracks are built as a sequence of
`Straight(length)` and `Arc(radius, signed_angle)` segments, integrated
analytically from a starting pose. This is exact — no accumulated numerical
error — which is why a segment list designed to close (e.g. 4 arcs of +90°
= 360° total turning, consistent straight lengths) closes to floating-point
precision. The V1 default track is a **rounded rectangle with 4
independently-sized corner radii**, chosen specifically so different
vehicles will visibly prefer different speeds through each corner.

**Powertrain (Phase 10a)** (`vehicle/powertrain.py`, `physics/dynamics.py`):
drive force is `T(rpm) × gear × final × η / r` in the best legal gear, then
capped by the *driven* axle's leftover friction circle. Front/rear loads
include static split, aero at the center of pressure, and `m a_x h / L`
transfer. FWD/RWD/AWD and `brake_bias_front` now change the lap, not just
the JSON label.

**Grip & aero** (`vehicle/tires.py`, `vehicle/aero.py`):
```
F_drag       = 0.5 * rho * C_D * A * v^2
F_downforce  = 0.5 * rho * C_L * A * v^2
N(v)         = m*g + F_downforce(v)
F_grip_max   = mu * N(v)
```

**Cornering speed limit** (`physics/dynamics.py`): setting required lateral
force `m*v^2*kappa` equal to available grip `mu*(m*g + 0.5*rho*Cl*A*v^2)`
gives a linear equation in `v^2` (solved in closed form, not iteratively,
because both sides are already polynomial in `v^2`). With zero downforce
this collapses exactly to the textbook `v_max = sqrt(mu*g*r)` — verified by
a dedicated test.

**Speed profile solver** (`optimization/racing_line.py`): the standard
forward-backward quasi-steady-state pass. Start at the point with the lowest
corner-speed limit (guaranteed achievable), sweep forward applying
traction-limited acceleration clipped to each point's corner limit, then
sweep backward applying braking-limited deceleration so the car actually has
room to slow down before each corner. Repeated for a few passes so the
closed-loop seam converges. (Earlier in development this had a subtle bug
where the seam between the last point and the start point wasn't
braking-checked, producing a spurious multi-hundred-g "phantom braking"
spike — now fixed and covered by
`test_speed_profile_respects_corner_speed_limit_everywhere`.)

**Driving modes (Phase 7)**: each mode is a friction-circle utilization cap
`u_max` plus weights on a scalar objective
`J = w_time*lap_time + w_risk*wall_risk + w_tire*tire_stress`.
Attack is minimum-time (`u_max = 1`, risk weights 0). Race and Safe leave
grip and wall margin, and the optimizer may refuse a wider line if J gets
worse. This is not a crash-probability or tire-temperature model.

## Known limitations (intentional, tracked for later phases)

- Tire model is a circular combined-slip envelope with optional load
  sensitivity. **Not yet:** Pacejka/Magic Formula, slip angle/ratio states,
  temperature, wear, compound, camber, inflation, or front/rear split.
- Engine modeled as a single max-power figure with a traction-force floor,
  not a gear-by-gear torque curve.
- No elevation. Extra track layouts are factory functions on the existing
  `Straight`/`Arc` builder, not a redesign.
- Weather is a dry/damp/wet grip scale (and optional air density from
  ambient temperature) — not a wet line or rain model.
- No vehicle has been validated against real-world data — vehicle JSON
  files are clearly labeled "illustrative, unvalidated."

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Track representation, geometry, visualization | ✅ Done |
| 2 | Vehicle, tires, aero, basic dynamics limits | ✅ Done |
| 3 | Geometric racing line + speed profile + lap time | ✅ Done |
| 4 | Vehicle-specific heuristic line (`shaped`) | ✅ Done (experimental; often slower) |
| 5 | Nonlinear / time-optimal line (`optimized`) | ✅ Done |
| 6 | Weather (wire up `weather.py`) | ✅ Done (grip lookup + air density) |
| 7 | Safety model + friction-circle / load-sensitive tires | ✅ Done |
| 8 | PySide6 GUI | ✅ Done |
| 9 | Validation against real data / other simulators | Planned (after more physics) |
| 10a | Torque curve + F/R split / load transfer | ✅ Done |

Later physics: Pacejka / bicycle, thermal/wear, elevation, a true wet racing line.

## Testing

```bash
pytest                    # preferred
python tests/_run_all.py  # fallback runner if pytest isn't installed
```

Coverage: track geometry closure and curvature correctness, tire/vehicle
JSON round-trip, physics limits (corner-speed formula matches the classical
closed form with zero downforce, downforce/grip/power monotonicity), and
full-pipeline integration tests (positive finite lap time, speed never
exceeds the corner limit, Safe mode is provably slower than Attack mode,
higher-power/higher-downforce/higher-grip vehicles are provably faster).

## Disclaimer

This is an educational/portfolio engineering simulation tool. Nothing in
this repository should be treated as validated real-world vehicle dynamics
data, and results should never be used as real-world driving guidance.
