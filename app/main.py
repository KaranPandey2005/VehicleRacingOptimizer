"""
CLI demo.

Builds one track (change the factory on the `track = ...` line), loads a
few vehicle configs, and reports every demo lap on the Phase 5 CasADi
optimized line. Centerline / Phase 4 shaped times are printed only as
baselines, then plots are saved to ./output/.

Run from the project root:
    python -m app.main
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # headless-safe; comment out if running interactively
import matplotlib.pyplot as plt

from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.simulation.results import save_run
from src.optimization.racing_line import shaped_racing_line
from src.visualization.track_plot import (
    plot_track, plot_speed_map, plot_telemetry, plot_line_comparison,
    plot_offset_vs_s, plot_corner_inset_pages, plot_weather_speed_overlay,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    track = Track.spa_francorchamps()
    print(track)
    print(f"Closed loop check: {track.is_closed()}\n")

    ax = plot_track(track)
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "track_layout.png"), dpi=150)
    plt.close(ax.figure)

    vehicle_files = ["subaru_brz.json", "honda_fit.json", "fia_f3.json"]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "vehicles")

    results = []
    for fname in vehicle_files:
        vehicle = Vehicle.from_json(os.path.join(data_dir, fname))
        tire = TireModel(mu=vehicle.tire_mu)
        sim = Simulator(track, vehicle, tire, mode="attack", line_mode="optimized")
        result = sim.run()
        save_run(result)
        results.append(result)
        print(result.summary())

        ax = plot_speed_map(track, result)
        fname_out = fname.replace(".json", "_speedmap.png")
        ax.figure.savefig(os.path.join(OUTPUT_DIR, fname_out), dpi=150)
        plt.close(ax.figure)

        fig = plot_telemetry(result)
        fig.savefig(os.path.join(OUTPUT_DIR, fname.replace(".json", "_telemetry.png")), dpi=150)
        plt.close(fig)

    # Driving-mode effect on the same CasADi line used above.
    vehicle = Vehicle.from_json(os.path.join(data_dir, "subaru_brz.json"))
    tire = TireModel(mu=vehicle.tire_mu)
    print("--- Driving mode comparison (Subaru BRZ, optimized line, Phase 7) ---")
    print("J = lap_time + w_risk*wall_risk + w_tire*tire_stress; u_max caps the GG circle.")
    for mode in ["attack", "race", "safe"]:
        sim = Simulator(track, vehicle, tire, mode=mode, line_mode="optimized")
        result = sim.run()
        save_run(result)
        print(
            f"{mode:6s}: lap = {result.lap_time:.2f} s   J = {result.objective:.2f}  "
            f"u {result.telemetry['mean_utilization']:.2f}/{result.telemetry['peak_utilization']:.2f}  "
            f"wall {result.telemetry['wall_risk']:.3f}"
        )

    from src.weather.weather import Weather, Condition
    print("\n--- Phase 6 weather (Subaru BRZ, optimized line; grip lookup) ---")
    print("Dry/damp/wet scale tire mu. Not a wet racing line or rain model.")
    weather_results = []
    for cond in (Condition.DRY, Condition.DAMP, Condition.WET):
        wx = Weather(condition=cond)
        t = Simulator(track, vehicle, tire, mode="attack",
                      line_mode="optimized", weather=wx).run()
        save_run(t)
        weather_results.append(t)
        print(f"{cond.value:4s}: lap time = {t.lap_time:.2f} s  "
              f"(grip x{wx.grip_modifier():.2f})")
        ax = plot_speed_map(track, t)
        ax.figure.savefig(
            os.path.join(OUTPUT_DIR, f"weather_{cond.value}_speedmap.png"), dpi=150,
        )
        plt.close(ax.figure)
    ax = plot_weather_speed_overlay(
        weather_results,
        title="Phase 6 weather — Subaru BRZ optimized line (dry / damp / wet)",
    )
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "weather_speed_comparison.png"), dpi=150)
    plt.close(ax.figure)

    # Reference only: centerline and Phase 4 shaped vs the optimized BRZ time.
    print("\n--- Reference baselines (Subaru BRZ Attack; not the demo result) ---")
    print("Phase 4 shaped is an outside-apex-outside heuristic; usually slower.")
    vehicle = Vehicle.from_json(os.path.join(data_dir, "subaru_brz.json"))
    tire = TireModel(mu=vehicle.tire_mu)
    centerline = Simulator(track, vehicle, tire, mode="attack",
                             line_mode="centerline").run()
    save_run(centerline)
    centerline_t = centerline.lap_time
    shaped = Simulator(track, vehicle, tire, mode="attack",
                         line_mode="shaped").run()
    save_run(shaped)
    shaped_t = shaped.lap_time
    opt_t = results[0].lap_time
    print(f"centerline: {centerline_t:.2f} s")
    print(f"shaped:     {shaped_t:.2f} s")
    print(f"optimized:  {opt_t:.2f} s")

    opt_lines = {
        r.vehicle_name: {"s": r.profile["s"], "x": r.profile["x"], "y": r.profile["y"],
                         "offset": r.profile.get("offset")}
        for r in results
    }
    ax = plot_line_comparison(
        track, opt_lines, title="Phase 5 CasADi optimized lines (per vehicle)",
    )
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "optimized_line_comparison.png"), dpi=150)
    plt.close(ax.figure)

    ax = plot_offset_vs_s(
        track, opt_lines,
        title="Phase 5 offset n(s) — +left of centerline, dashed = track edges",
    )
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "optimized_line_offset.png"), dpi=150)
    plt.close(ax.figure)

    import glob
    for stale in glob.glob(os.path.join(OUTPUT_DIR, "optimized_line_insets*.png")):
        os.remove(stale)
    figs = plot_corner_inset_pages(
        track, opt_lines, corners_per_fig=6,
        title="Phase 5 racing line — every detected corner (true scale)",
    )
    for i, fig in enumerate(figs, start=1):
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"optimized_line_insets_{i:02d}.png"),
            dpi=160,
        )
        plt.close(fig)
    print(f"Wrote {len(figs)} corner-inset page(s) "
          f"(optimized_line_insets_01.png …)")

    print(f"BRZ centerline vs optimized: {centerline_t:.2f} s -> {opt_t:.2f} s")
    if results[0].used_fallback:
        print("Phase 5 kept the centerline on this track (no faster path found).")

    shaped_lines = {}
    for fname in vehicle_files:
        v = Vehicle.from_json(os.path.join(data_dir, fname))
        shaped_lines[v.name] = shaped_racing_line(track, v)
    ax = plot_line_comparison(
        track, shaped_lines,
        title="Phase 4 shaped lines (experimental — not faster than centerline)",
    )
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "vehicle_line_comparison.png"), dpi=150)
    plt.close(ax.figure)

    print(f"\nPlots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
