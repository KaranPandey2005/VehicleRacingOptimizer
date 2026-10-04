import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.optimization.racing_line import (
    geometric_racing_line, shaped_racing_line, apex_bias_m,
)


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_simulator_defaults_to_centerline():
    sim = Simulator(default_track(), Vehicle(), mode="attack")
    assert sim.line_mode == "centerline"


def test_unknown_line_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        Simulator(default_track(), Vehicle(), line_mode="magic")


def test_shaped_line_stays_inside_track_boundaries():
    track = default_track()
    line = shaped_racing_line(track, Vehicle())
    dist = np.hypot(line["x"] - track.x, line["y"] - track.y)
    assert np.all(dist <= track.width / 2.0 + 1e-6)
    assert np.all(np.abs(line["offset"]) <= track.width / 2.0 + 1e-6)


def test_shaped_line_is_not_the_centerline():
    track = default_track()
    geo = geometric_racing_line(track)
    shaped = shaped_racing_line(track, Vehicle())
    assert np.max(np.hypot(shaped["x"] - geo["x"], shaped["y"] - geo["y"])) > 0.5


def test_shaped_line_is_vehicle_specific():
    track = default_track()
    brz = Vehicle()
    low = Vehicle(name="Low", max_power=60_000.0, mass=1000.0, tire_mu=1.0)
    high = Vehicle(name="High", max_power=300_000.0, mass=1100.0, tire_mu=1.6)
    n_low = shaped_racing_line(track, low)["offset"]
    n_high = shaped_racing_line(track, high)["offset"]
    n_brz = shaped_racing_line(track, brz)["offset"]
    assert np.max(np.abs(n_low - n_high)) > 0.05
    assert np.max(np.abs(n_low - n_brz)) > 0.02


def test_high_power_car_has_later_apex_bias_than_low_power():
    low = Vehicle(name="Low", max_power=75_000.0, mass=1000.0, tire_mu=1.0)
    high = Vehicle(name="High", max_power=260_000.0, mass=1100.0, tire_mu=1.2)
    assert apex_bias_m(high) > apex_bias_m(low)


def test_shaped_line_curvature_is_finite_and_not_hairpin_spike():
    track = default_track()
    line = shaped_racing_line(track, Vehicle())
    k = line["curvature"]
    assert np.all(np.isfinite(k))
    k_track = np.max(np.abs(track.curvature))
    # Heuristic can tighten the apex, but must not invent <~2 m radius corners.
    assert np.max(np.abs(k)) < max(0.5, 8.0 * k_track)


def test_shaped_line_offset_is_smooth_on_closed_loop():
    track = default_track()
    n = shaped_racing_line(track, Vehicle())["offset"]
    dn = np.abs(np.diff(n, append=n[0]))
    # Half-cosine + periodic gaussian: no single-sample jumps of meters.
    assert np.max(dn) < 0.8


def test_shaped_mode_runs_and_produces_finite_lap_time():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    result = Simulator(track, vehicle, tire, mode="attack", line_mode="shaped").run()
    assert result.line_mode == "shaped"
    assert result.lap_time > 0
    assert np.isfinite(result.lap_time)
    assert np.all(result.profile["v"] >= 0)
    assert np.all(result.profile["v"] <= result.profile["v_corner_limit"] + 1e-6)


def test_corner_insets_and_offset_plots_run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization.track_plot import plot_corner_insets, plot_offset_vs_s

    track = default_track()
    line = shaped_racing_line(track, Vehicle())
    fig = plot_corner_insets(track, {"shaped": line}, n_panels=4)
    assert len(fig.axes) >= 4
    ax = plot_offset_vs_s(track, {"shaped": line})
    assert ax.has_data()
    plt.close("all")


def test_spa_inset_pages_cover_all_detected_corners_in_order():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.optimization.trajectory import _detect_corners
    from src.visualization.track_plot import (
        plot_corner_inset_pages, corner_panel_titles,
    )

    track = Track.spa_francorchamps()
    corners = _detect_corners(track)
    assert len(corners) >= 20
    titles = corner_panel_titles(
        track, [c["s"] for c in corners], [c["sign"] for c in corners],
    )
    assert "La Source" in titles[0]
    assert "Bus Stop" in titles[-1]
    assert titles == sorted(titles, key=lambda t: int(t.split("s = ")[1].split()[0]))
    line = {"x": track.x, "y": track.y, "s": track.s,
            "offset": np.zeros_like(track.s)}
    figs = plot_corner_inset_pages(track, {"center": line}, corners_per_fig=6)
    assert len(figs) == int(np.ceil(len(corners) / 6.0))
    plt.close("all")


def test_list_corner_insets_names_spa_turns():
    from src.visualization.track_plot import list_corner_insets

    track = Track.spa_francorchamps()
    idx, titles = list_corner_insets(track)
    assert len(idx) >= 20
    blob = " ".join(titles)
    assert "Eau Rouge" in blob
    assert "Pouhon" in blob


def test_chicane_shaped_line_stays_in_bounds():
    track = Track.chicane_circuit(ds=2.0)
    line = shaped_racing_line(track, Vehicle())
    dist = np.hypot(line["x"] - track.x, line["y"] - track.y)
    assert np.all(dist <= track.width / 2.0 + 1e-6)
