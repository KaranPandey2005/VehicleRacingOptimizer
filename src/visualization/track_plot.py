"""
Matplotlib visualization for V1: track layout and speed-colored racing line.
GUI (PySide6) is `python3 -m app.gui`; this module is shared by the CLI and the GUI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_track(track, ax=None, show=False):
    if ax is None:
        span = max(track.x.max() - track.x.min(), track.y.max() - track.y.min())
        figsize = (11, 9) if span > 800 else (9, 6)
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(track.left_x, track.left_y, color="black", linewidth=1.2)
    ax.plot(track.right_x, track.right_y, color="black", linewidth=1.2)
    ax.plot(track.x, track.y, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="centerline")
    ax.plot(track.x[0], track.y[0], marker="o", color="green", markersize=8, label="start/finish")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(track.name)
    ax.legend(loc="upper right", fontsize=8)
    if show:
        plt.show()
    return ax


def plot_speed_map(track, result, ax=None, show=False, cmap="RdYlGn"):
    """Plot the racing line colored by speed, with track boundaries for context."""
    if ax is None:
        span = max(track.x.max() - track.x.min(), track.y.max() - track.y.min())
        figsize = (11, 9) if span > 800 else (9, 6)
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(track.left_x, track.left_y, color="black", linewidth=1.0)
    ax.plot(track.right_x, track.right_y, color="black", linewidth=1.0)

    profile = result.profile
    x, y, v = profile["x"], profile["y"], profile["v"]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=4)
    lc.set_array(v[:-1] * 3.6)  # km/h
    line = ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(track.x.min() - 20, track.x.max() + 20)
    ax.set_ylim(track.y.min() - 20, track.y.max() + 20)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{result.vehicle_name} — {result.mode.upper()} — "
        f"{getattr(result, 'weather', 'dry').upper()} — {result.lap_time:.2f} s"
    )

    cbar = ax.figure.colorbar(line, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Speed (km/h)")

    if show:
        plt.show()
    return ax


def plot_telemetry(result, fig=None, show=False):
    """Speed, longitudinal accel, lateral accel, and friction-circle u vs. distance."""
    profile = result.profile
    telemetry = result.telemetry
    s = profile["s"]
    has_u = "u" in telemetry

    if fig is None:
        n_ax = 4 if has_u else 3
        fig, axes = plt.subplots(n_ax, 1, figsize=(10, 9 if has_u else 8), sharex=True)
    else:
        axes = fig.axes

    axes[0].plot(s, profile["v"] * 3.6, color="tab:blue")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(s, telemetry["a_lon"] / 9.81, color="tab:red")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Long. accel (g)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(s, telemetry["a_lat"] / 9.81, color="tab:green")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Lat. accel (g)")
    axes[2].grid(alpha=0.3)

    if has_u:
        u_max = getattr(result, "u_max", 1.0)
        axes[3].plot(s, telemetry["u"], color="tab:purple")
        axes[3].axhline(u_max, color="black", linewidth=0.8, linestyle="--",
                        label=f"u_max = {u_max:.2f}")
        axes[3].set_ylabel("Friction u")
        axes[3].set_xlabel("Distance along lap (m)")
        axes[3].set_ylim(0.0, max(1.05, u_max + 0.05))
        axes[3].legend(loc="upper right", fontsize=8)
        axes[3].grid(alpha=0.3)
    else:
        axes[2].set_xlabel("Distance along lap (m)")

    fig.suptitle(
        f"{result.vehicle_name} — {result.mode.upper()} — "
        f"{getattr(result, 'weather', 'dry').upper()} — {result.lap_time:.2f} s lap"
    )
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_weather_speed_overlay(results, ax=None, show=False, title=None):
    """Overlay v(s) for several SimulationResults (e.g. dry / damp / wet)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = ["tab:orange", "tab:blue", "tab:green"]
    for i, result in enumerate(results):
        s = result.profile["s"]
        v = result.profile["v"] * 3.6
        label = (
            f"{result.weather.upper()}  {result.lap_time:.2f} s  "
            f"(grip x{result.grip_modifier:.2f})"
        )
        ax.plot(s, v, color=colors[i % len(colors)], linewidth=1.8, label=label)
    ax.set_xlabel("Distance along lap (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title or "Weather comparison — speed vs distance")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    if show:
        plt.show()
    return ax


def plot_line_comparison(track, lines, ax=None, show=False, title=None):
    """Overlay several racing lines (name -> line dict with x/y) on one track."""
    if ax is None:
        span = max(track.x.max() - track.x.min(), track.y.max() - track.y.min())
        figsize = (11, 9) if span > 800 else (9, 6)
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(track.left_x, track.left_y, color="black", linewidth=1.0)
    ax.plot(track.right_x, track.right_y, color="black", linewidth=1.0)
    ax.plot(track.x, track.y, color="gray", linewidth=0.8, linestyle="--",
            alpha=0.6, label="centerline")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (name, line) in enumerate(lines.items()):
        ax.plot(line["x"], line["y"], color=colors[i % len(colors)],
                linewidth=1.8, label=name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or "Racing line comparison")
    ax.legend(loc="upper right", fontsize=8)
    if show:
        plt.show()
    return ax


def _line_offset(track, line):
    if "offset" in line and line["offset"] is not None:
        return np.asarray(line["offset"], dtype=float)
    nx = -np.sin(track.heading)
    ny = np.cos(track.heading)
    return (np.asarray(line["x"]) - track.x) * nx + (np.asarray(line["y"]) - track.y) * ny


# Named windows along the TUM/OSM 2D Spa centerline (s = 0 at start/finish).
# Several curvature peaks can sit inside one F1-named corner.
_SPA_TURN_RANGES = (
    (0.0, 700.0, 1, "La Source"),
    (700.0, 1100.0, 2, "Eau Rouge"),
    (1100.0, 1700.0, 3, "Raidillon"),
    (1700.0, 2380.0, None, "Kemmel Straight"),
    (2380.0, 2580.0, 4, "Les Combes"),
    (2580.0, 2850.0, 5, "Malmedy"),
    (2850.0, 3550.0, 6, "Rivage"),
    (3550.0, 4300.0, 7, "Pouhon"),
    (4300.0, 4580.0, 8, "Fagnes"),
    (4580.0, 4820.0, 9, "Campus"),
    (4820.0, 5450.0, 10, "Stavelot"),
    (5450.0, 6550.0, 11, "Blanchimont"),
    (6550.0, 7100.0, 12, "Bus Stop"),
)


def _spa_turn_key(s):
    for lo, hi, num, name in _SPA_TURN_RANGES:
        if lo <= float(s) < hi:
            return num, name
    return None, "Unnamed"


def corner_panel_titles(track, stations_s, turn_signs=None):
    """Titles in lap order: 'T8 Fagnes (1/1)' plus station, for Spa; else Corner k."""
    stations_s = [float(s) for s in stations_s]
    is_spa = "spa" in str(getattr(track, "name", "")).lower()
    keys = []
    for s in stations_s:
        if is_spa:
            keys.append(_spa_turn_key(s))
        else:
            keys.append((None, None))
    from collections import Counter
    counts = Counter(keys)
    seen = Counter()
    titles = []
    for i, (s, key) in enumerate(zip(stations_s, keys)):
        seen[key] += 1
        num, name = key
        side = ""
        if turn_signs is not None and i < len(turn_signs):
            sgn = float(turn_signs[i])
            if sgn < 0:
                side = " · R"
            elif sgn > 0:
                side = " · L"
        if is_spa and name:
            turn = f"T{num} " if num else ""
            part = f" ({seen[key]}/{counts[key]})" if counts[key] > 1 else ""
            titles.append(f"{turn}{name}{part}{side}\ns = {s:.0f} m")
        else:
            titles.append(f"Corner {i + 1}{side}\ns = {s:.0f} m")
    return titles


def _detected_corner_indices(track):
    from src.optimization.trajectory import _detect_corners
    corners = _detect_corners(track)
    if not corners:
        return np.array([], dtype=int), np.array([]), np.array([])
    idx = np.array([int(np.argmin(np.abs(track.s - c["s"]))) for c in corners])
    stations = np.array([float(c["s"]) for c in corners])
    signs = np.array([float(c["sign"]) for c in corners])
    return idx, stations, signs


def _inset_stations(track, lines, n_panels=6, min_sep_m=180.0):
    """Stations where the drawn line leaves the centerline the most."""
    n_use = None
    for line in lines.values():
        n = np.abs(_line_offset(track, line))
        n_use = n if n_use is None else np.maximum(n_use, n)
    if n_use is None or float(np.max(n_use)) < 0.15:
        n_use = np.abs(track.curvature)
    ds = float(np.median(np.diff(track.s))) if len(track.s) > 1 else 1.0
    min_sep = max(int(min_sep_m / max(ds, 0.5)), 8)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(n_use, distance=min_sep)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(n_use))])
        heights = n_use[peaks]
    else:
        heights = props.get("peak_heights", n_use[peaks])
    order = np.argsort(heights)[::-1][:n_panels]
    return np.sort(peaks[order])


def plot_offset_vs_s(track, lines, ax=None, show=False, title=None):
    """Lateral offset n(s): + is left of centerline. Track edges as dashed rails."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.2))
    half_w = 0.5 * float(track.width)
    ax.axhline(half_w, color="black", linestyle="--", linewidth=1.0, label="left edge")
    ax.axhline(-half_w, color="black", linestyle="--", linewidth=1.0, label="right edge")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (name, line) in enumerate(lines.items()):
        s = np.asarray(line.get("s", track.s), dtype=float)
        ax.plot(s, _line_offset(track, line), color=colors[i % len(colors)],
                linewidth=1.6, label=name)
    ax.set_xlim(float(track.s[0]), float(track.s[-1]))
    ax.set_ylim(-half_w - 0.6, half_w + 0.6)
    ax.set_xlabel("Distance along lap (m)")
    ax.set_ylabel("Offset n (m)\n(+ left)")
    ax.set_title(title or "Racing-line offset vs centerline")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    if show:
        plt.show()
    return ax


def _plot_one_inset(ax, track, lines, i_c, title, pad_m):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    styles = ["-", "--", "-.", ":", (0, (5, 1))]
    x0, y0 = float(track.x[i_c]), float(track.y[i_c])
    ax.plot(track.left_x, track.left_y, color="black", linewidth=2.0, zorder=2)
    ax.plot(track.right_x, track.right_y, color="black", linewidth=2.0, zorder=2)
    ax.plot(track.x, track.y, color="0.55", linewidth=1.4, linestyle=":",
            zorder=3, label="centerline")
    for i, (name, line) in enumerate(lines.items()):
        ax.plot(line["x"], line["y"], color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)], linewidth=3.2,
                label=name, zorder=4, solid_capstyle="round")
    ax.set_xlim(x0 - pad_m, x0 + pad_m)
    ax.set_ylim(y0 - pad_m, y0 + pad_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def list_corner_insets(track, lines=None):
    """Detected corners in lap order: arrays of sample index and panel titles."""
    lines = lines or {"centerline": {
        "s": track.s, "x": track.x, "y": track.y, "offset": np.zeros_like(track.s),
    }}
    idx, stations, signs = _detected_corner_indices(track)
    if idx.size == 0:
        idx = _inset_stations(track, lines, n_panels=12)
        stations = track.s[idx]
        signs = np.sign(track.curvature[idx])
    titles = corner_panel_titles(track, stations, turn_signs=signs)
    return idx, titles


def plot_single_corner_inset(track, lines, corner_index, ax=None, pad_m=None,
                             title=None, show=False):
    """One true-scale zoomed corner (used by the GUI)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    else:
        fig = ax.figure
    if pad_m is None:
        pad_m = max(2.2 * float(track.width), 16.0)
    _plot_one_inset(ax, track, lines, int(corner_index), title or "", pad_m)
    if show:
        plt.show()
    return ax


def _draw_inset_page(track, lines, idx, titles, pad_m, page_title):
    n_show = len(idx)
    fig, axes = plt.subplots(2, 3, figsize=(22.0, 14.5))
    axes = np.atleast_1d(axes).ravel()
    for j, (ax, i_c) in enumerate(zip(axes, idx)):
        _plot_one_inset(ax, track, lines, i_c, titles[j], pad_m)
        ax.legend().remove()
    for ax in axes[n_show:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                   fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(page_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.97])
    return fig


def plot_corner_inset_pages(track, lines, corners_per_fig=6, pad_m=None,
                            show=False, title=None):
    """
    One zoomed panel per detected racing-line corner, in lap order.

    At most `corners_per_fig` panels per figure (default 6). Spa panels are
    labeled with turn number + name. Crop is tight so car-to-car line
    differences are visible (true scale, not exaggerated).
    """
    idx, stations, signs = _detected_corner_indices(track)
    if idx.size == 0:
        idx = _inset_stations(track, lines, n_panels=max(int(corners_per_fig), 1))
        stations = track.s[idx]
        signs = np.sign(track.curvature[idx])
    titles = corner_panel_titles(track, stations, turn_signs=signs)
    if pad_m is None:
        pad_m = max(2.2 * float(track.width), 16.0)

    n = len(idx)
    per = max(int(corners_per_fig), 1)
    n_pages = int(np.ceil(n / per)) if n else 1
    figs = []
    for p in range(n_pages):
        sl = slice(p * per, min((p + 1) * per, n))
        page_title = title or "Racing line — every detected corner (true scale)"
        page_title = (
            f"{page_title}  ·  page {p + 1}/{n_pages}  ·  "
            f"corners {p * per + 1}–{min((p + 1) * per, n)} of {n}"
        )
        fig = _draw_inset_page(track, lines, idx[sl], titles[sl], pad_m, page_title)
        figs.append(fig)
        if show:
            plt.show()
    return figs


def plot_corner_insets(track, lines, n_panels=6, pad_m=None, show=False, title=None):
    """First page of `plot_corner_inset_pages` (kept for older call sites)."""
    figs = plot_corner_inset_pages(
        track, lines, corners_per_fig=n_panels, pad_m=pad_m, show=show, title=title,
    )
    return figs[0]
