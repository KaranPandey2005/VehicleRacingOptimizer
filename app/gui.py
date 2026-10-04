"""
Phase 8 GUI.

Pick a car, track, driving mode, weather, and racing-line type, then run the
existing Simulator. Plots reuse src.visualization.track_plot.

Run from the project root:
    python3 -m app.gui
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSplitter, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from app.session import (
    LINE_MODES, MODES, OUTPUT_DIR, TRACK_FACTORIES, WEATHER, SessionSpec,
    get_track, run_session, slow_run_warning, vehicle_files,
)
from src.visualization.track_plot import (
    list_corner_insets, plot_single_corner_inset, plot_speed_map,
    plot_telemetry, plot_track,
)


class SimWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, spec: SessionSpec):
        super().__init__()
        self.spec = spec

    def run(self):
        try:
            track, result = run_session(self.spec)
            self.succeeded.emit(track, result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Racing Optimizer")
        self.resize(1280, 820)
        self._worker = None
        self._last_track = None
        self._last_result = None
        self._corner_idx = []
        self._vehicles = vehicle_files()

        root = QSplitter(Qt.Horizontal)
        self.setCentralWidget(root)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_plots())
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([340, 940])

        self.statusBar().showMessage("Ready. Centerline on the rectangle is the fast path.")
        self._on_selection_changed()
        self._draw_track_preview()

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)

        title = QLabel("Phase 8 — run a lap")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)
        blurb = QLabel(
            "Uses the same Simulator as the CLI. Optimized Spa is slow; "
            "start with the rounded rectangle + centerline."
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        self.vehicle_box = QComboBox()
        for name, path in self._vehicles:
            self.vehicle_box.addItem(name, path)
        self.track_box = QComboBox()
        self.track_box.addItems(list(TRACK_FACTORIES))
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.line_box = QComboBox()
        self.line_box.addItems(LINE_MODES)
        self.weather_box = QComboBox()
        self.weather_box.addItems(WEATHER)
        form.addRow("Vehicle", self.vehicle_box)
        form.addRow("Track", self.track_box)
        form.addRow("Mode", self.mode_box)
        form.addRow("Line", self.line_box)
        form.addRow("Weather", self.weather_box)
        layout.addLayout(form)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #8a5a00;")
        layout.addWidget(self.hint)

        corner_hint = QLabel(
            "After a run, open the Corners tab and pick a Spa turn "
            "(Eau Rouge, Pouhon, …) to zoom the line vs centerline."
        )
        corner_hint.setWordWrap(True)
        layout.addWidget(corner_hint)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run lap")
        self.run_btn.setDefault(True)
        self.save_btn = QPushButton("Save plots")
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Lap summary will appear here.")
        layout.addWidget(self.summary, 1)

        note = QLabel(
            "Illustrative sim — not validated against a real car or circuit."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(note)

        self.track_box.currentTextChanged.connect(self._on_selection_changed)
        self.line_box.currentTextChanged.connect(self._on_selection_changed)
        self.run_btn.clicked.connect(self._run)
        self.save_btn.clicked.connect(self._save_plots)
        return panel

    def _build_plots(self) -> QWidget:
        tabs = QTabWidget()
        self.speed_fig = Figure(figsize=(7.5, 6.2))
        self.speed_canvas = FigureCanvas(self.speed_fig)
        self.tel_fig = Figure(figsize=(7.5, 7.2))
        self.tel_canvas = FigureCanvas(self.tel_fig)

        corner_page = QWidget()
        corner_layout = QVBoxLayout(corner_page)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        self.corner_prev = QPushButton("Previous")
        self.corner_next = QPushButton("Next")
        self.corner_box = QComboBox()
        self.corner_box.setEnabled(False)
        self.corner_prev.setEnabled(False)
        self.corner_next.setEnabled(False)
        nav.addWidget(self.corner_prev)
        nav.addWidget(self.corner_box, 1)
        nav.addWidget(self.corner_next)
        corner_layout.addLayout(nav)
        self.inset_fig = Figure(figsize=(7.5, 6.5))
        self.inset_canvas = FigureCanvas(self.inset_fig)
        corner_layout.addWidget(self.inset_canvas, 1)
        self.corner_box.currentIndexChanged.connect(self._draw_selected_corner)
        self.corner_prev.clicked.connect(lambda: self._nudge_corner(-1))
        self.corner_next.clicked.connect(lambda: self._nudge_corner(1))

        tabs.addTab(self.speed_canvas, "Track / speed")
        tabs.addTab(self.tel_canvas, "Telemetry")
        tabs.addTab(corner_page, "Corners")
        self.plot_tabs = tabs
        return tabs

    def _spec(self) -> SessionSpec:
        return SessionSpec(
            vehicle_path=self.vehicle_box.currentData(),
            track_key=self.track_box.currentText(),
            mode=self.mode_box.currentText(),
            line_mode=self.line_box.currentText(),
            weather=self.weather_box.currentText(),
        )

    def _on_selection_changed(self, *_args):
        warn = slow_run_warning(self.track_box.currentText(), self.line_box.currentText())
        self.hint.setText(warn)
        if self._last_result is None:
            self._draw_track_preview()

    def _draw_track_preview(self):
        track = get_track(self.track_box.currentText())
        self.speed_fig.clear()
        ax = self.speed_fig.add_subplot(111)
        plot_track(track, ax=ax)
        self.speed_fig.tight_layout()
        self.speed_canvas.draw_idle()
        self.tel_fig.clear()
        ax = self.tel_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Run a lap to see telemetry", ha="center", va="center")
        ax.set_axis_off()
        self.tel_canvas.draw_idle()
        self._reset_corners_placeholder("Run a lap, then pick a corner here.")

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        spec = self._spec()
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("Running…")
        self._worker = SimWorker(spec)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_fail)
        self._worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._worker.start()

    def _on_success(self, track, result):
        self._last_track = track
        self._last_result = result
        self.summary.setPlainText(result.summary())
        self.save_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"{result.lap_time:.2f} s  ·  {result.mode}  ·  {result.line_mode}  ·  {result.weather}"
        )
        self._draw_result(track, result)
        self._load_corners(track, result)

    def _on_fail(self, message: str):
        self.statusBar().showMessage("Run failed")
        QMessageBox.critical(self, "Simulation failed", message)

    def _draw_result(self, track, result):
        self.speed_fig.clear()
        ax = self.speed_fig.add_subplot(111)
        plot_speed_map(track, result, ax=ax)
        self.speed_fig.tight_layout()
        self.speed_canvas.draw_idle()

        self.tel_fig.clear()
        n_ax = 4 if "u" in result.telemetry else 3
        self.tel_fig.subplots(n_ax, 1, sharex=True)
        plot_telemetry(result, fig=self.tel_fig)
        self.tel_canvas.draw_idle()

    def _line_dict(self, result):
        name = result.vehicle_name.split("(")[0].strip()
        return {
            name: {
                "s": result.profile["s"],
                "x": result.profile["x"],
                "y": result.profile["y"],
                "offset": result.profile.get("offset"),
            }
        }

    def _reset_corners_placeholder(self, message):
        self.corner_box.blockSignals(True)
        self.corner_box.clear()
        self.corner_box.blockSignals(False)
        self.corner_box.setEnabled(False)
        self.corner_prev.setEnabled(False)
        self.corner_next.setEnabled(False)
        self._corner_idx = []
        self.inset_fig.clear()
        ax = self.inset_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.inset_canvas.draw_idle()

    def _load_corners(self, track, result):
        lines = self._line_dict(result)
        idx, titles = list_corner_insets(track, lines)
        self._corner_idx = list(idx)
        self.corner_box.blockSignals(True)
        self.corner_box.clear()
        for title in titles:
            self.corner_box.addItem(title.replace("\n", "  ·  "))
        self.corner_box.blockSignals(False)
        has = len(self._corner_idx) > 0
        self.corner_box.setEnabled(has)
        self.corner_prev.setEnabled(has)
        self.corner_next.setEnabled(has)
        if has:
            self.corner_box.setCurrentIndex(0)
            self._draw_selected_corner()
            self.plot_tabs.setTabText(2, f"Corners ({len(self._corner_idx)})")
        else:
            self._reset_corners_placeholder("No corners detected on this track.")
            self.plot_tabs.setTabText(2, "Corners")

    def _nudge_corner(self, delta):
        n = self.corner_box.count()
        if n == 0:
            return
        self.corner_box.setCurrentIndex((self.corner_box.currentIndex() + delta) % n)

    def _draw_selected_corner(self, *_args):
        i = self.corner_box.currentIndex()
        if i < 0 or i >= len(self._corner_idx) or self._last_result is None:
            return
        self.inset_fig.clear()
        ax = self.inset_fig.add_subplot(111)
        title = self.corner_box.currentText().replace("  ·  ", "\n")
        plot_single_corner_inset(
            self._last_track,
            self._line_dict(self._last_result),
            self._corner_idx[i],
            ax=ax,
            title=title,
        )
        self.inset_fig.tight_layout()
        self.inset_canvas.draw_idle()

    def _save_plots(self):
        if self._last_result is None or self._last_track is None:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        slug = (
            f"gui_{self._last_result.vehicle_name.split('(')[0].strip().replace(' ', '_').lower()}"
            f"_{self._last_result.mode}_{self._last_result.line_mode}"
            f"_{self._last_result.weather}"
        )
        speed_path = os.path.join(OUTPUT_DIR, f"{slug}_speedmap.png")
        tel_path = os.path.join(OUTPUT_DIR, f"{slug}_telemetry.png")
        self.speed_fig.savefig(speed_path, dpi=140)
        self.tel_fig.savefig(tel_path, dpi=140)
        extra = ""
        if self._corner_idx:
            inset_path = os.path.join(OUTPUT_DIR, f"{slug}_corner.png")
            self.inset_fig.savefig(inset_path, dpi=140)
            extra = f" and {inset_path}"
        self.statusBar().showMessage(f"Saved {speed_path} and {tel_path}{extra}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Vehicle Racing Optimizer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
