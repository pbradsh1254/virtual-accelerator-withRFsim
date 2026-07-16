import sys
import time
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from virtaccl.server import not_ctrlc

# Keep OpenGL off by default -- it's not needed for line plots at this data
# rate and it adds a dependency (PyOpenGL) plus driver-compatibility risk.
# Flip to True if you start pushing very large point counts per curve.
pg.setConfigOptions(antialias=False, useOpenGL=False, background='w', foreground='k')

_app = None


def _get_app():
    """One QApplication per process, shared by every LiveDashboard instance."""
    global _app
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    _app = app
    return app


class LiveDashboard:
    """
    Live-updating multi-line dashboard for a set of named devices (cavities or
    BPMs), with a radio-button list on the left to pick which device is shown.

    This is a pyqtgraph reimplementation of the original matplotlib version.
    The matplotlib version needed a manual "blit" trick (save a snapshot of
    the axes background, restore it, draw only the line artists, push just
    that region to screen) to avoid the cost of a full canvas.draw() on every
    step. pyqtgraph's PlotDataItem.setData() already only touches that
    curve's own graphics item and lets Qt's scene graph figure out the
    minimal repaint -- so none of that bookkeeping is needed here, and
    updates are considerably faster.

    `refresh_every` and `rescale_every` are kept for API compatibility (and
    because Qt repaints / autoRange calls are still not free at high call
    rates), but you can push every simulation step regardless -- rendering
    is throttled internally.
    """

    def __init__(self, names, titles, colors, data_keys, phase_keys=None,
                 layout=(2, 2), figsize=(1000, 750), refresh_every=1, rescale_every=5,
                 window_title=None):
        self.data_keys = data_keys
        self.phase_keys = set(phase_keys or [])
        self.refresh_every = max(1, refresh_every)
        self.rescale_every = max(1, rescale_every)

        self._push_count = 0
        self._render_count = 0
        self._latest_history = {}

        self.app = _get_app()
        self._build_ui(sorted(names), layout, titles, colors, figsize, window_title)

    # ---- UI setup -------------------------------------------------

    def _build_ui(self, names, layout, titles, colors, figsize, window_title):
        rows, cols = layout

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(window_title or "Live Dashboard")
        self.window.resize(*figsize)
        outer_layout = QtWidgets.QHBoxLayout(self.window)

        # ---- device selector (radio-button list, scrollable for long lists) ----
        selector_container = QtWidgets.QWidget()
        selector_container.setFixedWidth(220)
        selector_layout = QtWidgets.QVBoxLayout(selector_container)
        selector_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        radio_holder = QtWidgets.QWidget()
        radio_vbox = QtWidgets.QVBoxLayout(radio_holder)

        self.radio_buttons = {}
        self._radio_group = QtWidgets.QButtonGroup(radio_holder)
        self._radio_group.setExclusive(True)
        for i, name in enumerate(names):
            btn = QtWidgets.QRadioButton(name)
            if i == 0:
                btn.setChecked(True)
            btn.toggled.connect(self._make_select_handler(name))
            radio_vbox.addWidget(btn)
            self._radio_group.addButton(btn)
            self.radio_buttons[name] = btn
        radio_vbox.addStretch(1)
        scroll.setWidget(radio_holder)
        selector_layout.addWidget(scroll)
        outer_layout.addWidget(selector_container)

        # ---- title + plot grid ----
        plot_container = QtWidgets.QWidget()
        plot_vbox = QtWidgets.QVBoxLayout(plot_container)
        plot_vbox.setContentsMargins(4, 4, 4, 4)

        self.title_label = QtWidgets.QLabel()
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(13)
        font.setBold(True)
        self.title_label.setFont(font)
        plot_vbox.addWidget(self.title_label)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        plot_vbox.addWidget(self.graphics_layout)
        outer_layout.addWidget(plot_container, stretch=1)

        self.plots = []
        self.curves = []
        for idx, (title, color) in enumerate(zip(titles, colors)):
            row, col = divmod(idx, cols)
            plot_item = self.graphics_layout.addPlot(row=row, col=col, title=title)
            plot_item.showGrid(x=True, y=True, alpha=0.3)
            curve = plot_item.plot([], [], pen=pg.mkPen(color=color, width=1.5))
            self.plots.append(plot_item)
            self.curves.append(curve)

        self.selected_name = names[0]
        self.title_label.setText(f"Monitoring: {self.selected_name}")

        self.window.show()
        self.app.processEvents()

    def _make_select_handler(self, name):
        def handler(checked):
            if checked:
                self._on_select(name)
        return handler

    # ---- device switching -----------------------------------------------

    def _on_select(self, name):
        self.selected_name = name
        self.title_label.setText(f"Monitoring: {self.selected_name}")
        # Switching devices can require a very different axis scale, so
        # always do a full autoRange here instead of the throttled path.
        self._render(self._latest_history, force_full_redraw=True)

    # ---- public update calls ---------------------------------------------

    def push(self, history):
        """Call once per simulation step. Rendering is throttled internally,
        so it is always safe to call this every step."""
        self._latest_history = history
        self._push_count += 1
        if self._push_count % self.refresh_every != 0:
            return

        self._render_count += 1
        need_full_redraw = (self._render_count % self.rescale_every == 1)
        self._render(history, need_full_redraw)

    def finalize(self, history):
        """Call once after the simulation loop ends, with the complete
        history. Converts any list-based history entries to numpy arrays
        and does one last full-quality render. Does not block -- call
        block_until_closed() after finalizing all dashboards."""
        converted = {
            name: {key: np.asarray(values) for key, values in data.items()}
            for name, data in history.items()
        }
        self._latest_history = converted
        self._render(converted, force_full_redraw=True)

    # ---- rendering ---------------------------------------------------------

    def _render(self, history, force_full_redraw):
        data = history.get(self.selected_name)
        if data is None or len(data['t']) == 0:
            return

        x_data = np.asarray(data['t'])
        for curve, key in zip(self.curves, self.data_keys):
            y_data = np.asarray(data[key])
            if key in self.phase_keys and len(y_data) > 0:
                # np.unwrap removes the artificial +/- pi jumps
                y_data = np.degrees(np.unwrap(y_data))
            curve.setData(x_data, y_data)

        if force_full_redraw:
            for plot_item in self.plots:
                plot_item.enableAutoRange()

        self.app.processEvents()


def block_until_closed():
    """Keep the dashboards open until Ctrl+C is pressed."""
    app = _get_app()

    while not_ctrlc():
        app.processEvents()
        time.sleep(0.05)

    for widget in QtWidgets.QApplication.topLevelWidgets():
        widget.close()

    app.quit()

# ---- data-recording helpers ------------------------------------------------
# These build up history dictionaries using plain Python lists. Appending to
# a list is cheap (amortized constant time). The old version used np.append,
# which reallocates and copies the *entire* array on every single call --
# with 40+ loop iterations that copying cost grows quadratically with the
# number of samples recorded. Lists are only converted to numpy arrays once,
# inside LiveDashboard._render, right before they're plotted.

def record_cav_pulse_step(step_data, all_cavities_history):
    """Append one chain.step()/chain.fill()/chain.decay() result into the
    cavity history dict (in place) and return it. Call
    dashboard.push(all_cavities_history) afterward to update the screen.

    NOTE: this assumes step_data[device]['refl_iq'] exists, analogous to
    'cav_iq' and 'fwd_iq'. If the reflected-wave IQ comes back under a
    different key (or has to be derived from fwd_iq/cav_iq and the coupling
    parameters instead of being provided directly), update the two lines
    marked below accordingly.
    """
    for device_name, data in step_data.items():
        if device_name not in all_cavities_history:
            print('error')
            continue

        hist = all_cavities_history[device_name]
        cav_iq = np.atleast_1d(data['cav_iq'])
        fwd_iq = np.atleast_1d(data['fwd_iq'])
        refl_iq = np.atleast_1d(data['refl_iq'])  # <-- adjust key/derivation here if needed

        hist['t'].extend(np.atleast_1d(data['t']))
        hist['amp'].extend(np.abs(cav_iq))
        hist['phase'].extend(np.angle(cav_iq))
        hist['fwd_amp'].extend(np.abs(fwd_iq))
        hist['fwd_phase'].extend(np.angle(fwd_iq))
        hist['refl_amp'].extend(np.abs(refl_iq))          # <-- adjust key/derivation here if needed
        hist['refl_phase'].extend(np.angle(refl_iq))      # <-- adjust key/derivation here if needed

    return all_cavities_history


def record_beam_pulse_step(new_measurements, pulsetime, all_beams_history):
    """Append one set of BPM measurements into the beam history dict (in
    place) and return it along with the incremented index. Call
    dashboard.push(all_beams_history) afterward to update the screen."""
    for device_name, data in new_measurements.items():
        if 'SCL' in device_name and 'BPM' in device_name:
            if device_name not in all_beams_history:
                all_beams_history[device_name] = {'t': [], 'phase_avg': [], 'current': []}

            all_beams_history[device_name]['t'].append(pulsetime)
            all_beams_history[device_name]['phase_avg'].append(data['phi_avg'])
            all_beams_history[device_name]['current'].append(data['current'])

    return all_beams_history
