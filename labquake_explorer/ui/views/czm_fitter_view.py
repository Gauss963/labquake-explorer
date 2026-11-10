"""Fit a cohesive-zone (slip-weakening) rupture model to near-fault strain.

Two shared-x axes show the shear (Exy, selected gauge) and normal (Eyy,
gauge 14; the lower panel is left empty with a note when the strain block
has fewer channels) strain around one event together with the cohesive-zone
prediction.  Three draggable vertical-line pairs mark the zeroing point for
Eyy (x_min), the rupture tip (x_tip) and the zeroing point / fit end for Exy
(x_max).  ``Fit`` adjusts Gc and Xc by L-BFGS-B over the tip..max window.
Results are saved under ``event['czm_parms']`` as a dict.

A record in volts is turned into strain and compared with the model's
strains, as it always was.  A record already in a stress unit (``unit`` of
``Pa``, ``kPa``, ``MPa`` or ``GPa``: a recorder read through a conversion)
is compared with the model's stresses directly, in MPa on the axes.  Its
channels are grouped by sensor through their names (``<sensor>_sigma_xy``,
``_sigma_xx``, ``_sigma_yy``): the gauge box lists the sensors the record
has by name (``shear_1``, ``rosette_1``, ...: five entries for three shear
gauges and two rosettes, whatever the number of channels), three panels show the sensor's xy, xx and yy
components, and ``Fit`` adjusts Gc and Xc to all the components ticked in
the ``Fit`` boxes at once (a rosette: all three; a shear gauge: xy alone).
sigma_xy is zeroed at x_max, sigma_xx and sigma_yy at x_min; the window is
tip..max for every component.  sigma_xx and sigma_yy change sign with the
side of the fault the sensor sits on: a negative ``y`` puts it on the other side.

The elastic medium is the class default (E, nu, C_s, C_d below) unless the
experiment gives its own: ``analysis['medium']`` with ``E_Pa`` and ``nu``
plus either the wave speeds (``vs_m_s``, ``vp_m_s``) or the density
(``rho_kg_m3``) they follow from.  When the record is in stress or the
medium is the experiment's, the saved dict also names the channel, the unit,
the medium used and the peak strength ``tau_p`` implied by Gc and Xc.
"""
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from scipy import optimize, signal

from labquake_explorer.data.data_processor import DataProcessor
from labquake_explorer.data.settings import get_setting
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.data.sources import (pick_waveform_block, read_arrivals, waveform_block, waveform_channels,
                                            waveform_data, waveform_time)
from labquake_explorer.utils.cohesive_crack import CohesiveCrack

STRESS_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}   # Pa per unit of a record stored in stress
COMPONENTS = ("xy", "xx", "yy")                                   # the panels of a stress record, top to bottom
COMPONENT_STYLE = {"xy": "b-", "xx": "m-", "yy": "r-"}
MEDIUM_SETTING = "medium"                                         # analysis['medium']: E_Pa, nu, vs_m_s, vp_m_s, rho_kg_m3


def experiment_medium(data) -> dict | None:
    """``{'E', 'nu', 'C_s', 'C_d'}`` from the experiment's ``analysis['medium']``,
    or None when it does not define an elastic medium: ``E_Pa`` and ``nu`` are
    needed, and either both wave speeds (``vs_m_s``, ``vp_m_s``) or the density
    ``rho_kg_m3``, from which a missing speed follows (C_s = sqrt(G / rho),
    C_d = sqrt(E (1 - nu) / (rho (1 + nu) (1 - 2 nu))))."""
    medium = get_setting(data, MEDIUM_SETTING)
    if not isinstance(medium, dict):
        return None

    def number(key):
        try:
            value = float(np.asarray(medium[key]).ravel()[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    E, nu, rho = number("E_Pa"), number("nu"), number("rho_kg_m3")
    if E is None or nu is None or nu >= 0.5:
        return None
    c_s, c_d = number("vs_m_s"), number("vp_m_s")
    if c_s is None and rho is not None:
        c_s = float(np.sqrt(E / (2 * (1 + nu)) / rho))
    if c_d is None and rho is not None:
        c_d = float(np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu))))
    if c_s is None or c_d is None:
        return None
    return {"E": E, "nu": nu, "C_s": c_s, "C_d": c_d}


@register_view("Fit Cohesive Zone Model", kinds=[EVENT], order=30)
class CZMFitterView(EventView):
    window_title = "Cohesive Zone Model Fitting"
    result_key = "czm_parms"

    # Material properties (the defaults; an experiment may give its own, see ``experiment_medium``)
    E = 51e9      # Young's modulus (Pa)
    nu = 0.25     # Poisson's ratio
    C_s = 2760    # Shear wave speed (m/s)
    C_d = 4790    # Longitudinal wave speed (m/s)

    DEFAULT_GAUGE = 6
    EYY_GAUGE = 14

    def __init__(self, app, run_idx, event_idx):
        self.filtering = False
        self.num_gauges = 0
        self.channel_names = []
        self.stress_scale = None        # Pa per unit of the record when it is stored in stress, else None (volts)
        self.medium_source = "default"  # 'default' (the class constants) or 'experiment' (analysis['medium'])
        self._opened = False            # True once the first event is shown: later events keep the channel in use
        self.x_lim_min, self.x_lim_max = -0.1, 0.1
        self.line_positions = []
        self.vlines = []
        self.vlines_twin = []
        self.vlines_other = []          # the marker lines of the panels below the second one (stress records)
        self.active_line_idx = None
        self.drag_active = False
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.strain_gauge = tk.IntVar(master=self, value=self.DEFAULT_GAUGE)
        self.filter_window = tk.IntVar(master=self, value=51)
        self.Cf = tk.DoubleVar(master=self)
        self.y = tk.DoubleVar(master=self)
        self.Xc = tk.DoubleVar(master=self)
        self.Gc = tk.DoubleVar(master=self)

        self.create_control_frame()
        self.create_matplotlib_figure()
        self.create_parameters_frame()

        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def create_control_frame(self):
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Event selection (columns 0-1)
        self.build_event_selector(control_frame, row=0, column=0)

        # Strain gauge selection
        ttk.Label(control_frame, text="Strain Gauge:").grid(row=0, column=2, padx=5, sticky="w")
        self.gauge_combobox = ttk.Combobox(control_frame, textvariable=self.strain_gauge,
                                           width=5, state="readonly")
        self.gauge_combobox.grid(row=0, column=3, padx=5, sticky="w")
        self.gauge_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        self.sensor_var = tk.StringVar(master=self)        # the box's variable when it lists sensors by name
        self.channel_var = tk.StringVar(master=self)       # the selected channel's name, when the record names it
        ttk.Label(control_frame, textvariable=self.channel_var).grid(row=0, column=6, padx=5, sticky="w")

        # which components of the selected sensor Fit uses (stress records only; hidden otherwise)
        self.fit_frame = ttk.Frame(control_frame)
        self.fit_frame.grid(row=0, column=7, padx=10, sticky="w")
        ttk.Label(self.fit_frame, text="Fit:").pack(side=tk.LEFT, padx=2)
        self.fit_vars, self.fit_boxes = {}, {}
        for component in COMPONENTS:
            self.fit_vars[component] = tk.BooleanVar(master=self, value=True)
            self.fit_boxes[component] = ttk.Checkbutton(self.fit_frame, text=component,
                                                        variable=self.fit_vars[component])
            self.fit_boxes[component].pack(side=tk.LEFT)
        self.fit_frame.grid_remove()

        # Filter controls
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=0, column=4, padx=10, sticky="w")
        ttk.Label(filter_frame, text="Filter Window:").pack(side=tk.LEFT, padx=2)
        self.filter_spinbox = ttk.Spinbox(
            filter_frame, from_=3, to=201, increment=2, textvariable=self.filter_window,
            width=10, validate='focusout',
            validatecommand=(self.register(self.validate_filter_window), '%P'))
        self.filter_spinbox.pack(side=tk.LEFT)
        self.filter_spinbox.bind("<Return>", self.update_plot)

        self.filter_button = tk.Button(control_frame, text="Filter Off", relief="raised",
                                       command=self.toggle_filter)
        self.filter_button.grid(row=0, column=5, padx=5, sticky="w")

    def create_matplotlib_figure(self):
        self.make_figure(figsize=(10, 6), row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.fig = self.figure
        self.gs = self.figure.add_gridspec(2, hspace=0.3)
        self.axs = self.gs.subplots(sharex=True)
        for ax in self.axs:
            ax.grid(True)

    def create_parameters_frame(self):
        params_frame = ttk.Frame(self)
        params_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        param_configs = [
            ("Cf", self.Cf, 10),
            ("y", self.y, 1e-3),
            ("Xc", self.Xc, 1),
            ("Gc", self.Gc, 1),
        ]
        self.param_spinboxes = {}
        for label_text, var, increment in param_configs:
            frame = ttk.Frame(params_frame)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=2)
            spinbox = ttk.Spinbox(frame, textvariable=var, width=10, from_=0, to=1e6, increment=increment)
            spinbox.pack(side=tk.LEFT)
            self.param_spinboxes[label_text] = spinbox

        button_frame = ttk.Frame(params_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        self.update_button = ttk.Button(button_frame, text="Update", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT, padx=5)
        self.fit_button = ttk.Button(button_frame, text="Fit", command=self.fit_parameters)
        self.fit_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_parameters)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.medium_var = tk.StringVar(master=self)        # what is compared, and the elastic medium of the model
        ttk.Label(self, textvariable=self.medium_var, foreground="gray").grid(row=4, column=0, padx=10, pady=(0, 5),
                                                                              sticky="w")

    # -------------------------------------------------------------- loading
    @staticmethod
    def strain_block(event):
        """The event's full-rate record: ``strain`` when present, else the first
        waveform block (``elsys``, ``ni``, ...); None without any."""
        if not isinstance(event, dict):
            return None
        key = pick_waveform_block(event, prefer_keys=("strain",))
        return waveform_block(event, key) if key else None

    @staticmethod
    def strain_raw(event):
        """The event's full-rate samples (2-D, >= 1 channel), else None."""
        strain = CZMFitterView.strain_block(event)
        if strain is None:
            return None
        try:
            raw = waveform_data(strain)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            if np.ndim(raw) != 2 or len(raw) == 0:
                return None
        except TypeError:
            return None
        return raw

    def set_event(self, event_idx: int) -> None:
        """Switch events, but refuse (with a warning) an event without strain data.

        The check happens before ``EventView.set_event`` touches ``event_idx``,
        ``event``, the title or the combobox, so a refused switch leaves the
        view (traces, marker lines, parameters) exactly as it was.
        """
        event_idx = int(event_idx)
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{event_idx}]")
        except (KeyError, IndexError, TypeError):
            candidate = None
        if self.strain_raw(candidate) is None:
            self.refresh_event_selector()  # combobox may already show the refused index
            messagebox.showwarning(
                "No strain data",
                f"Event {event_idx} has no strain data to fit a cohesive zone model to; "
                f"staying on event {self.event_idx}.",
                parent=self,
            )
            return
        super().set_event(event_idx)

    def on_event_loaded(self):
        self._clear_vlines()

        raw = self.strain_raw(self.event)
        if raw is None:
            # first load only (set_event refuses such events): EventView.__init__
            # unregisters and destroys the half-built window before re-raising
            raise ValueError(f"{self.event_path} has no strain data (no waveform record or strain block) "
                             "to fit a cohesive zone model to")
        self.num_gauges = len(raw)
        block = self.strain_block(self.event)
        self.stress_scale = STRESS_UNITS.get(self.record_unit(block))
        names = waveform_channels(block)
        self.channel_names = names if names != [str(i) for i in range(self.num_gauges)] else []
        sensors = self.sensors()
        if sensors:                                        # the sensors the record has, by name
            self.gauge_combobox.config(textvariable=self.sensor_var, values=list(sensors),
                                       width=max(5, max(len(name) for name in sensors) + 1))
        else:                                              # the record's channels, by index
            self.gauge_combobox.config(textvariable=self.strain_gauge, width=5,
                                       values=[str(i) for i in range(self.num_gauges)])
        self.layout_axes(len(COMPONENTS) if self.stress_mode else 2)
        if self.stress_mode:
            self.fit_frame.grid()
        else:
            self.fit_frame.grid_remove()
        self.load_medium()
        default_gauge = self.default_gauge()

        params = self.event.get(self.result_key)
        if isinstance(params, (list, tuple, np.ndarray)) and np.ndim(params) == 1 and len(params) == 8:
            # legacy layout: [Cf, y, Xc, Gc, x_min, x_tip, x_lim_min, x_lim_max]
            # (a list in memory / .npz, an ndarray after an HDF5 round trip)
            params = [float(v) for v in params]
            self._set_parameters(*params[:4])
            x0, x1 = params[4], params[5]
            self.line_positions = [x0, x1, 2 * x1 - x0]
            self.x_lim_min, self.x_lim_max = params[6], params[7]
            self.strain_gauge.set(default_gauge)
        elif isinstance(params, dict):
            self._set_parameters(params['Cf'], params['y'], params['Xc'], params['Gc'])
            self.line_positions = [params['x_min'], params['x_tip'], params['x_max']]
            self.x_lim_min, self.x_lim_max = params['x_lim_min'], params['x_lim_max']
            gauge = params.get('strain_gauge')
            if isinstance(gauge, (int, np.integer)) and 0 <= gauge < self.num_gauges:
                self.strain_gauge.set(int(gauge))
            else:
                self.strain_gauge.set(default_gauge)
            chosen = params.get('fit_components')
            if chosen is not None:
                chosen = {c.decode() if isinstance(c, bytes) else str(c) for c in np.atleast_1d(chosen)}
            for component, var in self.fit_vars.items():
                var.set(chosen is None or component in chosen)
        else:
            self._set_default_parameters()
            self.line_positions = []
            if not 0 <= self.strain_gauge.get() < self.num_gauges or (self.stress_mode and not self._opened):
                self.strain_gauge.set(default_gauge)     # a stress record opens on its first sigma_xy channel
        self.show_gauge()
        self._opened = True

        self.axs[0].set_xlim(self.x_lim_min, self.x_lim_max)
        self.update_plot()

    # ------------------------------------------------------- record and medium
    @staticmethod
    def record_unit(block) -> str:
        """The unit a waveform record is stored in (``'V'`` when it says nothing,
        as the PSU ``strain`` layout)."""
        unit = block.get("unit") if isinstance(block, dict) else None
        if unit is None:
            return "V"
        if isinstance(unit, bytes):
            return unit.decode(errors="replace")
        return str(np.asarray(unit).ravel()[0]) if isinstance(unit, np.ndarray) and unit.size else str(unit)

    @property
    def stress_mode(self) -> bool:
        """True when the record is stored in stress and fitted as stress."""
        return self.stress_scale is not None

    def load_medium(self) -> None:
        """Take E, nu, C_s and C_d from the experiment when it defines them
        (:func:`experiment_medium`), else restore the class defaults."""
        medium = experiment_medium(self.data_manager.data)
        cls = type(self)
        if medium is None:
            self.E, self.nu, self.C_s, self.C_d = cls.E, cls.nu, cls.C_s, cls.C_d
            self.medium_source = "default"
        else:
            self.E, self.nu, self.C_s, self.C_d = medium["E"], medium["nu"], medium["C_s"], medium["C_d"]
            self.medium_source = "experiment"
        compared = (f"stress record ({self.record_unit(self.strain_block(self.event))}) against the model's stress"
                    if self.stress_mode else "strain from the record's volts against the model's strain")
        self.medium_var.set(f"{compared};  medium ({self.medium_source}): E = {self.E / 1e9:.4g} GPa, "
                            f"nu = {self.nu:.3g}, C_s = {self.C_s:.0f} m/s, C_d = {self.C_d:.0f} m/s")

    def channel_name(self, gauge_idx) -> str:
        """The record's name for a channel index, '' when it has none."""
        return self.channel_names[gauge_idx] if 0 <= gauge_idx < len(self.channel_names) else ""

    def sensors(self) -> dict:
        """``{sensor name: index of its sigma_xy channel}`` for a stress record
        whose channels are named by sensor (``<sensor>_sigma_xy``), in channel
        order; empty otherwise.  These are the entries of the gauge box: as
        many as the record has sensors, not channels."""
        if not self.stress_mode:
            return {}
        return {name[:-len("_sigma_xy")]: i for i, name in enumerate(self.channel_names)
                if name.endswith("_sigma_xy")}

    def show_gauge(self) -> None:
        """Make the gauge box show the selection held in ``strain_gauge``: the
        channel index, or the name of the sensor that channel belongs to (the
        first sensor when it belongs to none)."""
        sensors = self.sensors()
        if not sensors:
            self.gauge_combobox.set(self.strain_gauge.get())
            return
        xy = self.sensor_channels()["xy"]
        name = next((n for n, i in sensors.items() if i == xy), next(iter(sensors)))
        self.strain_gauge.set(sensors[name])
        self.sensor_var.set(name)

    def selected_sensor(self) -> str:
        """The name of the selected sensor, '' when the box lists channels."""
        name = self.sensor_var.get()
        return name if name in self.sensors() else ""

    def sensor_channels(self, gauge_idx=None) -> dict:
        """``{component: channel index}`` of the sensor a channel of a stress
        record belongs to: the channels named ``<sensor>_sigma_xy|xx|yy``.  A
        channel not named that way stands alone as ``{'xy': index}``."""
        gauge = int(self.strain_gauge.get()) if gauge_idx is None else int(gauge_idx)
        sensor, sep, component = self.channel_name(gauge).rpartition("_sigma_")
        if not sep or component not in COMPONENTS:
            return {"xy": gauge}
        found = {c: self.channel_names.index(f"{sensor}_sigma_{c}") for c in COMPONENTS
                 if f"{sensor}_sigma_{c}" in self.channel_names}
        return found if "xy" in found else {"xy": gauge}

    def sensor_label(self) -> str:
        """What stands next to the gauge box: the channel's name; for a sensor
        of a stress record, the components it has (``xy, xx, yy``)."""
        if self.selected_sensor():
            channels = self.sensor_channels()
            return ", ".join(c for c in COMPONENTS if c in channels)
        return self.channel_name(int(self.strain_gauge.get()))

    def panels(self) -> list:
        """``(component, channel index or None)`` for each axes, top to bottom:
        Exy and Eyy of a record in volts; sigma_xy, sigma_xx and sigma_yy of the
        selected sensor of a stress record."""
        if self.stress_mode:
            channels = self.sensor_channels()
            return [(c, channels.get(c)) for c in COMPONENTS]
        return [("xy", int(self.strain_gauge.get())), ("yy", self.eyy_gauge())]

    def fit_components(self) -> list:
        """The components ``Fit`` compares with the model: xy alone for a record
        in volts; for a stress record, those of the sensor that are ticked."""
        if not self.stress_mode:
            return ["xy"]
        channels = self.sensor_channels()
        return [c for c in COMPONENTS if c in channels and self.fit_vars[c].get()]

    def layout_axes(self, n: int) -> None:
        """Give the figure ``n`` shared-x panels (2 for volts, 3 for stress),
        keeping the ones it has when the number does not change."""
        if len(self.axs) == n:
            return
        self._clear_vlines()
        self.figure.clear()
        self.gs = self.figure.add_gridspec(n, hspace=0.3)
        self.axs = self.gs.subplots(sharex=True)
        for ax in self.axs:
            ax.grid(True)

    def default_gauge(self) -> int:
        """The channel a fresh event opens on: ``DEFAULT_GAUGE`` (clipped to the
        record); for a stress record, its first ``sigma_xy`` channel when it names one."""
        if self.stress_mode:
            for i, name in enumerate(self.channel_names):
                if "sigma_xy" in name.lower():
                    return i
        return min(self.DEFAULT_GAUGE, self.num_gauges - 1)

    def gauge_distance(self, gauge_idx) -> float | None:
        """Distance of a channel's sensor from the fault (m) from the record's
        ``positions`` table (``y``, in the table's unit), or None."""
        positions = self.strain_block(self.event).get("positions")
        if not isinstance(positions, dict):
            return None
        try:
            y = abs(float(np.asarray(positions["y"], dtype=float).ravel()[gauge_idx]))
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}.get(str(positions.get("unit", "mm")), 1e-3)
        return y * scale if np.isfinite(y) and y > 0 else None

    def _set_parameters(self, Cf, y, Xc, Gc):
        self.Cf.set(Cf)
        self.y.set(y)
        self.Xc.set(Xc)
        self.Gc.set(Gc)

    def _set_default_parameters(self):
        self.x_lim_min, self.x_lim_max = -0.1, 0.1
        try:
            speed = read_arrivals(self.strain_block(self.event), self.event)["rupture_speed"]
            self.Cf.set(float(np.abs(speed)))
        except (KeyError, TypeError, ValueError):
            self.Cf.set(10)
        self.y.set(8e-3)
        self.Xc.set(1)
        self.Gc.set(1)
        if self.stress_mode:                               # what the record itself says about the sensor and its span
            distance = self.gauge_distance(self.default_gauge())
            if distance is not None:
                self.y.set(distance)
            t = self._time()
            if t.size and t[0] > self.x_lim_min and t[-1] < self.x_lim_max:
                self.x_lim_min, self.x_lim_max = float(t[0]), float(t[-1])

    def _clear_vlines(self):
        for line in self.vlines + self.vlines_twin + [l for lines in self.vlines_other for l in lines]:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.vlines = []
        self.vlines_twin = []
        self.vlines_other = []

    def current_line_positions(self):
        """x positions of the three marker lines (from the artists when present)."""
        if self.vlines:
            return [float(line.get_xdata()[0]) for line in self.vlines]
        if self.line_positions:
            return list(self.line_positions)
        # evenly spaced across the view: 5 points, keep the middle 3
        return list(np.linspace(self.x_lim_min, self.x_lim_max, 5)[1:-1])

    # --------------------------------------------------------------- filter
    def toggle_filter(self):
        self.filtering = not self.filtering
        if self.filtering:
            self.filter_button.config(text="Filter On", relief="sunken")
        else:
            self.filter_button.config(text="Filter Off", relief="raised")
        self.update_plot()

    def validate_filter_window(self, value):
        """Validate that the filter window value is an odd integer."""
        if value == "":  # Allow empty field for editing
            return True
        try:
            val = int(value)
            return 3 <= val <= 201 and val % 2 == 1
        except ValueError:
            return False

    def _filter_window_length(self):
        window_length = self.filter_window.get()
        if window_length % 2 == 0:
            window_length += 1
            self.filter_window.set(window_length)
        return window_length

    def _strain(self, gauge_idx):
        """One channel as it is fitted, Savitzky-Golay filtered when filtering is
        on: strain from a record in volts, stress in Pa from a record in stress."""
        raw = waveform_data(self.strain_block(self.event))[gauge_idx]
        if self.stress_mode:
            strain = np.asarray(raw, dtype=float) * self.stress_scale
        else:
            strain = DataProcessor.voltage_to_strain(raw)
        if self.filtering:
            strain = signal.savgol_filter(strain, self._filter_window_length(), 2)
        return strain

    def eyy_gauge(self):
        """Index of the channel of the lower panel, or None: channel ``EYY_GAUGE``
        of a record in volts; for a stress record, the ``sigma_yy`` channel of
        the selected sensor (the selected name with ``sigma_xy`` -> ``sigma_yy``)."""
        if self.stress_mode:
            return self.sensor_channels().get("yy")
        return self.EYY_GAUGE if 0 <= self.EYY_GAUGE < self.num_gauges else None

    def _time(self):
        return waveform_time(self.strain_block(self.event)) - self.event["event_time"]

    def _model_components(self, t, x_tip, Xc, Gc) -> dict:
        """Cohesive-zone prediction along the gauge line at times ``t`` as
        ``{'xy', 'xx', 'yy'}``, in the quantity of :meth:`_strain`: strains, or
        stresses in Pa for a stress record."""
        Cf = self.Cf.get()
        x_zeroed = -t * Cf + x_tip * Cf  # position relative to the tip, in meters
        delta_sigma_xx, delta_sigma_xy, delta_sigma_yy = CohesiveCrack.delta_sigmas(
            x_zeroed, self.y.get(), Xc, Cf, self.C_s, self.C_d, self.nu, Gc, self.E)
        if self.stress_mode:
            return {"xy": delta_sigma_xy, "xx": delta_sigma_xx, "yy": delta_sigma_yy}
        delta_e_xx, delta_e_xy, delta_e_yy = DataProcessor.stress_to_strain(
            self.E, self.nu, delta_sigma_xx, delta_sigma_xy, delta_sigma_yy)
        return {"xy": delta_e_xy, "xx": delta_e_xx, "yy": delta_e_yy}

    def _model_strains(self, t, x_tip, Xc, Gc):
        """The (xy, yy) pair of :meth:`_model_components`."""
        model = self._model_components(t, x_tip, Xc, Gc)
        return model["xy"], model["yy"]

    # ----------------------------------------------------------------- plot
    def update_plot(self, event=None):
        line_positions = self.current_line_positions()

        xlim_temp = self.axs[0].get_xlim()
        for ax in self.axs:
            ax.clear()

        t = self._time()
        sensor = self.selected_sensor()
        if sensor:                                         # the box holds a sensor's name: its sigma_xy channel
            self.strain_gauge.set(self.sensors()[sensor])
        gauge = self.strain_gauge.get()
        self.channel_var.set(self.sensor_label())
        if self.stress_mode:                               # only the components the sensor has can be ticked
            available = self.sensor_channels()
            for component, box in self.fit_boxes.items():
                box.state(["!disabled"] if component in available else ["disabled"])
        scale = 1e-6 if self.stress_mode else 1.0          # a stress record is drawn in MPa; strains as they are

        # xy is zeroed at x_max (the level behind the front), the other components at x_min (before it arrives)
        idx_zero = {"xy": int(np.argmin(np.abs(t - line_positions[2])))}
        idx_zero["xx"] = idx_zero["yy"] = int(np.argmin(np.abs(t - line_positions[0])))
        model = self._model_components(t, line_positions[1], self.Xc.get(), self.Gc.get())
        for ax, (component, channel) in zip(self.axs, self.panels()):
            label = f"sigma_{component}" if self.stress_mode else f"E{component}"
            zero = idx_zero[component]
            if channel is not None:
                measured = self._strain(channel)
                ax.plot(t, (measured - measured[zero]) * scale, COMPONENT_STYLE[component], label=label)
                ax.plot(t, (model[component] - model[component][zero]) * scale, 'g--', label='CZM')
            elif self.stress_mode:
                ax.text(0.5, 0.5, f"no {label} channel for {sensor or self.channel_name(gauge) or f'channel {gauge}'}",
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.text(0.5, 0.5, f"no Eyy gauge: channel {self.EYY_GAUGE} missing "
                                  f"({self.num_gauges} strain channels)",
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(f"{label} (MPa)" if self.stress_mode else label)
        self.axs[-1].set_xlabel('Time (s)')
        self.figure.suptitle(self.figure_title())

        self.vlines = []
        self.vlines_twin = []
        self.vlines_other = [[] for _ in self.axs[2:]]
        for x_pos in line_positions:
            self.vlines.append(self.axs[0].axvline(x=x_pos, color='g', linestyle='--', alpha=0.5))
            self.vlines_twin.append(self.axs[1].axvline(x=x_pos, color='g', linestyle='--', alpha=0.5))
            for ax, lines in zip(self.axs[2:], self.vlines_other):
                lines.append(ax.axvline(x=x_pos, color='g', linestyle='--', alpha=0.5))
        self.line_positions = list(line_positions)

        for ax in self.axs:
            ax.grid(True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        self.axs[0].set_xlim(xlim_temp)
        self.canvas.draw()

    # ------------------------------------------------------------- dragging
    def on_mouse_press(self, event):
        if self.toolbar_active() or event.button != 1 or not event.inaxes or event.xdata is None:
            return
        xlim = self.axs[0].get_xlim()
        tolerance = 0.01 * (xlim[1] - xlim[0])
        for i, vline in enumerate(self.vlines):
            if abs(event.xdata - vline.get_xdata()[0]) < tolerance:
                self.drag_active = True
                self.active_line_idx = i
                break

    def on_mouse_release(self, event):
        self.drag_active = False
        self.active_line_idx = None

    def on_mouse_move(self, event):
        if self.toolbar_active():
            return
        if self.drag_active and event.inaxes and self.active_line_idx is not None:
            self.move_line(self.active_line_idx, event.xdata)

    def move_line(self, idx, new_x):
        """Move marker pair ``idx`` to ``new_x`` and redraw."""
        new_x = float(new_x)
        self.vlines[idx].set_xdata([new_x, new_x])
        self.vlines_twin[idx].set_xdata([new_x, new_x])
        for lines in self.vlines_other:
            lines[idx].set_xdata([new_x, new_x])
        self.update_plot()

    # ------------------------------------------------------------------ fit
    def build_fit_objective(self):
        """Return ``objective([Gc, Xc])`` for the data between the tip and max lines.

        A record in volts: the sum of squared residuals (in nano-strain)
        between the zeroed measured Exy and the cohesive-zone prediction over
        the tip..max window.  A stress record: the same sum (in kPa) over every
        component of :meth:`fit_components` at once, sigma_xy zeroed at x_max
        and sigma_xx, sigma_yy at x_min, as they are drawn.  Returns None when
        the three marker lines are not available, no samples lie inside the
        window, or no component is ticked.
        """
        if len(self.vlines) < 3:
            print("Need 3 vertical lines to define fitting region")
            return None
        t0, t1, t2 = sorted(self.current_line_positions())

        t = self._time()
        exy = self._strain(self.strain_gauge.get())
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            print("No samples between the tip and max lines")
            return None
        t_fit = t[mask]
        if self.stress_mode:
            return self.build_joint_objective(t, t_fit, mask, t0, t1, t2)
        exy_fit = exy[mask]
        idx_zero = int(np.argmin(np.abs(t_fit - t2)))
        exy_fit = exy_fit - exy_fit[idx_zero]

        residual_scale = 1e9   # strain -> nano-strain

        def objective(params):
            Gc, Xc = params
            delta_e_xy, _ = self._model_strains(t_fit, t1, Xc, Gc)
            delta_e_xy = delta_e_xy - delta_e_xy[idx_zero]
            return float(np.sum(((exy_fit - delta_e_xy) * residual_scale) ** 2))

        return objective

    def build_joint_objective(self, t, t_fit, mask, t0, t1, t2):
        """The objective of a stress record: every ticked component of the
        selected sensor over the window, each minus its value at its zero line
        (xy: ``t2``, x_max; xx and yy: ``t0``, x_min), residuals in kPa."""
        components = self.fit_components()
        if not components:
            print("No component ticked to fit")
            return None
        channels = self.sensor_channels()
        i_min, i_max = int(np.argmin(np.abs(t - t0))), int(np.argmin(np.abs(t - t2)))
        targets = {}
        for component in components:
            measured = self._strain(channels[component])
            targets[component] = measured[mask] - measured[i_max if component == "xy" else i_min]
        t_model = np.concatenate([t_fit, [t[i_min], t[i_max]]])     # the window, then the two zero points

        def objective(params):
            Gc, Xc = params
            model = self._model_components(t_model, t1, Xc, Gc)
            total = 0.0
            for component in components:
                values = model[component]
                zeroed = values[:-2] - (values[-1] if component == "xy" else values[-2])
                total += float(np.sum(((targets[component] - zeroed) * 1e-3) ** 2))    # Pa -> kPa
            return total

        return objective

    def fit_parameters(self):
        """Fit Gc and Xc to the data between the tip and max lines (see
        :meth:`build_fit_objective`).

        Returns the scipy ``OptimizeResult`` (with ``initial_fun`` added: the
        objective at the starting guess), or None if no fit could be set up.
        """
        objective = self.build_fit_objective()
        if objective is None:
            return None

        initial_guess = [self.Gc.get(), self.Xc.get()]
        bounds = ((1e-6, None), (1e-6, None))
        result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        result.initial_fun = objective(initial_guess)

        if result.success:
            self.Gc.set(float(result.x[0]))
            self.Xc.set(float(result.x[1]))
            self.update_plot()
            print(f"Fitted parameters: Gc={result.x[0]:.2e}, Xc={result.x[1]:.2f}")
        else:
            print("Fitting failed:", result.message)
        return result

    # ----------------------------------------------------------------- save
    def collect_results(self):
        x0, x1, x2 = self.current_line_positions()
        self.x_lim_min, self.x_lim_max = (float(v) for v in self.axs[0].get_xlim())
        results = self.fit_parameters_dict(x0, x1, x2)
        if self.stress_mode or self.medium_source == "experiment":
            results.update(self.fit_context())
        return results

    def fit_context(self) -> dict:
        """What the parameters were fitted to and with: the channel's name (the
        sensor's ``sigma_xy`` channel for a stress record, with the ``sensor``
        itself), the record's unit,
        the elastic medium, the components fitted together (stress records)
        and the peak strength ``tau_p`` (Pa) that Gc and Xc imply at this Cf."""
        gauge = int(self.strain_gauge.get())
        if self.stress_mode:
            gauge = self.sensor_channels(gauge)["xy"]
        context = {"channel": self.channel_name(gauge), "unit": self.record_unit(self.strain_block(self.event)),
                   "E": float(self.E), "nu": float(self.nu), "C_s": float(self.C_s), "C_d": float(self.C_d)}
        if self.selected_sensor():
            context["sensor"] = self.selected_sensor()
        if self.stress_mode and self.fit_components():
            context["fit_components"] = self.fit_components()
        with np.errstate(all="ignore"):
            Cf = self.Cf.get()
            alpha_s, alpha_d = CohesiveCrack.alpha_s(Cf, self.C_s), CohesiveCrack.alpha_d(Cf, self.C_d)
            A2 = CohesiveCrack.compute_A2(Cf, self.C_s, self.nu, CohesiveCrack.D(alpha_s, alpha_d))
            tau_p = CohesiveCrack.compute_tau_p(CohesiveCrack.compute_K2(self.Gc.get(), self.E, self.nu, A2),
                                                self.Xc.get())
        if np.isfinite(tau_p):
            context["tau_p"] = float(tau_p)
        return context

    def fit_parameters_dict(self, x0, x1, x2) -> dict:
        return {
            'Cf': self.Cf.get(),
            'y': self.y.get(),
            'Xc': self.Xc.get(),
            'Gc': self.Gc.get(),
            'x_min': x0,
            'x_tip': x1,
            'x_max': x2,
            'x_lim_min': self.x_lim_min,
            'x_lim_max': self.x_lim_max,
            'strain_gauge': int(self.strain_gauge.get()),
        }

    def save_parameters(self):
        """Save the current parameters to the event data."""
        if len(self.vlines) < 3:
            return
        results = self.collect_results()
        self.save_results(results)
        print(f"Saved parameters for event {self.event_idx}: {results}")
