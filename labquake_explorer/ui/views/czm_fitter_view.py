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
sigma_xx and sigma_yy change sign with the side of the fault the sensor
sits on: a negative ``y`` puts it on the other side.

Lines and levels of a stress record.  The three marker lines are named on
the plot by what they do, left to right: ``background end``, ``tip`` (the
rupture front reaches the sensor: the model's origin) and ``fit end``.  The
grey band, from the dotted ``background start`` line to ``background end``,
is the background window: the mean of each component over it is removed, so
every trace starts at 0 MPa.  The red band, ``tip`` to ``fit end``, is the
fit window of every component; the model is drawn black inside it and as a
thin dashed line outside.  The model's sigma_xx and sigma_yy are referred to
the background window.  The model's sigma_xy is zero at the residual level
behind the front, which the record does not give, so it is shifted to match
the measured trace best over the fit window.  The line under the parameters
gives the misfit over the fit window.  ``Auto`` gives a starting point: it
takes ``Cf`` from the saved arrivals or from the fronts at the record's
sensors (kept below the Rayleigh speed), places the lines on the selected
sensor's front, derives Xc and Gc from the duration and size of the drop,
and runs the fit; on a sensor without a clear front it changes nothing.

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
STRESS_BOUNDS = ((1e-6, 1e9), (1e-6, 1e3))                        # Gc (J/m^2) and Xc (m) a stress record's fit may reach
COMPONENT_STYLE = {"xy": "b-", "xx": "m-", "yy": "r-"}
MODEL_STYLE = {"color": "k", "linestyle": "-", "linewidth": 1.8, "zorder": 5}   # the CZM curve, on top of the data
MODEL_OUTSIDE_STYLE = {"color": "0.35", "linestyle": "--", "linewidth": 1.0, "zorder": 4}   # ... outside the fit window
MARKER_STYLE = {"color": "g", "linestyle": "--", "alpha": 0.5}    # the three marker lines of a record in volts
# the marker lines of a stress record, left to right: name, style
LINE_ROLES = (("background end", {"color": "0.35", "linestyle": "--", "alpha": 0.9}),
              ("tip", {"color": "crimson", "linestyle": "-", "linewidth": 1.4, "alpha": 0.9}),
              ("fit end", {"color": "crimson", "linestyle": "--", "alpha": 0.9}))
STRESS_HINT = ("grey band: background, its mean is removed (drag its two edges)   |   red band: fit window, from 'tip' "
               "(the front reaches the sensor) to 'fit end'   |   black: the model, shifted to match sigma_xy inside it")
CLEAR_FRONT = 8.0                                                 # Auto wants a front this many times the noise
CF_LIMIT = 0.98                                                   # Auto keeps Cf below this fraction of the Rayleigh speed


def rayleigh_speed(c_s: float, c_d: float) -> float:
    """The Rayleigh wave speed of a medium: the speed below ``c_s`` at which
    ``D = 4 alpha_s alpha_d - (1 + alpha_s^2)^2`` vanishes.  The steady
    cohesive-zone solution exists only for rupture speeds below it."""
    def d(c):
        a_s, a_d = np.sqrt(1 - (c / c_s) ** 2), np.sqrt(1 - (c / c_d) ** 2)
        return 4 * a_s * a_d - (1 + a_s ** 2) ** 2
    return float(optimize.brentq(d, 0.3 * c_s, c_s * (1 - 1e-9)))
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
        self.x_base = None              # where the background window of a stress record starts (None: the record's start)
        self.auto_note = ""             # what Auto did about Cf, kept in the result line until the event changes
        self.base_lines = []            # the dotted line at x_base, one per panel
        self.drag_base = False
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
        self.auto_button = ttk.Button(button_frame, text="Auto", command=self.auto_fit)   # stress records only
        self.update_button = ttk.Button(button_frame, text="Update", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT, padx=5)
        self.fit_button = ttk.Button(button_frame, text="Fit", command=self.fit_parameters)
        self.fit_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_parameters)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.result_var = tk.StringVar(master=self)        # stress records: the misfit over the fit window, Auto's notes
        ttk.Label(self, textvariable=self.result_var).grid(row=4, column=0, padx=10, pady=(0, 2), sticky="w")
        self.hint_var = tk.StringVar(master=self)          # stress records: what the bands and lines are
        ttk.Label(self, textvariable=self.hint_var, foreground="gray").grid(row=5, column=0, padx=10, pady=(0, 2),
                                                                            sticky="w")
        self.medium_var = tk.StringVar(master=self)        # what is compared, and the elastic medium of the model
        ttk.Label(self, textvariable=self.medium_var, foreground="gray").grid(row=6, column=0, padx=10, pady=(0, 5),
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
            self.auto_button.pack(side=tk.LEFT, padx=5, before=self.update_button)
        else:
            self.fit_frame.grid_remove()
            self.auto_button.pack_forget()
        self.x_base = None
        self.auto_note = ""
        self.hint_var.set(STRESS_HINT if self.stress_mode else "")
        self.result_var.set("")
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
            try:
                self.x_base = float(params['x_base']) if self.stress_mode and 'x_base' in params else None
            except (TypeError, ValueError):
                self.x_base = None
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
        text = (f"{compared};  medium ({self.medium_source}): E = {self.E / 1e9:.4g} GPa, "
                f"nu = {self.nu:.3g}, C_s = {self.C_s:.0f} m/s, C_d = {self.C_d:.0f} m/s")
        if self.stress_mode:
            text += f", C_R = {rayleigh_speed(self.C_s, self.C_d):.0f} m/s (Cf must stay below it)"
        self.medium_var.set(text)

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

    def sync_gauge(self) -> str:
        """Make ``strain_gauge`` follow the gauge box when it lists sensors (the
        selected sensor's sigma_xy channel); returns the sensor's name, or ''."""
        sensor = self.selected_sensor()
        if sensor:
            self.strain_gauge.set(self.sensors()[sensor])
        return sensor

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
        for line in (self.vlines + self.vlines_twin + [l for lines in self.vlines_other for l in lines]
                     + self.base_lines):
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.vlines = []
        self.vlines_twin = []
        self.vlines_other = []
        self.base_lines = []

    def set_lines(self, positions) -> None:
        """Put the three marker lines at ``positions`` (the artists when drawn)."""
        positions = [float(p) for p in positions]
        self.line_positions = list(positions)
        for lines in [self.vlines, self.vlines_twin] + list(self.vlines_other):
            for line, x in zip(lines, positions):
                line.set_xdata([x, x])

    # ------------------------------------------------ levels of a stress record
    def baseline_window(self, t, positions) -> tuple:
        """``(start, end)`` of the background window: from ``x_base`` (the
        record's start when it was never moved) to the left marker line."""
        end = min(positions)
        start = float(t[0]) if self.x_base is None else float(self.x_base)
        return (start, end) if start <= end else (end, start)

    def baseline_mask(self, t, positions) -> np.ndarray:
        """The samples of the background window; the one nearest to its end
        when the window holds none."""
        start, end = self.baseline_window(t, positions)
        mask = (t >= start) & (t <= end)
        if not mask.any():
            mask[int(np.argmin(np.abs(t - end)))] = True
        return mask

    def level_window(self, index: int) -> slice:
        """The samples a level is read over at one line: the Filter Window
        centred on ``index`` (noise on a single sample would shift the curve)."""
        half = max(self._filter_window_length() // 2, 0)
        return slice(max(index - half, 0), index + half + 1)

    def referenced(self, component, measured, model, base, fit, i_max):
        """A stress component and its model on the levels they are compared on:
        the measured trace minus its background (mean over ``base``); the
        model's sigma_xx and sigma_yy minus theirs; the model's sigma_xy
        shifted to the measured trace's mean over the fit window ``fit`` (when
        that holds no sample: attached at ``i_max``, x_max, over
        :meth:`level_window`)."""
        data = measured - float(np.mean(measured[base]))
        if component == "xy":
            window = fit if np.any(fit) else self.level_window(i_max)
            return data, model - float(np.mean(model[window])) + float(np.mean(data[window]))
        return data, model - float(np.mean(model[base]))

    def line_styles(self, positions) -> list:
        """``(name, style)`` for each of the three marker lines, in the order
        they are held: a stress record names them by position, left to right
        (:data:`LINE_ROLES`); a record in volts draws them alike, unnamed."""
        if not self.stress_mode:
            return [("", MARKER_STYLE)] * len(positions)
        rank = {int(i): r for r, i in enumerate(np.argsort(positions, kind="stable"))}
        return [LINE_ROLES[rank[i]] for i in range(len(positions))]

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
        sensor = self.sync_gauge()
        gauge = self.strain_gauge.get()
        self.channel_var.set(self.sensor_label())
        if self.stress_mode:                               # only the components the sensor has can be ticked
            available = self.sensor_channels()
            for component, box in self.fit_boxes.items():
                box.state(["!disabled"] if component in available else ["disabled"])
        scale = 1e-6 if self.stress_mode else 1.0          # a stress record is drawn in MPa; strains as they are

        if self.stress_mode:
            # every trace minus its background; see ``referenced`` for the model's levels
            x_min, x_tip, x_max = sorted(line_positions)
            base = self.baseline_mask(t, line_positions)
            fit = (t >= x_tip) & (t <= x_max)
            i_max = int(np.argmin(np.abs(t - x_max)))
        else:
            # Exy is zeroed at x_max (the level behind the front), Eyy at x_min (before it arrives)
            x_tip = line_positions[1]
            idx_zero = {"xy": int(np.argmin(np.abs(t - line_positions[2])))}
            idx_zero["xx"] = idx_zero["yy"] = int(np.argmin(np.abs(t - line_positions[0])))
        with np.errstate(all="ignore"):
            model = self._model_components(t, x_tip, self.Xc.get(), self.Gc.get())
        for ax, (component, channel) in zip(self.axs, self.panels()):
            label = f"sigma_{component}" if self.stress_mode else f"E{component}"
            if channel is not None:
                measured = self._strain(channel)
                if self.stress_mode:
                    data, curve = self.referenced(component, measured, model[component], base, fit, i_max)
                else:
                    zero = idx_zero[component]
                    data, curve = measured - measured[zero], model[component] - model[component][zero]
                ax.plot(t, data * scale, COMPONENT_STYLE[component], label=label, linewidth=0.8, zorder=2)
                if self.stress_mode and fit.any():         # the model counts inside the fit window: faint outside it
                    ax.plot(t, curve * scale, **MODEL_OUTSIDE_STYLE)
                    ax.plot(t[fit], curve[fit] * scale, label='CZM', **MODEL_STYLE)
                else:
                    ax.plot(t, curve * scale, label='CZM', **MODEL_STYLE)
            elif self.stress_mode:
                ax.text(0.5, 0.5, f"no {label} channel for {sensor or self.channel_name(gauge) or f'channel {gauge}'}",
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.text(0.5, 0.5, f"no Eyy gauge: channel {self.EYY_GAUGE} missing "
                                  f"({self.num_gauges} strain channels)",
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_ylabel(f"{label} (MPa)" if self.stress_mode else label)
        self.base_lines = []
        if self.stress_mode:
            start, end = self.baseline_window(t, line_positions)
            for ax in self.axs:                            # the background window (grey) with its movable start,
                ax.axvspan(start, end, color='0.85', alpha=0.6, zorder=0)        # and the fit window (red)
                ax.axvspan(x_tip, x_max, color='crimson', alpha=0.08, zorder=0)
                self.base_lines.append(ax.axvline(x=start, color='0.35', linestyle=':'))
            top = self.axs[0]                              # the lines' names, along the top of the top panel
            named = [(start, "background start", "left", "0.35")]
            named += [(x, name, "left" if name == "fit end" else "right", style["color"])
                      for x, (name, style) in zip(line_positions, self.line_styles(line_positions))]
            for x, name, side, color in named:
                top.text(x, 0.98, f" {name} ", transform=top.get_xaxis_transform(), ha=side, va='top',
                         fontsize=9, color=color, clip_on=True, zorder=6,
                         bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.5})
            if not np.isfinite(model["xy"]).any():
                self.axs[0].text(0.5, 0.9, f"no model: Cf = {self.Cf.get():g} m/s is not below the Rayleigh speed "
                                           f"({rayleigh_speed(self.C_s, self.C_d):.0f} m/s)",
                                 ha='center', va='center', transform=self.axs[0].transAxes, color='crimson')
        self.axs[-1].set_xlabel('Time (s)')
        self.figure.suptitle(self.figure_title())

        self.vlines = []
        self.vlines_twin = []
        self.vlines_other = [[] for _ in self.axs[2:]]
        for x_pos, (_, style) in zip(line_positions, self.line_styles(line_positions)):
            self.vlines.append(self.axs[0].axvline(x=x_pos, **style))
            self.vlines_twin.append(self.axs[1].axvline(x=x_pos, **style))
            for ax, lines in zip(self.axs[2:], self.vlines_other):
                lines.append(ax.axvline(x=x_pos, **style))
        self.line_positions = list(line_positions)
        if self.stress_mode:
            self.result_var.set(self.result_text())

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
        else:
            if self.base_lines and abs(event.xdata - self.base_lines[0].get_xdata()[0]) < tolerance:
                self.drag_base = True                      # the start of the background window

    def on_mouse_release(self, event):
        self.drag_active = False
        self.drag_base = False
        self.active_line_idx = None

    def on_mouse_move(self, event):
        if self.toolbar_active():
            return
        if self.drag_active and event.inaxes and self.active_line_idx is not None:
            self.move_line(self.active_line_idx, event.xdata)
        elif self.drag_base and event.inaxes and event.xdata is not None:
            self.move_baseline(event.xdata)

    def move_baseline(self, new_x) -> None:
        """Move the start of the background window to ``new_x`` and redraw."""
        self.x_base = float(new_x)
        self.update_plot()

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

        self.sync_gauge()
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
        selected sensor over the window, on the levels they are drawn on
        (:meth:`referenced`), residuals in kPa.  ``n_values`` on the returned
        function is the number of values it sums."""
        components = self.fit_components()
        if not components:
            print("No component ticked to fit")
            return None
        channels = self.sensor_channels()
        base = self.baseline_mask(t, [t0, t1, t2])
        i_max = int(np.argmin(np.abs(t - t2)))
        targets = {}
        for component in components:
            measured = self._strain(channels[component])
            targets[component] = (measured - float(np.mean(measured[base])))[mask]
        t_base = t[base]
        t_model = np.concatenate([t_fit, t_base])                   # the fit window, then the background window
        n = t_fit.size

        def objective(params):
            Gc, Xc = params
            with np.errstate(all="ignore"):
                model = self._model_components(t_model, t1, Xc, Gc)
            total = 0.0
            for component in components:
                values = model[component]
                if component == "xy":       # shifted to the measured mean over the fit window
                    curve = values[:n] - float(np.mean(values[:n])) + float(np.mean(targets[component]))
                else:                       # referred to the background window, as the data
                    curve = values[:n] - float(np.mean(values[n:]))
                total += float(np.sum(((targets[component] - curve) * 1e-3) ** 2))    # Pa -> kPa
            return total

        objective.n_values = n * len(components)
        return objective

    def result_text(self) -> str:
        """The misfit of the current parameters over the fit window of a stress
        record, with the peak strength they imply and Auto's note on ``Cf``."""
        if len(self.vlines) < 3:
            return self.auto_note
        t0, t1, t2 = sorted(self.current_line_positions())
        t = self._time()
        mask = (t >= t1) & (t <= t2)
        components = self.fit_components()
        if not mask.any() or not components:
            return "no sample between 'tip' and 'fit end'" if components else "no component ticked to fit"
        try:
            objective = self.build_joint_objective(t, t[mask], mask, t0, t1, t2)
            value = objective([self.Gc.get(), self.Xc.get()])
        except (tk.TclError, ValueError, FloatingPointError):
            return self.auto_note
        if not np.isfinite(value):
            return "no model at these parameters"
        rms = np.sqrt(value / objective.n_values) * 1e-3                           # kPa -> MPa
        text = (f"fit window {1e6 * (t2 - t1):.0f} us, {', '.join(components)}:  misfit (rms) {rms:.3f} MPa"
                f"   |   tau_p = {self.fit_context().get('tau_p', float('nan')) / 1e6:.2f} MPa")
        without = objective([1e-12, self.Xc.get()])                                # the same with no model at all
        if np.isfinite(without) and value >= 0.9 * without:
            text += "   |   no better than no model: these components do not follow it"
        elif any(not 1.01 * low < v < 0.99 * high for v, (low, high) in zip((self.Gc.get(), self.Xc.get()), STRESS_BOUNDS)):
            text += "   |   Gc or Xc is at the limit of the search: these data do not constrain it"
        return f"{text}   |   {self.auto_note}" if self.auto_note else text


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
        if self.stress_mode:
            # Gc and Xc span orders of magnitude between samples: search their logarithms
            initial_guess = [max(v, 1e-6) for v in initial_guess]
            if not np.isfinite(objective(initial_guess)):
                print("No fit: the model has no value at these parameters (Cf must be below the Rayleigh speed)")
                return None
            start = objective(initial_guess)
            result = optimize.minimize(lambda u: objective(np.exp(u)), np.log(initial_guess), method='Nelder-Mead',
                                       bounds=[tuple(np.log(b)) for b in STRESS_BOUNDS],
                                       options={"xatol": 1e-4, "fatol": max(1e-7 * start, 1e-9), "maxiter": 2000})
            result.x = np.exp(result.x)
            if not result.success and np.isfinite(result.fun) and result.fun <= start:
                result.success = True                      # out of iterations, yet no worse than the start: keep it
        else:
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

    # ----------------------------------------------------------------- auto
    def front(self, channel) -> dict | None:
        """The rupture front on one channel of a stress record, from the trace
        smoothed over 50 us: ``change`` (largest change over 50 us, signed, Pa),
        ``t_front`` (when), ``t_peak`` (the extreme before it, within 0.3 ms),
        ``t_end`` (when 90 % of the way from that peak to the opposite extreme
        of the next 0.5 ms is covered), ``size`` (the level at the peak minus
        that opposite extreme, Pa) and ``noise`` (the robust scatter of the
        50 us changes over the whole record, Pa: pure noise reaches about 4.5
        times it somewhere).  None for a record too short to look at."""
        t = self._time()
        if t.size < 50:
            return None
        fs = 1.0 / float(np.median(np.diff(t)))
        lag = int(np.clip(round(50e-6 * fs), 3, t.size // 4))
        window = min(lag | 1, t.size - 1 - (t.size % 2 == 0))
        raw = np.asarray(waveform_data(self.strain_block(self.event))[channel], dtype=float) * self.stress_scale
        smooth = signal.savgol_filter(raw, max(window, 5), 2)
        step = smooth[lag:] - smooth[:-lag]
        i = int(np.argmax(np.abs(step)))
        sign = 1.0 if step[i] >= 0 else -1.0                # +1: the trace rises at the front; -1: it drops
        before = max(i - int(round(300e-6 * fs)), 0)
        i_peak = before + int(np.argmin(sign * smooth[before: i + lag // 2 + 1]))
        after = min(i + int(round(500e-6 * fs)), t.size)
        travelled = sign * (smooth[i_peak: after] - smooth[i_peak])        # how far the trace has moved from the peak
        size = float(travelled.max())
        i_end = i_peak + int(np.argmax(travelled >= 0.9 * size))
        noise = 1.4826 * float(np.median(np.abs(step - np.median(step))))  # robust scatter of the 50 us changes
        return {"change": float(step[i]), "t_front": float(t[min(i + lag // 2, t.size - 1)]),
                "t_peak": float(t[i_peak]), "t_end": float(t[i_end]), "size": size, "noise": noise}

    def estimate_rupture_speed(self) -> float | None:
        """Rupture speed (m/s, positive) for ``Auto``: the one Pick Arrivals
        saved with the record, else the median slope between the fronts of the
        record's sensors (their sigma_xy) and their positions; None when the
        record has fewer than two sensor positions."""
        block = self.strain_block(self.event)
        try:
            return float(abs(read_arrivals(block, self.event)["rupture_speed"]))
        except (KeyError, TypeError, ValueError):
            pass
        positions = block.get("positions") if isinstance(block, dict) else None
        sensors = self.sensors()
        if not isinstance(positions, dict) or len(sensors) < 2:
            return None
        scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}.get(str(positions.get("unit", "mm")), 1e-3)
        points = []
        for channel in sensors.values():
            found = self.front(channel)
            try:
                x = float(np.asarray(positions["x"], dtype=float).ravel()[channel]) * scale
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if found is not None and np.isfinite(x):
                points.append((x, found["t_front"]))
        slopes = [(tb - ta) / (xb - xa) for k, (xa, ta) in enumerate(points) for xb, tb in points[k + 1:] if xb != xa]
        slope = float(np.median(slopes)) if slopes else 0.0
        return 1.0 / abs(slope) if slope else None

    def auto_fit(self):
        """A starting point for a stress record, then the fit.

        ``Cf`` from :meth:`estimate_rupture_speed`, kept below ``CF_LIMIT`` of the
        Rayleigh speed (the value in the box when there is nothing to estimate
        from); the tip line on the selected sensor's sigma_xy extreme just
        before its front, x_max where the change ends, the left line (end of
        the background window) four drop durations before the tip; Xc as the
        distance the front covers during the drop and Gc as the value whose
        peak strength equals the size of the drop.  Returns what
        :meth:`fit_parameters` returns, or None when no front is found.
        """
        if not self.stress_mode:
            return None
        self.sync_gauge()
        found = self.front(self.sensor_channels()["xy"])
        name = self.selected_sensor() or self.channel_name(self.sensor_channels()["xy"]) or "this channel"
        if found is None or found["size"] <= 0 or abs(found["change"]) < CLEAR_FRONT * found["noise"]:
            what = ("the record is too short" if found is None else
                    f"its largest change over 50 us is {abs(found['change']) / 1e6:.3f} MPa against a scatter of "
                    f"{found['noise'] / 1e6:.3f} MPa")
            self.auto_note = f"Auto: no clear front on {name} ({what}); nothing changed"
            print(self.auto_note)
            self.result_var.set(self.result_text())
            return None
        t = self._time()
        c_r = rayleigh_speed(self.C_s, self.C_d)
        limit = CF_LIMIT * c_r
        speed = self.estimate_rupture_speed()
        Cf = min(speed, limit) if speed else min(self.Cf.get(), limit)
        if speed and speed > limit:
            self.auto_note = (f"Auto: the fronts give {speed:.0f} m/s, not below C_R = {c_r:.0f} m/s, where the model "
                              f"does not exist: Cf kept at {CF_LIMIT:g} C_R")
        elif speed:
            self.auto_note = f"Auto: Cf = {speed:.0f} m/s from the fronts"
        else:
            self.auto_note = "Auto: no rupture speed to take; Cf left as it was"
        self.Cf.set(float(Cf))
        tip, end = found["t_peak"], found["t_end"]
        if end - tip < 5 * float(np.median(np.diff(t))):          # no measurable drop time: 0.1 ms
            end = min(tip + 1e-4, float(t[-1]))
        drop = end - tip
        left = max(tip - 4 * drop, float(t[0]) + 0.1 * (tip - float(t[0])))
        self.x_base = None                                        # the background window: record start .. left line
        self.set_lines([left, tip, end])
        Xc = max(Cf * drop, 1e-6)
        with np.errstate(all="ignore"):
            alpha_s, alpha_d = CohesiveCrack.alpha_s(Cf, self.C_s), CohesiveCrack.alpha_d(Cf, self.C_d)
            A2 = CohesiveCrack.compute_A2(Cf, self.C_s, self.nu, CohesiveCrack.D(alpha_s, alpha_d))
            Gc = found["size"] ** 2 * (32 * Xc / (9 * np.pi)) * (1 - self.nu ** 2) * A2 / self.E   # tau_p = size
        self.Xc.set(float(Xc))
        self.Gc.set(float(Gc) if np.isfinite(Gc) and Gc > 0 else 1.0)
        margin = 0.5 * (end - left)
        self.axs[0].set_xlim(max(left - margin, float(t[0])), min(end + 2 * margin, float(t[-1])))
        self.update_plot()
        print(f"Auto: Cf={Cf:.0f} m/s, tip at {tip:.6f} s, drop over {drop * 1e6:.0f} us of {found['size'] / 1e6:.3f} MPa; "
              f"start Xc={self.Xc.get():.3g} m, Gc={self.Gc.get():.3g} J/m^2")
        start = (self.Gc.get(), self.Xc.get())
        result = self.fit_parameters()
        if result is not None and set(self.fit_components()) & {"xx", "yy"}:
            # sigma_xx and sigma_yy change sign with the side of the fault: try the other one, keep the better
            first = (result, self.y.get(), self.Gc.get(), self.Xc.get())
            self.y.set(-first[1]), self.Gc.set(start[0]), self.Xc.set(start[1])
            other = self.fit_parameters()
            if other is None or not other.fun < first[0].fun:
                self.y.set(first[1]), self.Gc.set(first[2]), self.Xc.set(first[3])
                self.update_plot()
                result = first[0]
            else:
                result = other
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
        the elastic medium, the components fitted together and the start of
        the background window ``x_base`` (stress records), and the peak
        strength ``tau_p`` (Pa) that Gc and Xc imply at this Cf."""
        gauge = int(self.strain_gauge.get())
        if self.stress_mode:
            gauge = self.sensor_channels(gauge)["xy"]
        context = {"channel": self.channel_name(gauge), "unit": self.record_unit(self.strain_block(self.event)),
                   "E": float(self.E), "nu": float(self.nu), "C_s": float(self.C_s), "C_d": float(self.C_d)}
        if self.selected_sensor():
            context["sensor"] = self.selected_sensor()
        if self.stress_mode:
            context["x_base"] = self.baseline_window(self._time(), self.current_line_positions())[0]
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
