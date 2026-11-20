"""Base classes for view windows.

``BaseView`` handles what every Toplevel in this application needs: a
reference to the application, its DataManager, the window icon, child-window
bookkeeping, and a figure/toolbar helper.  ``EventView`` adds the event
combobox, event loading, and load/save of the view's own result dict stored
under ``runs/[r]/events/[e]/<result_key>``.  ``RunView`` does the same for
run-scoped results under ``runs/[r]/<result_key>``.

Subclasses implement ``build_ui()`` to create their widgets and
``on_event_loaded()`` / ``on_run_loaded()`` to refresh from data.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.ui.context import TreeContext


def event_list(run) -> list:
    """The ``events`` container of a run dict as a list (``[]`` when absent).

    Never truth-tests the container: an empty ``events`` list comes back
    from an HDF5 file as an empty ``ndarray``, whose ``bool()`` raises.
    Only list/tuple containers are events; anything else (an empty array,
    a scalar, None) means "no events".
    """
    events = run.get("events") if isinstance(run, dict) else None
    return list(events) if isinstance(events, (list, tuple)) else []


def nearest_sample(x, y, cx: float, cy: float, xlim, ylim) -> Optional[int]:
    """Index of the finite sample of ``(x, y)`` nearest to ``(cx, cy)``.

    Distances are measured in axes-normalised units (``xlim``/``ylim`` are
    the axis ranges).  Samples with a non-finite coordinate are never
    candidates (a plain ``argmin`` returns the first NaN); ``None`` when no
    finite sample exists.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xw = float(xlim[-1] - xlim[0]) or 1.0
    yw = float(ylim[-1] - ylim[0]) or 1.0
    with np.errstate(invalid="ignore"):
        d = ((x - cx) / xw) ** 2 + ((y - cy) / yw) ** 2
    finite = np.isfinite(d)
    if not finite.any():
        return None
    d = np.where(finite, d, np.inf)
    return int(np.argmin(d))


class BaseView(tk.Toplevel):
    """A Toplevel owned by the main window."""

    window_title: str = "View"

    def __init__(self, app, title: Optional[str] = None):
        self.app = app
        self.parent = app  # legacy alias used by older views
        super().__init__(app.root)
        self.data_manager = app.data_manager
        self.title(title or self.window_title)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        app.register_child(self)

    @classmethod
    def from_context(cls, app, ctx: TreeContext):
        return cls(app)

    def make_figure(self, master=None, figsize=(10, 6), dpi=100, toolbar=True,
                    figure_kwargs=None, **grid):
        """Create a Figure + Tk canvas (+ toolbar) and grid them into ``master``.

        ``grid`` keywords are passed to ``canvas_widget.grid``; the toolbar is
        placed in the next row with the same column span.  ``figure_kwargs``
        (e.g. ``{"layout": "constrained"}``) go to ``Figure``.  Without grid
        keywords only the figure/canvas are created and the caller places
        ``self.canvas_widget`` (and a toolbar) itself.  Returns the figure.
        """
        master = master or self
        self.figure = Figure(figsize=figsize, dpi=dpi, **(figure_kwargs or {}))
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        if grid:
            self.canvas_widget.grid(**grid)
            if toolbar:
                row = grid.get("row", 0) + 1
                toolbar_frame = ttk.Frame(master)
                toolbar_frame.grid(row=row, column=grid.get("column", 0),
                                   columnspan=grid.get("columnspan", 1), sticky="ew")
                self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
                self.toolbar.update()
        return self.figure

    def toolbar_active(self) -> bool:
        """True while the matplotlib toolbar is in pan or zoom mode."""
        toolbar = getattr(self, "toolbar", None)
        return bool(toolbar is not None and toolbar.mode)

    def on_close(self) -> None:
        self.app.unregister_child(self)
        self.destroy()

    def has_unsaved_edits(self) -> bool:
        """True while the view holds edits its Save has not stored (views that
        track them override this)."""
        return False

    def confirm_leave(self, what: str = "go on") -> bool:
        """Before a user's click that would drop unsaved edits: True to go on.
        Only widget callbacks ask; the methods a script calls never do."""
        if not self.has_unsaved_edits():
            return True
        return bool(messagebox.askyesno("Unsaved edits", f"The edits shown here are not saved.\n\n"
                                        f"Discard them and {what}?", icon=messagebox.WARNING, parent=self))


class EventView(BaseView):
    """A view of one event, with an event selector and a result namespace.

    Construction order: the subclass sets plain attributes, then calls
    ``super().__init__``, which loads ``self.event``, calls ``build_ui()``
    (create widgets and Tk variables here; the Toplevel exists by now), sets
    the title, and calls ``on_event_loaded()``.  If any of those raise, the
    half-built window is unregistered and destroyed before the exception
    propagates.
    """

    result_key: Optional[str] = None

    def __init__(self, app, run_idx: int, event_idx: int, title: Optional[str] = None):
        self.run_idx = int(run_idx)
        self.event_idx = int(event_idx)
        self.event: dict = {}
        self.run: Optional[dict] = None
        super().__init__(app, title=title)
        try:
            self.event = self.data_manager.get_data(self.event_path)
            self.event_combobox: Optional[ttk.Combobox] = None
            self.build_ui()
            self.update_title()
            self.on_event_loaded()
        except Exception:
            self.app.unregister_child(self)
            try:
                self.destroy()
            except tk.TclError:
                pass
            raise

    @classmethod
    def from_context(cls, app, ctx: TreeContext):
        if ctx.run_idx is None or ctx.event_idx is None:
            raise ValueError(f"{cls.__name__} needs an event context, got {ctx.path!r}")
        return cls(app, ctx.run_idx, ctx.event_idx)

    # ------------------------------------------------------------------ paths
    @property
    def run_path(self) -> str:
        return f"runs/[{self.run_idx}]"

    @property
    def event_path(self) -> str:
        return f"{self.run_path}/events/[{self.event_idx}]"

    @property
    def result_path(self) -> str:
        if not self.result_key:
            raise ValueError(f"{type(self).__name__} has no result_key")
        return f"{self.event_path}/{self.result_key}"

    def n_events(self) -> int:
        return len(self.data_manager.get_data(f"{self.run_path}/events"))

    def get_run(self) -> dict:
        if self.run is None:
            self.run = self.data_manager.get_data(self.run_path)
        return self.run

    def experiment_name(self) -> str:
        try:
            return str(self.data_manager.get_data("name"))
        except Exception:
            return ""

    def figure_title(self) -> str:
        return f"{self.experiment_name()} run{self.run_idx:02d} event{self.event_idx}"

    def update_title(self) -> None:
        self.title(f"{self.window_title} - Event {self.event_idx}")

    # ---------------------------------------------------------------- widgets
    def build_ui(self) -> None:
        """Create widgets. Subclasses override; call ``build_event_selector``."""

    def build_event_selector(self, master, **grid) -> ttk.Combobox:
        """Create the 'Event Index' combobox and bind it to ``set_event``."""
        ttk.Label(master, text="Event Index:").grid(row=grid.get("row", 0), column=grid.get("column", 0),
                                                    padx=5, pady=5, sticky="w")
        self.event_combobox = ttk.Combobox(master, width=8, state="readonly")
        self.event_combobox.grid(row=grid.get("row", 0), column=grid.get("column", 0) + 1,
                                 padx=5, pady=5, sticky="w")
        self.refresh_event_selector()
        self.event_combobox.bind("<<ComboboxSelected>>", self._event_chosen)
        return self.event_combobox

    def _event_chosen(self, event=None) -> None:
        if self.event_combobox.get() != str(self.event_idx) and not self.confirm_leave("open another event"):
            self.refresh_event_selector()
            return
        self.on_event_selected(event)

    def step_event_asked(self, step: int) -> None:
        """The arrow buttons: ``step_event`` once unsaved edits may go."""
        if self.confirm_leave("open another event"):
            self.step_event(step)

    def refresh_event_selector(self) -> None:
        if self.event_combobox is None:
            return
        n = self.n_events()
        self.event_combobox.config(values=[str(i) for i in range(n)])
        if 0 <= self.event_idx < n:
            self.event_combobox.current(self.event_idx)

    def on_event_selected(self, event=None) -> None:
        new_idx = int(self.event_combobox.get())
        if new_idx != self.event_idx:
            self.set_event(new_idx)

    def set_event(self, event_idx: int) -> None:
        self.event_idx = int(event_idx)
        self.event = self.data_manager.get_data(self.event_path)
        self.update_title()
        if self.event_combobox is not None and self.event_combobox.get() != str(self.event_idx):
            self.event_combobox.set(str(self.event_idx))
        self.on_event_loaded()

    def on_event_loaded(self) -> None:
        """Refresh the view from ``self.event``. Subclasses override."""

    # ---------------------------------------------------------------- results
    def load_results(self) -> Optional[dict]:
        """The saved result dict for this event, or None."""
        if not self.result_key:
            return None
        value = self.event.get(self.result_key)
        return value if isinstance(value, dict) else None

    def save_results(self, results: dict) -> None:
        """Persist ``results`` under ``event/<result_key>`` and refresh the tree."""
        self.data_manager.set_data(self.result_path, results, True)
        self.event[self.result_key] = results
        self.app.refresh_tree()

    def save_field(self, relative_path: str, value, refresh: bool = True) -> None:
        """Persist one value under ``event/<relative_path>`` (for views without a single result key)."""
        self.data_manager.set_data(f"{self.event_path}/{relative_path}", value, True)
        if refresh:
            self.app.refresh_tree()


class RunView(BaseView):
    """A view of one run, with a result namespace under ``runs/[r]/<result_key>``."""

    result_key: Optional[str] = None

    def __init__(self, app, run_idx: int, title: Optional[str] = None):
        self.run_idx = int(run_idx)
        super().__init__(app, title=title)
        try:
            self.run = self.data_manager.get_data(self.run_path)
            self.build_ui()
            self.title(f"{title or self.window_title} - run{self.run_idx:02d}")
            self.on_run_loaded()
        except Exception:
            self.app.unregister_child(self)
            try:
                self.destroy()
            except tk.TclError:
                pass
            raise

    @classmethod
    def from_context(cls, app, ctx: TreeContext):
        if ctx.run_idx is None:
            raise ValueError(f"{cls.__name__} needs a run context, got {ctx.path!r}")
        return cls(app, ctx.run_idx)

    @property
    def run_path(self) -> str:
        return f"runs/[{self.run_idx}]"

    @property
    def result_path(self) -> str:
        if not self.result_key:
            raise ValueError(f"{type(self).__name__} has no result_key")
        return f"{self.run_path}/{self.result_key}"

    def experiment_name(self) -> str:
        try:
            return str(self.data_manager.get_data("name"))
        except Exception:
            return ""

    def build_ui(self) -> None:
        """Create widgets. Subclasses override."""

    def on_run_loaded(self) -> None:
        """Refresh the view from ``self.run``. Subclasses override."""

    def load_results(self) -> Optional[dict]:
        if not self.result_key:
            return None
        value = self.run.get(self.result_key)
        return value if isinstance(value, dict) else None

    def save_results(self, results: dict) -> None:
        self.data_manager.set_data(self.result_path, results, True)
        self.run[self.result_key] = results
        self.app.refresh_tree()
