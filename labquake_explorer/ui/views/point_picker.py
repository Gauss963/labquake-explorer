"""Draggable sample markers on a matplotlib axes.

``PointPicker`` owns a list of picked sample indices and one circular marker
per pick.  A marker is dragged along the curve (it snaps to the nearest
finite sample); with ``add_remove`` a left double-click on the curve adds a
pick and a right double-click on a marker removes it.  The view owns the
axes: after (re)plotting it calls ``set_curve`` and ``draw_markers`` to put
the markers back, and receives ``on_change(kind, idx)`` once per completed
edit (``kind`` is ``"add"``, ``"remove"`` or ``"move"``; ``idx`` the sample
added, removed or moved to); ``on_drag(position, idx)`` while a marker is
dragged, after each step to another sample.  ``color`` is one colour or one
per marker position.  ``on_release_hook`` runs after every button
release, so a view that must not rebuild the axes while a marker is held can
postpone that work until the mouse is up.  Markers are sized in data units
and follow every change of the axes limits (toolbar zoom, pan, home).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import matplotlib.patches as patches
import numpy as np
from matplotlib.backend_bases import MouseButton

from labquake_explorer.ui.views.base import nearest_sample

ChangeCallback = Callable[[str, int], None]


class PointPicker:
    def __init__(self, ax, canvas, x, y, picks: Sequence[int] = (), add_remove: bool = True,
                 toolbar_active: Callable[[], bool] = lambda: False,
                 on_change: Optional[ChangeCallback] = None,
                 on_release_hook: Optional[Callable[[], None]] = None, color="red",
                 on_drag: Optional[ChangeCallback] = None):
        self.ax = ax
        self.canvas = canvas
        self.figure = ax.figure
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.picks: list[int] = [int(i) for i in picks]
        self.add_remove = add_remove
        self.toolbar_active = toolbar_active
        self.on_change = on_change
        self.on_release_hook = on_release_hook
        self.on_drag = on_drag
        self.color = color
        self._lim_cids: list = []
        self.markers: list[patches.Ellipse] = []
        self.offset = [0.0, 0.0]
        self.current_artist = None
        self.currently_dragging = False
        self.moved_to: Optional[int] = None
        self._cids = [
            canvas.mpl_connect("pick_event", self.on_pick),
            canvas.mpl_connect("motion_notify_event", self.on_motion),
            canvas.mpl_connect("button_press_event", self.on_press),
            canvas.mpl_connect("button_release_event", self.on_release),
            canvas.mpl_connect("resize_event", self.on_resize),
            canvas.mpl_connect("scroll_event", self.on_resize),
        ]

    def disconnect(self) -> None:
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []

    # ------------------------------------------------------------- markers
    def set_curve(self, x, y) -> None:
        """Point the markers at a new curve; the picks (sample indices) stay."""
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def _make_marker(self, idx: int, label: int, width: float, height: float) -> patches.Ellipse:
        color = self.color if isinstance(self.color, str) else self.color[label % len(self.color)]
        marker = patches.Ellipse((self.x[idx], self.y[idx]), width=width, height=height,
                                 color=color, fill=False, lw=2, picker=8, label=str(label))
        self.ax.add_patch(marker)
        self.markers.append(marker)
        return marker

    def draw_markers(self) -> None:
        """Recreate every marker (after the axes were cleared or the curve changed)."""
        for marker in self.markers:
            if marker in self.ax.patches:
                marker.remove()
        self.markers = []
        width, height = self.marker_size()
        n = len(self.x)
        for i, idx in enumerate(self.picks):
            if 0 <= idx < n:
                self._make_marker(idx, i, width, height)
        self._watch_limits()
        self.canvas.draw_idle()

    def _watch_limits(self) -> None:
        """Resize the markers whenever the axes limits change.  ``ax.clear()``
        replaces the axes' callback registry, so this is redone per draw."""
        for cid in self._lim_cids:
            self.ax.callbacks.disconnect(cid)
        self._lim_cids = [self.ax.callbacks.connect(name, lambda ax: self.update_marker_size())
                          for name in ("xlim_changed", "ylim_changed")]

    def renumber_markers(self) -> None:
        """Keep marker label == position in ``picks``."""
        for i, marker in enumerate(self.markers):
            marker.set_label(str(i))

    def add_point(self, idx: int) -> int:
        """Append sample ``idx`` as a new pick; returns its position in ``picks``."""
        idx = int(idx)
        width, height = self.marker_size()
        self._make_marker(idx, len(self.picks), width, height)
        self.picks.append(idx)
        self.canvas.draw_idle()
        self._notify("add", idx)
        return len(self.picks) - 1

    def remove_point(self, i: int) -> None:
        """Remove the pick at position ``i`` and renumber the remaining markers."""
        marker = self.markers.pop(i)
        marker.remove()
        idx = self.picks.pop(i)
        if self.current_artist is marker:
            self.current_artist = None
        self.renumber_markers()
        self.canvas.draw_idle()
        self._notify("remove", idx)

    def sort(self) -> None:
        """Sort the picks by sample index; the markers follow."""
        order = sorted(range(len(self.picks)), key=self.picks.__getitem__)
        self.picks.sort()
        if len(self.markers) == len(order):
            self.markers = [self.markers[j] for j in order]
        self.renumber_markers()

    def nearest_index(self, cx: float, cy: float) -> int:
        """Finite sample nearest to (cx, cy) in axes-normalised distance.

        NaN samples are never candidates; with no finite sample at all the
        drag/add is refused (ValueError, caught by the mouse handlers).
        """
        idx = nearest_sample(self.x, self.y, cx, cy, self.ax.get_xlim(), self.ax.get_ylim())
        if idx is None:
            raise ValueError("no finite samples to snap to")
        return idx

    def marker_size(self) -> tuple[float, float]:
        """Ellipse width/height in data units that render as a circle."""
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        xw = (xl[-1] - xl[0]) or 1.0
        fig_w, fig_h = self.figure.get_size_inches()
        ratio = (yl[-1] - yl[0]) / xw * fig_w / (fig_h or 1.0)
        width = xw / fig_w * 0.15
        return width, width * ratio

    def update_marker_size(self) -> None:
        if not self.markers:
            return
        width, height = self.marker_size()
        for marker in self.markers:
            marker.set_width(width)
            marker.set_height(height)
        self.canvas.draw_idle()

    def _notify(self, kind: str, idx: int) -> None:
        if self.on_change is not None:
            self.on_change(kind, idx)

    # ------------------------------------------------------------- mouse
    def on_pick(self, event):
        if self.toolbar_active():
            return
        if self.current_artist is not None or event.artist not in self.markers:
            return
        self.current_artist = event.artist
        mouse = event.mouseevent
        if mouse.dblclick:
            if self.add_remove and mouse.button != MouseButton.LEFT:
                self.remove_point(int(self.current_artist.get_label()))
                self.current_artist = None
        else:
            x0, y0 = self.current_artist.center
            self.offset = [x0 - mouse.xdata, y0 - mouse.ydata]

    def on_motion(self, event):
        if not self.currently_dragging or self.current_artist is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.current_artist not in self.markers:
            return
        try:
            dx, dy = self.offset
            idx = self.nearest_index(event.xdata + dx, event.ydata + dy)
            position = int(self.current_artist.get_label())
            if self.picks[position] != idx:
                self.current_artist.set_center((self.x[idx], self.y[idx]))
                self.picks[position] = idx
                self.moved_to = idx
                if self.on_drag is not None:
                    self.on_drag(position, idx)
                self.canvas.draw_idle()
        except Exception as e:
            print(f"Error in on_motion: {e}")

    def on_press(self, event):
        self.currently_dragging = True
        if self.toolbar_active():
            return
        # The figure dispatches pick events before this handler, so a double
        # click on an existing marker has already set current_artist.
        if (event.button == MouseButton.LEFT and event.dblclick and self.add_remove
                and self.current_artist is None and event.xdata is not None and event.ydata is not None):
            try:
                self.add_point(self.nearest_index(event.xdata, event.ydata))
            except ValueError as e:
                print(f"Cannot add a point: {e}")

    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.update_marker_size()
        if self.moved_to is not None:
            idx, self.moved_to = self.moved_to, None
            self._notify("move", idx)
        if self.on_release_hook is not None:
            self.on_release_hook()

    def on_resize(self, event):
        self.update_marker_size()
