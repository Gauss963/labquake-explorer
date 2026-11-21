# Extending Labquake Explorer

## Adding a view

1. Create `labquake_explorer/ui/views/<name>_view.py` with a class deriving from
   `EventView` (one event), `RunView` (one run) or `BaseView` (anything else)
   from `labquake_explorer/ui/views/base.py`.
2. Decorate it with `@register_view(label, kinds=[...])` from
   `labquake_explorer/ui/actions.py`. `kinds` are tree-node kinds from
   `labquake_explorer/ui/context.py` (`EVENT`, `RUN`, `RUN_ARRAY`,
   `EVENT_ARRAY`, `EVENT_EXTRACTION`, `ARRAY`, `STRING`). The main window builds
   the right-click menu for a node from the registry; nothing else to edit.
3. Import the module in `labquake_explorer/ui/views/__init__.py` so it is
   registered at start-up.
4. Put numerics in `labquake_explorer/analysis/` as pure functions (arrays in,
   dataclass or JSON-like dict out; never raise on degenerate data, return a
   `valid` flag and a reason) and test them with synthetic signals in `tests/`
   (`tests/synthetic.py` builds stick-slip runs with known answers).

```python
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

@register_view("My Analysis", kinds=[EVENT], order=50)
class MyView(EventView):
    window_title = "My Analysis"
    result_key = "my_analysis"          # saved under runs/[r]/events/[e]/my_analysis

    def build_ui(self):                 # widgets; call self.build_event_selector(frame, row=0, column=0)
        ...
        self.make_figure(figsize=(8, 5), row=2, column=0, sticky="nsew")

    def on_event_loaded(self):          # refresh from self.event; self.load_results() gives the saved dict
        ...

    def save(self):
        self.save_results({"version": 1, ...})   # writes the dict and refreshes the tree
```

`EventView` provides `run_idx`, `event_idx`, `event`, `event_path`, `set_event()`,
the event combobox, `load_results()` / `save_results()`, `make_figure()`,
`toolbar_active()` (ignore drags while pan/zoom is on) and `on_close()`.
`RunView` provides `run_idx`, `run`, `run_path`, `load_results()` /
`save_results()` under `runs/[r]/<result_key>`, `make_figure()`,
`toolbar_active()`, `on_close()` and the hook `on_run_loaded()` (called once
the run is loaded; there is no event selector and no `on_event_loaded()`).

A tall column of controls goes into a `ScrollPanel`
(`labquake_explorer/ui/views/scroll_panel.py`): grid the panel with
`sticky="ns"`, build the controls in `panel.inner`, then call
`panel.bind_children()`. The panel opens at its full height (at most the
screen height less 160 px), follows the window's height, shows a scrollbar
only when the controls do not fit, and takes the mouse wheel over its
widgets, comboboxes included. The Slip Distribution and PZT Spectrum views
use one for each of their two control columns.

A menu command that is not a window registers a function instead:

```python
@register_action("Do Something", kinds=[RUN_ARRAY])
def do_something(app, ctx): ...       # ctx.path, ctx.run_idx, ctx.event_idx, ctx.key
```

## Data rule

Raw extracted data lives at the top level of an event. Each analysis view
(`event_analysis`, `slip_analysis`, `interevent`, `spectrum`, `source_scaling`) writes only
under its own key, and every saved dict carries a `version` integer, the
inputs needed to reproduce the numbers, and outputs with units in the key
names. Cross-view reads go through the other view's key.

Two older views predate the rule and are kept as they are so existing files
stay readable: Pick Arrivals writes into the event's raw layout (see its
schema below) and `czm_parms` carries no `version`.

## Run layout

A run built by the preprocessing package (`labquake_explorer/preprocessing/`)
looks like this:

| key | content |
|---|---|
| `time` | 1-D, seconds on the run clock |
| `elsys`, `ni`, ... | one block per recorder: its raw-file reference (next section) plus `raw_data`, the channel array of the voltages it recorded: `data` (channels x samples, float32), `channels` (`pzt_1`, `pressure_1`, `eddy_3`, ...), `unit` (`V`), `positions` |
| `normal_stress`, `shear_stress`, `friction`, `LP_displacement` | 1-D physical channels (MPa, um, ...). PSU-era files also have `displacement` (fault slip); without it the views take the first `slip` channel |
| `slip` | channel array of the eddy-current sensors in um: `data`, `channels` (`slip_1`...), `source` (the voltage channel of each row), `slope_mm_per_v`, `positions` |
| `units` | unit of every top-level array and channel array |
| `calibration` | the steps that produced the physical channels (see the preprocessing package) |
| `event_extraction`, `events` | the Pick Events form's record, with the picks (`event_indices`) inside (schema below), and the extracted events |
| `slip_analysis` | the events chosen for the Slip Distribution analysis: `version` (1) and `events` (event indices); written when a list is typed in the view's Auto windows box, used when it is left blank. Without it every event is used |

Everything a view stores for the whole experiment goes into one dict next to
`runs`, `analysis` (`labquake_explorer/data/settings.py`: `get_setting`,
`set_setting`), so the top level keeps to the metadata the preprocessing
wrote (`name`, `date`, `sample`, a `fault` description, ...):

| key | meaning |
|---|---|
| `analysis['fault']` | `patch_x_mm = [from, to]`, the edges of the bare PMMA patch along the fault in mm (in either order), and `ends_x_mm = [from, to]`, where the fault (the sample) ends, both in the frame of the sensor positions. The Slip Distribution view draws them, cuts the slip profile at the ends and can save both; a notebook sets them with `experiment(..., analysis={'fault': {'patch_x_mm': [x0, x1], 'ends_x_mm': [0, 500]}})`. A value without two finite, different numbers is not drawn, and the view says it is unusable |
| `analysis['medium']` | `rho_kg_m3`, `vp_m_s`, `vs_m_s` of the sample, as last saved by the PZT Spectrum view and offered to every new window |
| `analysis['pzt_spectrum']` | per result set (see the `spectrum` schema), `sets[<name>]['settings']` (the PZT Spectrum view's controls as last saved in that set: the keys of a `spectrum` record below plus `phase` and `radiation_coefficient`) and `sets[<name>]['sensors']` (per sensor, keyed by the channel's field name such as `pzt_1`, else `ch<k>`: the `calibration` table, `calibration_path`, `calibration_unit`, `calibration_db`, `calibration_divisor`, `t_star_s`, `distance_m`, `free_surface_factor`), and `current_set`, the set last saved, where a new window opens. Files written before result sets (2026-11) keep flat `settings` / `sensors`: they are read as the set `default` and moved on the next save or by the migration tool |
| `analysis['pick_arrivals']` | `sets[<name>]['records'][<recorder>]`: what the Pick Arrivals view last saved in that pick set on that kind of record (`elsys`, `ni`, `strain`): `filter`, `auto_pick` (`method`, `threshold`, `smoothing_us`, and the search window's `search_from_s` / `search_to_s`), `front_mode`, and the channels `shown` and `fit` by name. An event without picks in the set starts from it; choosing the set on such an event takes these controls and keeps the picks on screen. `current_set`: the set saved last, where a new window opens (as `analysis['pzt_spectrum']['current_set']`) |

Files saved earlier kept the patch edges and fault ends as top-level
`patch_x_mm` / `fault_ends_x_mm` or as `fault = {'patch_x_mm': ...,
'ends_x_mm': ...}`; the view reads those and moves a value into `analysis`
when it saves.

Next to it, `labquake_explorer` says which package wrote the file
(`labquake_explorer/data/stamp.py`):

| key | meaning |
|---|---|
| `version`, `saved_at` | the package version that last wrote the file and the local time it did (ISO 8601 with the UTC offset). Every write refreshes them: `DataManager.save_file` and the migration tool's in-place edits |
| `git` | `git describe` of the checkout the package ran from (`v0.2.0-3-g1bf4fca`, `-dirty` when it had uncommitted edits), so a file written between releases still names its commit; absent when the package did not run from a checkout |
| `created_version`, `created_at` | the package that built the experiment (`preprocessing.build.experiment`); later saves leave them |

A file made before the stamp existed gets one when it is next saved. The
version itself is written once, in `labquake_explorer/_version.py` (`setup.py`
reads it; the window title and `labquake-explorer --version` show it); bump
it for a release or a layout change and tag that commit `v<version>`.

A **channel array** (`labquake_explorer/data/channels.py`) is a dict with
`data` shaped `(n_channels, n)`, `channels` naming the rows, `unit`, and an
optional `positions` table (`x`, `y`, `z` per channel, NaN when unknown, plus
`unit` and `frame`). Any further key is metadata. Views address a row by its
path (`slip/slip_3`, `elsys/raw_data/pzt_1`); `aligned_fields(container, n)`
lists every 1-D series of length `n` this way and `get_field(container,
path)` resolves one, so a field is the same whether it is a top-level array
or a row. Event extraction slices channel arrays along the sample axis and
keeps their metadata: `slip` is sliced, the recorders' blocks (file reference
and `raw_data` volts) are not copied, and each recorder's full-rate record
becomes the channel array `waveform/<recorder>` (with `time`, `sample_rate`,
`filename` and, for a recorder that writes separate records, `block` and
`trigger_time`, where the recorder fired, on the run clock). Older files with
top-level 1-D channels keep working: a channel array is only an additional
place a field can live.

## Saved result schemas

### `event_extraction` (EventPickerView, run level, version 1)

What the Pick Events form last saved, and what it starts from when reopened:
`event_indices` (the picks, sample indices on the run's time axis), `start_s` and
`end_s` (the window around each pick, seconds, start negative), `x_field` and
`y_field` (the series shown), `n_events` (how many events the last Extract
wrote) and `waveform_fields`, the "Waveform channels" choice: `reference`
(each recorder's `event_fields`, as before), `all` (every channel of the
recorder, the eddy probes at the full rate too) or a list of names
(`EventProcessor.extract_events(..., fields=)`). Save picks updates
`event_indices` only; Extract writes the whole record. Older files that kept `event_indices` at the run's top level (and the
window under `event_window`) are read; saving moves both into the record.
`labquake_explorer/data/picks.py` gives readers `picked_indices(run)` for
either layout.

### `event_analysis` (EventAnalyzerView, version 3)

Six picked samples on the event slice and what they give on the chosen
fields, nothing more. The numbers are slopes and differences of whatever
`x_field` and `y_field` were: with fault slip on X and shear stress on Y the
slopes are stiffnesses, `delta_x` the coseismic slip and `-delta_y` the
stress drop.

| key | meaning |
|---|---|
| `version` | 3 |
| `x_field`, `y_field` | the fields plotted |
| `loading_indices`, `unloading_indices` | inclusive index ranges of the two slopes |
| `delta_indices` | the two samples differenced (start, end) |
| `loading_slope`, `unloading_slope` | dY/dX by least squares over each range (NaN when degenerate) |
| `delta_x`, `delta_y` | X and Y at `delta_indices[1]` minus at `delta_indices[0]` |

Version 1 and 2 records (`rupture_start_index`/`rupture_end_index`,
`stress_drop`, `displacement`, trend fits) are still read for their picks;
the inter-event and scaling views take `displacement`/`stress_drop` from
them and `delta_x`/`-delta_y` from version 3. "Apply to All Events" turns
the current picks into times relative to the event, places them on every
event of the run and saves each result.

### `slip_analysis` (SlipDistributionView, version 2)

Coseismic slip of every sensor of a slip channel array (`field`, normally
`slip`), each sensor with its own event time `t_i`
(`labquake_explorer/analysis/coseismic_slip.py`). Top-level lists hold one
entry per sensor, in the order of `channels`, except `patch_x_mm`
(`[from, to]`); inside `profile` and `front` only `use` is per sensor.

| key | meaning |
|---|---|
| `version` | 2 (version 1 gave the fit windows relative to `t_i`; they are moved to the event time when read) |
| `field`, `unit`, `channels`, `x_mm` | the channel array analysed, its unit (um), its rows and the sensor positions along the fault. A `field` of the form `waveform/<recorder>` is the full-rate record's eddy channels (those named in the run's `slip.source`), converted with `slip.slope_mm_per_v` to micrometres on the record's own time axis (2 MHz for an Elsys block, extracted with "Waveform channels: all"), each zeroed on its first 1 % of samples, with the run's sensor positions; the record's PZT channels give the burst pick. The view offers it next to `slip` when the record holds those channels |
| `event_time`, `start_s`, `end_s` | the event time and the extraction window the default fit windows came from |
| `time_method` | per sensor: `max_rate` (largest filtered slip rate), `onset` (where the rate, followed back from that peak, last rose through `onset_fraction` of it, default 0.2), `step` (where the slip crosses `step_fraction`, default 0.5, of the way from the before to the after line, between the windows), `shared` (one t_i for all: `event_time + shared_shift_s`, or the reference time when that is NaN), `burst` (one t_i for all at the AE burst, below) or `manual` (set by hand); `onset` and `step` interpolate between samples, and keep the max-rate pick, flagged, when they find no crossing |
| `window_mode`, `burst_halfwidth_s`, `level_width_s`, `burst_search_s`, `burst_source`, `t_burst_shift_s`, `before_level_um`, `after_level_um` | the Burst tab (`window_mode = levels`): every sensor's t_i is the burst, the peak of the PZT envelope (`burst_source = pzt_envelope`, the record's PZT traces, |pzt| averaged over 20 us, within `t_ref +- burst_search_s`, default 3 ms; falls back to the peak of the mean slip rate, flagged `no PZT trace`) or that peak itself (`peak_rate`); the fit windows are `[t_i - w - d, t_i - w]` and `[t_i + w, t_i + w + d]` with `w = burst_halfwidth_s` (default 1 ms) and `d = level_width_s` (0.1 ms), and the coseismic slip is `after_level_um - before_level_um`, the mean slip in those windows (no line extrapolation); `t_burst_shift_s` is the burst relative to the event time. Records without these keys are `lines` mode, the slip from the lines at t_i |
| `onset_fraction`, `step_fraction`, `shared_shift_s`, `shared_set_s` | the settings of those methods; `shared_set_s` is the value typed in the view's Shared tab, kept while the reference is used |
| `t_shift_s` | `t_i - event_time` |
| `rate_filter`, `rate_cutoff_hz`, `rate_lowpass_type`, `rate_lowpass_order`, `rate_savgol_order`, `rate_smoothing_s`, `search_halfwidth_s`, `rate_snr`, `reference_halfwidth_s` | `max_rate` settings: the slip-rate filter (the view's filter tab shown), `lowpass` (the default: the slip through a `butterworth` (default) or `bessel` low-pass of order `rate_lowpass_order` (4) at `rate_cutoff_hz` (50), run forwards and backwards for zero phase, then the derivative), `savgol` (Savitzky-Golay derivative of polynomial order `rate_savgol_order` (2) over the window `rate_smoothing_s`; the view's default 0.02) or `boxcar` (the derivative averaged over `rate_smoothing_s`, default 0.01; records without `rate_filter` were made with it); `rate_smoothing_s` is the window of the filter used; NaN gaps are bridged for the filter and blanked again; search half-width around the reference time (0.05), flag threshold in units of the rate noise (5), half-width around the event time in which the reference time is found (0.5) |
| `t_ref_shift_s` | reference time minus event time: the peak of the mean slip rate of all sensors within `event_time +- reference_halfwidth_s`; each sensor's `max_rate` pick is searched within `search_halfwidth_s` of it |
| `fit_before_s`, `fit_after_s` | `[start, end]` of the two fit windows, relative to `event_time` (moving `t_i` leaves them in place); defaults 75 %..25 % of the negative side and 25 %..75 % of the positive side of the extraction window |
| `same_windows` | the view applies a window edit to every sensor |
| `before_line`, `after_line` | `[slope (um/s), value at t_i (um)]` of the two least-squares lines |
| `coseismic_slip_um` | after-line minus before-line at `t_i` |
| `noise_um` | standard deviation of the residuals of the before-window line (two degrees of freedom removed): the sensor's per-sample noise, used as the error bar of the slip |
| `peak_rate_um_s` | filtered slip rate at `t_i` |
| `rate_noise_um_s` | standard deviation of the filtered slip rate in the before window; a `max_rate` peak below `rate_snr` times this is flagged `weak rate peak` |
| `flags` | one text per sensor, `"; "`-joined: `pick at search edge`, `weak rate peak`, `no rate peak`, `no crossing found` (onset or step), `before/after window clipped` (the window runs past the data), `before/after window has gaps` (less than half of its samples are finite), `before/after window reaches t_i` (`t_i` lies inside the window), `before/after fit invalid` |
| `rate_filter_tabs`, `fault_ends_x_mm` | the windows of both windowed filter tabs (`savgol_window_s`, `boxcar_window_s`), whichever is used; the fault ends the profile was cut at (NaN when none; the experiment's `analysis['fault']['ends_x_mm']` wins when the view opens) |
| `patch_x_mm` | the bare PMMA patch edges drawn, `[from, to]` (NaN when none). An event opens with the experiment's `analysis['fault']['patch_x_mm']` (see Run layout), else with this copy; edges typed in the view and not saved to the experiment stay as it moves between events. The copy counts towards "differs from the saved record" only while the experiment has no edges |
| `profile` | slip profile fit (`labquake_explorer/analysis/slip_profile.py`): `model` (`quadratic`, `eaz` or `eaz_asymmetric`), `params`, `valid`, `reason`, `n` (the sensors fitted: those in `use` with a finite slip and, when weighted, a positive `sigma`), `weighted`, `r2`, `rms_um`, `tips_mm` (where the fitted profile reaches zero), `peak_x_mm`, `peak_um`, `at_bounds`, `unresolved` (the parameters the sensors do not pin down, and `tips` when a rupture edge is only bracketed), `ranges` (name to `[lo, hi]`, the values that fit equally well where known, NaN for an open end; `tip_minus`/`tip_plus` are the brackets of free edges, and an EAZ `D` that moves with a free `lambda` has a range without being listed), `fixed` (the parameters held at the user's values), `end_slip_um` (the profile at the two fault ends where it reaches them, else NaN), `settings` (every model tab's parameters: per model and name, `fit` false for a held one, `value` the held value, `min`/`max` the user's bounds, NaN for automatic), and `use` (the sensors chosen) |
| `front` | rupture-front fit of `t_shift_s` (`t_i - event_time`) against `x`: `mode` (`unilateral` or `bilateral`, below; records before 0.3.0 say `single`, read as `unilateral`), `valid`, `reason`, `split_x_mm` and `t0_s` (bilateral: the apex and its time relative to `event_time`; NaN when there is none), `segments` (`x_from`, `x_to`, `n`, `slope_s_per_mm`, `intercept_s` at x = 0 relative to `event_time`, `velocity_m_s` positive towards +x and infinite for a flat line, `r2`, `resolved` false when the front crosses the segment within one sample, and `min_speed_m_s`, the speed of crossing it in one sample: the lower bound of an unresolved one), `use` and `use_auto` (while true, `use` follows the pick flags: the sensors whose `t_i` is doubtful, flagged `pick at search edge`, `weak rate peak` or `no rate peak`, are left out) |
| `auto_windows` | present when the Auto windows box placed windows (`labquake_explorer/analysis/fit_windows.py`), version 1: `settings` (the keyword arguments of `fit_windows.auto_windows` used: `width_s`, `min_start_s`, `max_start_s`, `step_s`, `plateau_tol_um`, `plateau_span_s`, `background_factor`, `rate_smoothing_s`, `min_after_s`, `max_decay_s`, `creep_ratio`, `onset_fraction`, `before_gap_s`, `onset_search_s`, `event_margin_s`) and, per sensor, `applied` (its `fit_before_s` / `fit_after_s` are the ones placed; false once a window is moved by hand. Such a sensor is manual, and so is every sensor of a manual event (one saved without `auto_windows`). Running the box again keeps them unless its choice says otherwise: `keep all manual windows` (the default), `keep manual events only` (the manual sensors of events with auto windows are placed again) or `overwrite everything`; the confirmation counts what is kept and what is overwritten), `kind` (`plateau`: the after line levels off; `decay`: the after window starts where the slip rate, averaged over `rate_smoothing_s`, falls to `background_factor` times the background rate; "" when not applied), `flags` (`"; "`-joined, each starting `auto:`; those ending "check the windows" or "check the after window" ask for a check by hand; `auto: negative slip, check the windows` is set from the slip measured with the placed windows and is not read back from the record), `creep_ratio` (the background rate over the event's peak slip rate, the largest from `t_i` to `t_i + min_start_s`; above `settings.creep_ratio` the sensor is flagged; records before 0.4.0 divided by the rate at `t_i`) and `background_rate_um_s` (the slope of the before line). The slip itself is measured as for windows placed by hand |

Profile models, with `x` in mm: `quadratic` is `max(0, c0 + c1 x + c2 x^2)`,
the mask being part of the model so that sensors outside the rupture sit on
the zero part; `eaz` is the earthquake arrest zone model of Ke, McLaskey &
Kammer (2021, GJI 224, 581), eq. 2, an ellipse of radius `lambda a` in the
middle and a `(a - r)^(3/2)` tip beyond `r_joint = a (sqrt(1 + 3 lambda^2) - 1)`
(parameters `D`, `xc`, `a`, `lambda`, `0 < lambda <= 1`); `eaz_asymmetric` is
eq. 3, the same with `a_minus`, `a_plus`, `lambda_minus`, `lambda_plus` on the
two sides of `xc`. Fits are bounded least squares from several starts,
unweighted unless `weighted` (then `1/sigma^2` with `sigma = noise_um`; a
sigma below 1e-6 of the median, a dead channel, is left out). A fit is invalid
when its profile stays within each sensor's sigma or it did not converge. The
sensors leave a parameter free when no sensor sees the part of the profile it
shapes: EAZ `lambda` when no sensor lies inside `r_joint` (any `lambda` up to the
value that puts `r_joint` at the nearest sensor fits equally), the extent and the
edges when no more sensors slip than the model has parameters.

The front is fitted to the `t_i` of the sensors in `use`, with the event's
sampling interval. `unilateral` is one line. `bilateral` is a hinge, two lines that
meet at a fitted apex (`split_x_mm`, `t0_s`): least squares over the apex
position, at least four sensors and two on each side; with fewer sensors, two
lines through the earliest one; one line when both sides run the same way.

The view opens with `eaz`, unweighted, and `bilateral`; an event without a
record keeps the current model, weighting, mode and every tab's held values and
bounds, and uses every sensor with `use_auto` true; a record without `profile`
or `front` is read as those defaults, with every parameter fitted within the
automatic bounds. Each model has its own tab: a value typed for a parameter
holds it there (`fit_profile(..., fixed=...)`) and the others are fitted, a
bound typed replaces the automatic one (`bounds=...`), and with every
parameter held the profile is drawn without a fit. Held parameters count as
resolved.

### `arrivals` (Pick Arrivals, per record, version 2)

Pick Arrivals works on one full-rate record at a time (the Record selector:
`waveform/elsys`, `waveform/ni`, ... or the PSU `strain`), and its results
belong to that record, `event['waveform'][recorder]['arrivals']`, so strain
and PZT picks never overwrite each other. A record keeps named pick sets
(hand picks, onset picks, peak picks, the picks of another burst):

```
{"version": 2, "current": "<set saved last>", "sets": {"<name>": {<record>}}}
```

A version-1 `arrivals` (the record itself, with its own `version`) is read as
the set `default` and becomes version 2 on the next save or with
`data.migrate`. The set saved last is the record's `current` one, which
`read_arrivals(block, event)` returns when no set is named, so the CZM
fitter's default `Cf` follows it; PZT Spectrum's "picked arrival" trigger
takes it for a new record and stores it (`trigger_pick_set`), and a saved
record keeps reading that set. As in
PZT Spectrum, a new Pick Arrivals window opens on the set saved last in the
experiment, `analysis['pick_arrivals']['current_set']`, else on the record's
current set; what an event without picks in a set starts from is kept in
`analysis['pick_arrivals']` too (see Run layout). A set name is non-empty and has no `/`. Each set's record:

| key | meaning |
|---|---|
| `picked_idx` | picked arrival sample index per channel on the record's `time` |
| `arrival_time` | the picked arrival times (s, run clock), read by the PZT view's "picked arrival" trigger |
| `enabled_channels`, `fitting_channels` | one bool per channel: shown, and used in the `Cf` fit |
| `locations` | sensor position along the fault (mm) per channel, from the record's `positions` table; `NaN` for a channel without one (an NI record's pressure or LVDT channel), which the view hides, draws past the last sensor when shown, and never fits |
| `filter` | the trace filter the picks were made on (`labquake_explorer/analysis/trace_filter.py`): `kind` `off`, `savgol` (`window_samples`, `polyorder`), or a Butterworth `lowpass`, `highpass` or `bandpass` (`order`, `zero_phase`, and `low_hz` and/or `high_hz`); `off` when the settings shown could not filter the record. Records saved before this key open with the filter left as it was |
| `rupture_speed` | fitted `Cf` in m/s: one line `time = slope * location` through the fitted picks, `Cf = -1 / slope` (the sign of the published analyses); absent when the picks define no line |
| `front_mode`, `front` | the rupture front through the same picks, fitted as in `slip_analysis` (`slip_profile.fit_front`): `unilateral` or `bilateral` (two lines meeting at a fitted apex; `single` in records before 0.3.0, read as `unilateral`), and `FrontFit.as_dict()` (`valid`, `reason`, `split_x_mm` and `t0_s` of the apex relative to the event time, `segments` with `velocity_m_s` **positive towards +x**, `r2` and `resolved`) plus `use`, the channels fitted. Records saved before these keys open in `unilateral` |
| `auto_pick` | present once "Auto pick" placed the picks (`labquake_explorer/analysis/arrival_picks.py`): `method` (`difference` from the noise level, `slope`, or `peak`, the largest departure from the noise level), `threshold` (times the noise's spread, 1.4826 MAD), `smoothing_samples` (the slope's moving average), `noise_end_s` (the noise is the record before this time, relative to the event time: its trigger, else the event time, less a tenth of the record before it, else its first fifth), `filter` (as above, the traces it ran on), `search_from_s` / `search_to_s` when a search window was set (relative to the event time: the pick, and the walk back to the quiet stretch, stay inside it, and the noise ends no later than its start) and `flags` per channel (`weak onset: ...`, `onset at the end of the noise window`, `onset at the start of the search window`, ...; "" when fine, and cleared when that pick is moved by hand). The pick is the onset of the largest change: back from it to the last stretch of at least 10 samples below the threshold, so an oscillating arrival is picked at its start. `peak` takes the largest departure itself, and, like the other two, leaves a channel whose largest value is not above the threshold unpicked and flagged weak |

The CZM fitter takes its default `Cf` from the strain record's current set.
The PSU `strain` layout keeps its `default` set in the historical places
(`strain/original/picked_idx`, `strain/original/rupture_arrival_time`,
`strain/enabled_channels`, `strain/fitting_channels`, `strain/locations`, and
`rupture_speed` at the event level), that set's further keys (`front_mode`,
`front`, ...) in `strain['arrivals']['sets']['default']`, and every other set
whole in `strain['arrivals']['sets']`. `sources.read_arrivals(block, event,
set_name)`, `write_arrivals(block, arrivals, event, set_name)`,
`arrival_sets(block)` and `current_arrival_set(block)` hide the difference
from the views.

### `czm_parms` (CZMFitterView, no version)

Dict with `Cf`, `y`, `Xc`, `Gc`, `x_min`, `x_tip`, `x_max`, `x_lim_min`,
`x_lim_max`, `strain_gauge`; `Cf` defaults to `|rupture_speed|` when Pick
Arrivals has saved one. A legacy list of eight values is still read.

A record in volts is fitted as strain, with the view's default medium. A
record stored in a stress unit (`Pa`, `kPa`, `MPa`, `GPa`: a recorder read
through a `conversion`) is fitted as stress; an experiment sets its own
medium with `analysis['medium']`: `E_Pa` and `nu`, plus the wave speeds
(`vs_m_s`, `vp_m_s`) or the density `rho_kg_m3` they follow from. In either
of those two cases the dict also holds `channel` (the channel's name),
`unit`, the medium used (`E`, `nu`, `C_s`, `C_d`) and `tau_p` (Pa), the peak
strength Gc and Xc imply at that `Cf`.

A stress record's channels are grouped by sensor through their names
(`<sensor>_sigma_xy`, `_sigma_xx`, `_sigma_yy`). The gauge box lists the
record's sensors by name, the view shows the three components of the
selected one and fits Gc and Xc to the ticked ones at once; `sensor` records
its name, `fit_components` which components (`['xy', 'xx', 'yy']` for a
rosette, `['xy']` for a shear gauge), and `channel` is the sensor's
`sigma_xy` channel. Pick Arrivals starts such a record with one trace per
sensor (its `sigma_xy`); the menus switch the other components on.

Lines and levels of a stress record: the three marker lines are named on
the plot, left to right `background end`, `tip` (the front reaches the
sensor) and `fit end`. The mean of each component over the background
window (grey band, from `x_base`, the record's start unless moved, to
`background end`) is removed, so every trace starts at 0; the model's
`sigma_xx` and `sigma_yy` are referred to the same window and its
`sigma_xy`, which is zero at the residual level, is shifted to match the
measured trace over the fit window (red band, `tip` to `fit end`). The line
under the parameters gives the misfit there. `Auto` sets `Cf` (the saved
arrivals, else the fronts at the record's sensors, kept below the Rayleigh
speed), the lines, and starting values of `Xc` and `Gc`, then fits; it tries
both sides of the fault when `sigma_xx` or `sigma_yy` are ticked and leaves
a sensor without a clear front alone. `x_base` is saved with the other keys.

A `conversion` may carry `valid`, one flag per output channel, False where
the channel is not a real measurement (an input it uses was railed or
unplugged). Event records inherit it (`waveform/<recorder>/valid`). Pick
Arrivals still draws such a channel, since it tells when the front arrives;
the CZM view draws it grey and leaves it out of every fit. `Auto` takes the
rupture speed from sensors of the same kind (cross-correlation of their
fronts), puts the tip on the largest drop of `sigma_xy` and ends the fit
window at the end of the initial rapid drop.

The model follows the rupture speed (`utils.cohesive_crack`): below the
Rayleigh speed the sub-Rayleigh solution (`CohesiveCrack`), between `C_s`
and `C_d` the supershear solution with the same cohesive zone
(`SupershearCohesiveCrack`); between the Rayleigh and shear speeds, and
above `C_d`, no steady solution exists and nothing is drawn. When it saves
the extra keys, `czm_parms` records which as `model`.

For a stress record the `Model` box can fix the solution instead (`auto`,
`sub-Rayleigh`, `supershear`; an experiment's default is
`analysis['czm']['model']`, the choice is saved as `model_choice`). A chosen
model keeps `Cf` in its own speed range: below `C_R` for `sub-Rayleigh`,
between `C_s` and `C_d` for `supershear`; a measured speed outside it is moved
to the nearest edge (0.98 of the way in) and the result line says so. `auto`
takes `Cf` back from the measured speed.

Where the model's zero of `sigma_xy` is taken in a stress record is a
choice (`Model zero` box; an experiment's default is
`analysis['czm']['reference']`, the choice is saved as `reference`):

- `fit end` (default): the stress drops at the front to a residual level;
  the model is shifted onto the measured trace over the fit window, which
  runs from `tip` to `fit end`.
- `background`: the stress rises above its level before the front and falls
  back to it, and what the record does below that level afterwards (the
  fault unloading as a whole) is not the cohesive zone. The model is
  referred to the background window like the data and never shifted; the
  fit window runs from `background end` through `tip` to `fit end`, which
  belongs where the trace falls back to the background. `Auto` puts the tip
  on the maximum above the background, `fit end` on that return, and fits
  the tip time together with `Gc` and `Xc`.

### `interevent` (InterEventView, run level, version 1)

`delay_s`, `width_s`, `lp_field`, `slip_field`, `event_times`, `recurrence`,
`lp_after`, `slip_after`, `lp_per_cycle`, `slip_per_cycle`, `coseismic_slip`
(from each event's `event_analysis.delta_x`, `displacement` in older records, used only when that
analysis was computed on the same X field, recorded as `coseismic_field`;
events skipped for that reason are listed in `coseismic_skipped`), `creep`. Each event is
sampled by a mean over `width_s` starting `delay_s` after its `event_time`;
per-cycle values are differences between consecutive events; the first event
and any window outside the run are NaN.

### `spectrum` (PZTSpectrumView, per record, store version 2)

Kept with the record it was computed on, `event['waveform'][recorder]['spectrum']`
(or `event['strain']['spectrum']` in PSU files), as
`{"version": 2, "sets": {"<name>": {"channels": {"ch<k>": record}}}}`
(`labquake_explorer/data/spectrum_store.py`): named **result sets**, one
record per analysed channel in each, so that the same channel analysed by
different routes (a displacement calibration, an EGF moment calibration)
keeps both results; the view's "Result set" box chooses one or names a new
one (any text without `/`), Save writes into it, and the record carries its
set's name as `result_set`. Store version 1, `{"version": 1, "channels":
{...}}`, is read as the set `default` and becomes version 2 on the next save
or with the migration tool; the event-level `pzt_spectrum` map of still
older files is read as the `default` set and moves likewise.
A record stores every control (trigger source and time, `trigger_pick_set`
(the pick set of a "picked arrival" trigger; absent in records before 0.4.0,
which read the current set), pre/post window and noise window in ms (a noise
window longer than the signal window keeps its samples nearest the signal), taper, baseline mode, step removal, calibration path/unit/dB,
`calibration_divisor` (a constant the CSV gain is divided by, None for 1),
t* or a per-sensor Q table (`q_csv_path`, `q_travel_time_s`, `q_hold_ends`
and the table itself, `attenuation`: `Attenuation.as_dict()` with the travel
time and the hold-ends flag, so the stored table is reused with each
sensor's travel time; the correction `exp(pi f T Q^-1(f))` applies inside
the table's band, outside it only with `q_hold_ends`; a Q table and a t*
together are refused), fit band, bins per decade, SNR threshold, n fixed, `n_max`, loss, `estimator`
(the per-log-bin statistic, `rms` (default) / `logmean` / `median`, also in
`spectrum.meta.estimator`), the uncertainty-ensemble inputs (`ensemble_on`,
`ensemble_fmins_hz`, `ensemble_fmaxs_hz`, `ensemble_calibration_paths`)
and, once run, `ensemble` (below)), the calibration
table itself (`calibration`: `Calibration.as_dict()` plus `gain_db` and
`gain_divisor`, how the CSV was read, the gains already divided; None when
uncalibrated), the source entries as typed
(`source_inputs`: phase, `rho_kg_m3`, `vp_m_s`, `vs_m_s`, `distance_m`,
`radiation_coefficient`, `free_surface_factor`, None where blank), the binned
spectrum (`spectrum`: binned frequencies, amplitudes, noise, SNR, counts and
the window/taper/calibration metadata), the fit (`fit`: `omega0` in the
spectrum's units, `fc`, `n`, weighted rms of ln residuals, bins used, band
used, `converged`, `at_bounds`, `band_limited`, `valid`, `reason`,
`n_bounds`) and, once "Compute source" has run, `source` (phase, medium
constants, distance, radiation coefficient, free-surface factor,
`seismic_moment_nm`, `mw`, `source_radius_m`, `rupture_area_m2`,
`stress_drop_pa`, `kR`, warnings).

`ensemble` (`labquake_explorer/analysis/spectrum_ensemble.py`, version 1)
is the record's fit repeated over a grid of fit bands (`bands_hz`, every
`fmin < fmax` of the `fmin` and `fmax` grids typed in the view, default
{1, 1.5, 2, 3} kHz x {15, 22, 36, 50} kHz) and over the record's own
calibration table plus the alternative CSVs listed (`calibrations`, read
with the same unit, dB flag and divisor): `members` (per member `fmin_hz`,
`fmax_hz`, `calibration` index, `omega0`, `fc`, `n`, `valid`,
`band_limited`, and `seismic_moment_nm` when the source was computed, the
moment following the plateau by the record's own M0 / Omega0), `stats`
(`min`, `max`, `p16`, `p84`, `median`, `n` of each over the valid members)
and `stats_resolved` (the same over the valid members that are not
band-limited: a band whose fmin sits at fc pins the fit at the edge), with
`n_members` / `n_valid` / `n_resolved`. It is run by "Run ensemble now", or
by Save when "run on Save" is ticked; a recompute clears it unless the
recompute reproduces the saved record from the same ensemble inputs. The
view's ensemble row and Source Scaling's error bars (the 16-84 % of M0 and
fc) use `stats_resolved` when any member is resolved, else `stats`.

The ensemble gives a range, not a point estimate: the record's own `fit` is
the estimate, and neither view shows a median. `stats_resolved` selects
members by their own outcome, so for a corner below about twice the larger
fmins the members that land low are dropped as band-limited and the
survivors are those that landed high. Its median is then biased upward (a
2 kHz corner injected into the 8 MPa BAC records came back near 4.3 kHz from
the default 16 bands), and its 16-84 % range can sit wholly above the
record's fc (276 of the 1288 BAC records of T0177/T0183/T0184/T0185). The
1 kHz members, the only ones whose band brackets such a corner, sit on the
step of the BAC table at 1.08-1.17 kHz, so choosing members by whether
their band brackets the record's own fc does not cure it for those files.
Read a low-corner record's error bar as band sensitivity, not as a bracket
around its fc.

Calibration units are `V/m`, `V/(m/s)`, `V/(m/s^2)` (a displacement spectrum
in `m*s`), `V/V` (uncalibrated, `V*s`) and `V/(N m)`: an in-situ response
that turns volts into a **moment spectrum** in `N*m*s` (the empirical
Green's function route; a Psi table in V/(N m) per `C_FM = Vp + Vs` is used
as stored with `calibration_divisor` = C_FM). With a moment spectrum
"Compute source" takes `M0 = Omega0` (`source_parameters(...,
moment_spectrum=True)`): the radius still comes from `fc`, `k` and Vs and the
stress drop from both; rho, Vp, distance, the radiation coefficient and the
free-surface factor are not used, their entries are greyed out, they are
None in `source.constants`, `constants.moment_spectrum` is True and `kR` is
NaN.

The noise window (`noise_start_ms`, `noise_end_ms`, from the trigger) lies
before the signal window (end at or before `-pre_ms`) or after it (start at
or after `post_ms`); blank entries mean the same length just before the
signal, or just after it when fewer than a quarter of the signal window's
samples lie before (a recorder keeping little pre-trigger data), and the
status line says so. `spectrum.meta.noise_region` records which: `pre`,
`post` or `explicit` (`""` when no usable noise window). A free `n`
(`n_fixed` false) is kept between 1 and the "n max" entry (`n_max`, default
4; `fit.n_bounds` is the pair used and `fit.at_bounds.n` refers to it).
Source parameters are computed also when the analysis flags the plateau as
attenuated (transient outside the taper's flat region, as a long coda under
a bh4 taper gives, `taper_energy_fraction` below 0.95): the warning is
carried in `source.warnings`, the warnings row and the status instead of
refusing the computation. Records written before 2026-11 have no `n_max`
(read as 4) and no `noise_region`.
Save also writes the controls, the medium and the sensor's own values to the
experiment's `analysis` dict (see Run layout), per result set, so a window on
any other event opens with them (on the set last saved): the set's controls
and the medium when the first channel shown has no record, a sensor's
calibration, divisor, t*, distance and free-surface factor whenever its
channel is selected without a record; switching sets does the same for the
new set. The calibration is read
from the CSV when its path is typed or browsed (the dB flag follows the
header), kept as a table and used from the table afterwards, so a saved
file works without the CSV; the frame says what is loaded.
Calibration CSVs are recognised by header: the frequency column is any
`frequency`/`freq`/`f` column, with or without a unit (`Frequency (Hz)`,
`freq_kHz`; kHz is converted), the gain column is the one mentioning gain,
amplitude, response, ratio, dB, psi or calib (or the only other column), and a
`(dB)` header is honoured when the dB flag is left to auto. A `Q^-1(f)` table
(`freq_kHz,Qp_inv_median`) with a travel time gives the frequency-dependent
attenuation correction `exp(pi f T Q^-1(f))` inside the table's band.
Conventions: amplitude = |FFT(taper * x)| dt (one-sided, no coherent-gain
division); noise divided by sqrt(mean(taper^2)); calibration converts to a
displacement spectral density in m s and is masked outside the table; fit
`ln A = ln Omega0 - ln(1 + (f/fc)^n)` in `(ln Omega0, ln fc, n)` with n fixed
at 2 by default over log-binned, SNR-gated bins.

### `source_scaling` (SourceScalingView, run level, version 2)

Also `exclude_near_field` (whether records flagged near-field, kR < 3 at the
plateau frequency, were excluded from the fit; by default they are only
warned about) and `result_set`, the spectrum result set the records were
taken from (the view's "Result set" box lists the sets saved in the run and
opens on the one the PZT Spectrum view last saved; results written before
result sets read as `default`).

Rows also carry `fc_p16`, `fc_p84`, `m0_p16`, `m0_p84` from the records'
uncertainty ensembles (geometric medians over the channels for `all`, NaN
without an ensemble), drawn as error bars.

The view's "Shared t*" frame grid-searches one constant attenuation `t*`
per sensor over the shown set's records
(`labquake_explorer/analysis/t_star_scan.py`): each record's binned
spectrum is corrected by `exp(pi f t*)` and refitted with its own settings
(band, n fixed or free with its `n_max`, loss, SNR gate), the cost being
the summed weighted ln residual squared (`rms^2 * n_bins`) over the records
that fit; the plot box switches the figure to cost against t*. The scan is
kept with the run as `t_star_scan` (version 1: `result_set`,
`t_star_grid_us` as typed, and per sensor `t_star_grid_s`, `cost`,
`n_valid`, `best_t_star_s`, `best_index`, `n_records`, `at_grid_end`) and
comes back when the set is shown again. "Save refits" writes every record of
the set refitted at its sensor's best t* into a new result set: `t_star_s`
increased by it, the binned `spectrum` corrected, `fit` redone, `source`
recomputed from the record's own constants (None when it had none),
`ensemble` dropped and `t_star_refit` (`from_set`, `shared_t_star_s`) added;
the set's controls are copied with the sensors' `t_star_s` updated. The PZT
Spectrum view, which applies t* to the raw FFT, reproduces such a record to
about a percent.

The channel and Y variable used, quality-gating choice, OLS exponent with
standard error and bootstrap 16-84 range, reduced-major-axis exponent,
coefficient, r2, the fitted `(x, y)` pairs and their event indices, excluded
events with their flags, and the distinct medium/model constant sets of the
records used.

## File formats

"Save As" writes `.npz` (a pickled experiment dict) or `.h5`/`.hdf5`; any
other suffix is refused. Files are written to a temporary sibling and moved
into place once complete, so a failed save leaves the previous file intact.
In HDF5, dicts and lists become groups tagged with a `container` attribute
so digit-keyed dicts and lists round-trip as what they were (older files
without the tag treat a group keyed `0..n-1` as a list); `None` values are
dropped; a scalar string comes back as `str` and any string list as a
`list` (a one-element list stays a list); lists numpy cannot stack (ragged
arrays, mixed content) become a group with one entry per index.

## Raw-data references and event extraction

A run may point at the acquisition files it was built from so that **Extract
Events** reads the full-rate records around picked event times instead of
copying them into the experiment file. A reference is a top-level run entry, a
dict with a `format` key, handled by `labquake_explorer/data/sources.py`:

| format | class | file |
|---|---|---|
| `tpc5` | `Tpc5Source` | Elsys TranAX tpc5, single block or ECR dual mode (block 1 continuous, blocks 2.. around triggers) |
| `ni_npz` | `NINpzSource` | National Instruments recording saved as npz (`aiN.npy` members, `sample_rate`, `channels`, `trigger_sample_index`) |
| `tpc5_legacy` | `LegacyTpc5Source` | the PSU-era `strain` dict (`filename`, `time_offset`, `time`, `raw`; no `format` key) |

Reference keys name the recorder (`elsys`, `ni`); `strain` is the PSU-era
key. Every reference carries `filename` (relative to the experiment file),
`time_offset` (`t_run = t_file + time_offset`), `fields` (the run field each
file channel feeds) and optionally `event_fields` (subset copied into events)
and `event_window_s` (`[pre, post]` seconds, replacing the start/end typed in
the Pick Events form for that source). A `Tpc5Source` also stores the block table so the
explorer knows the trigger times without opening the file; an `NINpzSource`
stores the record's metadata. `open_source(ref, base_dir, run)` rebuilds the
reader; `source.read_window(t_from, t_to, fields)` returns samples on the run
clock, or None when no record covers the time (a tpc5 with trigger blocks only
answers inside a trigger block). `Tpc5Source` reads the first measurement
group only, with one block table for every channel, so `Tpc5Source.open`
raises `ValueError` listing what it cannot read (`tpc5.layout_problems`): more
than one measurement group, selected channels whose block numbers differ, or
blocks whose sample rate, sample count, `triggerSample` or
`triggerTimeSeconds` differ between the channels.

**Adding a recorder**: subclass `Source`, implement `time_history`,
`read_window`, `to_reference`/`from_reference` (and `trigger_times` if it has
any), decorate the class with `@register_source`, and write a synthetic-file
test next to `tests/test_sources.py`.

`EventProcessor.extract_events(run, indices, window, pre, post)` slices every
1-D run field aligned with `time` around each pick (the processed channels;
the recorders' blocks with their `raw_data` volts are not copied) and adds
`waveform/<recorder>`, the
full-rate record as a channel array (`data (n_channels, n)`, `channels`,
`unit`, `positions`, `time` on the run clock, `sample_rate`, `filename`,
`block`, `trigger_time`). Which samples make the record
is the source's decision, `Source.waveform(event_time, pre, post)`: a
recorder that writes separate records around triggers (tpc5 ECR mode) copies
one whole record per event, every sample from its first pre-trigger sample
to its last (never trimmed to the window), chosen by
`sources.select_trigger_block`: among the records containing the event time,
the one overlapping the chosen window most (the lower block number on a tie);
when none contains it, the record in the window nearest to the event time
(the gap between the event time and the record's span, then the nearer
trigger, then the lower block number); nothing when no record overlaps the
window. A continuous recorder (NI npz, single-block tpc5) copies the
window itself, its own `event_window_s` taking precedence. The references are
not copied into the event. Views find the records with
`sources.waveform_blocks(event)` (keys `waveform/<recorder>`, or `strain`)
and read them through `waveform_time`, `waveform_data`, `waveform_channels`
and `waveform_store` (where per-record picks such as `picked_idx` and
`rupture_arrival_time` are kept), so the PSU layout and the new one look the
same to them; the PZT Spectrum view offers every record and opens the one
with `pzt` channels, the strain-gauge views (arrival picker, CZM) open
`strain` when present, else the first record. Recorders with no record
overlapping the event window are listed in `notes`. The legacy layout keeps its
historical output (`time`/`raw` downsampled copies plus `original` with the
first 1 % removed as baseline).

**Checking the records of an experiment file** against its tpc5 files:

```bash
python -m labquake_explorer.data.check_blocks experiment.h5 [--base-dir DIR] [--csv rows.csv] [--fill] [--no-backup]
```

For every run with events and a `tpc5` reference it lists the raw file's
trigger blocks itself (every measurement group and channel, attributes and
shapes only), takes the window from `event_extraction` (`start_s`, `end_s`),
or from the event's own time axis when the run has no extraction window,
and prints one row per event: the blocks overlapping the window, the block
the event holds and the block the rule above expects, a status (`OK`,
`NOT LOADED`, `WRONG BLOCK`, `NO BLOCK` when no block overlaps the window),
the overlapping blocks not loaded (marked `(one-block rule)` when the rule
leaves them out) and flags (blocks crossing the window start or end,
`pre-trigger truncated` (a record that is not its whole block), trigger blocks
overlapping each other), then notes on each raw file (extra measurement
groups, channels whose blocks differ, other `.tpc5` files starting with the
same name, as TranAX split files would) and a summary. It exits with 1 when an
event is not loaded, holds another block or a truncated one, else with 2 when
a raw or experiment file cannot be read. `--fill` adds the expected record to
every event that has none, built by the event processor as an extraction
would, and saves the file, keeping a copy `<name>.before-fill.h5` unless
`--no-backup` (an existing copy is not overwritten); a record already in an
event is never replaced. An `.h5` file is read with h5py, only the values
needed, so a large file is checked in seconds.

## Preprocessing package

`labquake_explorer/preprocessing/` turns raw files into the experiment dict:

- `build.run_from_tpc5(path, channel_map, base_dir, calibration, ...)` and
  `build.run_from_tpc5_ni(tpc5_path, ni_path, elsys_map, ni_map, base_dir,
  calibration, ...)` produce run dicts (time history, volt fields, `units`,
  references under `elsys` / `ni`, `calibration` record); `run_from_sources` is the general form
  (any sources, one of them the time base, the others interpolated onto its
  axis for fields it lacks). `experiment(name, runs, **metadata)` wraps them.
- `calibration.Calibration(*steps)` applies ordered steps and records them:
  `Linear` (`field = factor * source + offset`), `EddySlip` (eddy-current volts
  to `slip_k` in um, zeroed at the run start, plus `displacement`), `Friction`.
  `pressure_transducers(section)` gives the lab's stress conversions
  (`PRESSURE_PRESETS`: Tseng 2026, Table B.1, with the 1D shear factors
  tripled since 2026-11-02 because a 1D run presses one shear jack, not the
  three the table assumed; `shear_stress` and `friction` in 1D files built
  before that are three times too high and are fixed by dividing by three,
  in the runs and in every event's slice of them, along with the factor and
  formula of the `calibration` step).
  `pressure_transducers(section, calibration='e-pump-2026-11')` converts both
  lines with the transducer calibration made in place against the e-pump
  (`PRESSURE_CALIBRATIONS`, `P = 59.147 V + 1.254` bar, T0238 and T0239)
  through the jack and fault areas of the section (`PRESSURE_MPA_PER_BAR`);
  the default keeps the Table B.1 factors. A new
  conversion is a class with `apply(run, units)` and `describe()`.
- `alignment.offset_from_trigger(ni, elsys)` puts the Elsys clock on the NI
  clock from the NI trigger sample; `slip_step_table` checks that slip follows
  each PZT trigger.

The notebooks in `examples/preprocessing/` are thin scripts over this package:
paths, channel maps, the calibration, the builders, plots, save.

## Upgrading files saved by an older explorer

`labquake_explorer/data/migrate.py` brings an experiment file up to the
layouts above: run keys that were dropped, picks moved into
`event_extraction`, events reduced to the processed channels plus
`waveform/<recorder>` records (with `trigger_time` and `positions`), arrival
picks under `arrivals` (version 2: a version-1 record becomes the pick set
`default`, in the PSU `strain` layout too), `event_analysis` records at version 3, spectrum
stores at version 2 (result sets) with the flat PZT Spectrum settings of
`analysis['pzt_spectrum']` moved into its `default` set. The PSU
`strain` layout is left as it is. The patch edges and fault ends an earlier
explorer kept at the top level (`patch_x_mm`, `fault_ends_x_mm`) or in a
`fault` dict move into `analysis['fault']`. A copy of the original is kept
next to the file unless `--no-backup` is given:

```bash
python -m labquake_explorer.data.migrate experiment.h5
```

The fault geometry alone can be set without rewriting the file: with
`--patch-x-mm` and/or `--ends-x-mm` only `analysis['fault']` is written (in
place for an HDF5 file, so no copy is kept), and the legacy keys move:

```bash
python -m labquake_explorer.data.migrate t0213e.h5 --patch-x-mm=187.5,312.5 --ends-x-mm=0,500
```
