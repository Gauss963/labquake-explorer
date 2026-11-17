"""Tests for labquake_explorer.analysis.fit_windows (added on the slip-auto-windows branch)."""
import numpy as np
import pytest

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.analysis import fit_windows as fw

FS = 2000.0
EVENT_TIME = 10.0
RISE_S = 0.02                     # duration scale of the coseismic step


def trace(t, t_i, step_um, afterslip_um=0.0, afterslip_tc_s=0.3, load_um_s=2.0, noise_um=0.05, seed=0,
          rise_s=RISE_S):
    """Steady loading, a smooth step at t_i, an optional afterslip that keeps rising
    (``afterslip_um * log(1 + (t - t_i) / tc)``), and white noise."""
    rng = np.random.default_rng(seed)
    y = load_um_s * (t - t[0]) + 0.5 * step_um * (1.0 + np.tanh((t - t_i) / rise_s))
    after = np.clip(t - t_i, 0.0, None)
    y = y + afterslip_um * np.log1p(after / afterslip_tc_s)
    return y + noise_um * rng.standard_normal(t.size)


def make(rows, shift_s=0.05):
    t = EVENT_TIME + np.arange(-3.0, 3.0, 1.0 / FS)
    t_i = EVENT_TIME + shift_s
    data = np.vstack([trace(t, t_i, seed=k, **row) for k, row in enumerate(rows)])
    rate = cs.slip_rate(t, data, rate_filter="lowpass", cutoff_hz=50.0)
    return t, data, rate, [shift_s] * len(rows)


def slips(t, data, shifts, windows):
    res = cs.analyze_slip(t, data, EVENT_TIME, -3.0, 3.0, methods="manual", t_shift_s=shifts,
                          fit_before_s=[w.before for w in windows], fit_after_s=[w.after for w in windows])
    return np.array([s.coseismic_slip for s in res.sensors])


def test_plateau_finds_the_step():
    t, data, rate, shifts = make([dict(step_um=60.0)])
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert w.kind == fw.KIND_PLATEAU and w.flags == []
    a = w.after[0] - shifts[0]
    assert fw.DEFAULT_MIN_START_S <= a <= 0.3        # a sharp step levels off early
    assert w.after[1] - w.after[0] == pytest.approx(fw.DEFAULT_WIDTH_S)
    assert slips(t, data, shifts, [w])[0] == pytest.approx(60.0, abs=0.5)


def test_afterslip_without_a_plateau_starts_where_the_rate_has_settled():
    t, data, rate, shifts = make([dict(step_um=20.0, afterslip_um=5.0, afterslip_tc_s=0.1, rise_s=0.1)])
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert w.kind == fw.KIND_DECAY and w.flags == []
    assert w.background_rate == pytest.approx(2.0, rel=0.05)      # the loading rate of the trace
    a = w.after[0] - shifts[0]
    settle = fw.settle_time(t, rate[0], EVENT_TIME + shifts[0], fw.DEFAULT_BACKGROUND_FACTOR * w.background_rate)
    assert a == pytest.approx(max(settle, fw.DEFAULT_MIN_AFTER_S))
    assert fw.DEFAULT_MIN_AFTER_S < a < fw.DEFAULT_MAX_DECAY_S
    # a larger factor accepts a higher rate: the window starts earlier
    (w2,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate, background_factor=10.0)
    assert w2.after[0] < w.after[0]


def test_settle_time_follows_the_smoothed_rate():
    t = np.arange(0.0, 2.0, 1.0 / FS)
    r = np.where(t < 1.0, 10.0, 1.0)
    r[400] = 0.0                                   # one low sample before the drop: smoothed away
    assert fw.settle_time(t, r, 0.1, 2.0, smoothing_s=0.05) == pytest.approx(1.0 - 0.1, abs=0.02)
    assert fw.settle_time(t, r, 0.1, 2.0, smoothing_s=0.0) == pytest.approx(0.2 - 0.1)
    assert fw.settle_time(t, r, 0.1, 0.5) is None


def test_a_rate_that_stays_high_gets_the_provisional_start_and_a_flag():
    # afterslip that keeps the rate above five times the loading rate past t_i + 0.8 s
    t, data, rate, shifts = make([dict(step_um=20.0, afterslip_um=10.0, rise_s=0.15)])
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert w.kind == fw.KIND_DECAY
    assert fw.FLAG_NO_DECAY in w.flags
    assert w.after[0] - shifts[0] == pytest.approx(fw.DEFAULT_MAX_DECAY_S)
    # no positive background rate to compare with: the same provisional start, its own flag
    t, data2, rate2, _ = make([dict(step_um=20.0, afterslip_um=10.0, rise_s=0.15, load_um_s=-1.0)])
    (w,) = fw.auto_windows(t, data2, EVENT_TIME, shifts, rate2)
    assert fw.FLAG_NO_BACKGROUND in w.flags
    assert w.after[0] - shifts[0] == pytest.approx(fw.DEFAULT_MAX_DECAY_S)
    # a next event that leaves no room for it: the latest window that fits
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate, next_event_s=1.0)
    assert w.after[1] <= 1.0 - fw.DEFAULT_EVENT_MARGIN_S + 1e-9
    assert w.after[0] - shifts[0] < fw.DEFAULT_MAX_DECAY_S


def test_before_window_ends_ahead_of_the_onset():
    t, data, rate, shifts = make([dict(step_um=60.0)])
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert w.before[1] == pytest.approx(w.onset_s - fw.DEFAULT_BEFORE_GAP_S)
    assert w.before[1] - w.before[0] == pytest.approx(fw.DEFAULT_WIDTH_S)
    assert shifts[0] - 0.2 < w.onset_s < shifts[0]   # the onset sits on the rise, just before t_i


def test_windows_stay_clear_of_neighbouring_events():
    t, data, rate, shifts = make([dict(step_um=60.0)])
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate, prev_event_s=-1.0, next_event_s=1.0)
    assert w.after[1] <= 1.0 - fw.DEFAULT_EVENT_MARGIN_S + 1e-9
    assert w.before[0] >= -1.0 + fw.DEFAULT_EVENT_MARGIN_S - 1e-9
    assert fw.FLAG_BEFORE_SHORT in w.flags
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate, prev_event_s=-0.5, next_event_s=0.6)
    assert w.after is None and fw.FLAG_NO_ROOM_AFTER in w.flags
    assert w.before is None and fw.FLAG_NO_ROOM_BEFORE in w.flags


def test_an_after_window_moved_earlier_by_the_next_event_is_flagged():
    # afterslip without a plateau settles near t_i + 0.5 s; the next event leaves room only up to 0.95 s
    t, data, rate, shifts = make([dict(step_um=20.0, afterslip_um=5.0, afterslip_tc_s=0.1, rise_s=0.1)])
    (free,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert free.kind == fw.KIND_DECAY and fw.FLAG_AFTER_MOVED not in free.flags
    (w,) = fw.auto_windows(t, data, EVENT_TIME, shifts, rate, next_event_s=1.05)
    assert w.kind == fw.KIND_DECAY and fw.FLAG_AFTER_MOVED in w.flags
    assert w.after[0] < free.after[0] and w.after[1] <= 1.05 - fw.DEFAULT_EVENT_MARGIN_S + 1e-9


def test_creep_close_to_the_coseismic_rate_is_flagged():
    t, data, rate, shifts = make([dict(step_um=60.0), dict(step_um=2.0, load_um_s=10.0, rise_s=0.1)])
    sharp, creeping = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert sharp.creep_ratio < 0.01 and fw.FLAG_CREEP not in sharp.flags
    assert creeping.creep_ratio > fw.DEFAULT_CREEP_RATIO and fw.FLAG_CREEP in creeping.flags
    k = np.argmin(np.abs(t - EVENT_TIME - shifts[1]))
    assert creeping.creep_ratio == pytest.approx(creeping.background_rate / rate[1][k])
    (w,) = fw.auto_windows(t, data[1:], EVENT_TIME, shifts[1:], rate[1:], creep_ratio=0.9)
    assert fw.FLAG_CREEP not in w.flags


def test_plateau_start_rejects_a_level_stretch_at_the_end():
    rising = np.linspace(0.0, 10.0, 30)
    assert fw.plateau_start(rising, 0.02) is None
    level_at_end = np.concatenate([rising, np.full(12, 10.0)])
    assert fw.plateau_start(level_at_end, 0.02) is None
    level_inside = np.concatenate([rising, np.full(25, 10.0)])
    assert fw.plateau_start(level_inside, 0.02) == 28     # 9.66 um is within 0.5 um of the level


def test_one_result_per_sensor_and_input_checks():
    t, data, rate, shifts = make([dict(step_um=60.0),
                                  dict(step_um=20.0, afterslip_um=5.0, afterslip_tc_s=0.1, rise_s=0.1)])
    out = fw.auto_windows(t, data, EVENT_TIME, shifts, rate)
    assert [w.kind for w in out] == [fw.KIND_PLATEAU, fw.KIND_DECAY]
    (w,) = fw.auto_windows(t, data[:1], EVENT_TIME, [float("nan")], rate[:1])
    assert w.before is None and w.after is None
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts[:1], rate)
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts, rate[:1])
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts, rate, onset_fraction=1.5)
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts, rate, background_factor=0.0)
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts, rate, min_after_s=0.9, max_decay_s=0.8)
    with pytest.raises(ValueError):
        fw.auto_windows(t, data, EVENT_TIME, shifts, rate, creep_ratio=0.0)
