"""Trace filters for Pick Arrivals: Savitzky-Golay and Butterworth low-, high- and band-pass."""
import numpy as np
import pytest

from labquake_explorer.analysis import trace_filter as tf

FS = 100e3
T = np.arange(20000) / FS


def tone(f_hz, amp=1.0):
    return amp * np.sin(2 * np.pi * f_hz * T)


def rms(x):
    return float(np.sqrt(np.mean(x[2000:-2000] ** 2)))      # away from the ends


@pytest.mark.parametrize("kind, values, kept, removed", [
    (tf.KIND_LOWPASS, dict(high_hz=2e3), 200.0, 20e3),
    (tf.KIND_HIGHPASS, dict(low_hz=2e3), 20e3, 200.0),
    (tf.KIND_BANDPASS, dict(low_hz=1e3, high_hz=5e3), 2.5e3, 20e3),
    (tf.KIND_BANDPASS, dict(low_hz=1e3, high_hz=5e3), 2.5e3, 100.0),
])
def test_butterworth_keeps_the_band_and_removes_the_rest(kind, values, kept, removed):
    for zero_phase in (True, False):
        settings = {"kind": kind, "order": 4, "zero_phase": zero_phase, **values}
        out = tf.filter_traces(np.vstack([tone(kept), tone(removed)]), FS, settings)
        assert out.shape == (2, T.size)
        assert rms(out[0]) == pytest.approx(rms(tone(kept)), rel=0.05)
        assert rms(out[1]) < 0.02 * rms(tone(removed))


def test_a_causal_filter_never_moves_an_onset_earlier():
    onset = 10000
    step = (np.arange(T.size) >= onset).astype(float)
    settings = {"kind": tf.KIND_LOWPASS, "high_hz": 2e3, "order": 4}
    causal = tf.filter_traces([step], FS, {**settings, "zero_phase": False})[0]
    zero = tf.filter_traces([step], FS, {**settings, "zero_phase": True})[0]
    assert np.all(np.abs(causal[:onset]) < 1e-9)                   # nothing before the onset
    assert np.abs(zero[onset - 50:onset]).max() > 0.01              # zero phase spreads it earlier
    # starting from the first sample's steady state: an offset does not ring at the start
    offset = tf.filter_traces([3.0 + 0 * T], FS, {**settings, "zero_phase": False})[0]
    np.testing.assert_allclose(offset, 3.0, atol=1e-9)


def test_savgol_off_offsets_and_nans():
    from scipy import signal
    x = tone(500.0) + 0.1 * np.random.default_rng(0).standard_normal(T.size)
    np.testing.assert_allclose(tf.filter_traces([x], FS, {"kind": tf.KIND_SAVGOL, "window_samples": 21})[0],
                               signal.savgol_filter(x, 21, 2))
    np.testing.assert_array_equal(tf.filter_traces([x], FS, {"kind": tf.KIND_OFF})[0], x)
    out = tf.filter_traces([5.0 + 0.1 * tone(10e3)], FS, {"kind": tf.KIND_HIGHPASS, "low_hz": 1e3})
    assert abs(np.median(out[0][2000:-2000])) < 1e-3                   # a high-pass removes an offset
    x = tone(500.0)
    x[100:110] = np.nan                                                # NaN samples stay NaN, the rest finite
    out = tf.filter_traces([x], FS, {"kind": tf.KIND_LOWPASS, "high_hz": 5e3})[0]
    assert np.isnan(out[100:110]).all() and np.isfinite(out[:100]).all() and np.isfinite(out[110:]).all()


@pytest.mark.parametrize("settings, message", [
    ({"kind": "notch"}, "unknown filter"),
    ({"kind": tf.KIND_LOWPASS, "high_hz": 60e3}, "below the Nyquist frequency, 50 kHz"),
    ({"kind": tf.KIND_HIGHPASS, "low_hz": 0.0}, "low corner must be a frequency > 0"),
    ({"kind": tf.KIND_HIGHPASS}, "low corner must be a frequency > 0"),
    ({"kind": tf.KIND_BANDPASS, "low_hz": 5e3, "high_hz": 1e3}, "below the high corner"),
    ({"kind": tf.KIND_LOWPASS, "high_hz": 1e3, "order": 0}, "the order must be 1 to 10"),
    ({"kind": tf.KIND_SAVGOL, "window_samples": 2}, "longer than 2 samples"),
])
def test_unusable_settings_say_why(settings, message):
    with pytest.raises(ValueError, match=message):
        tf.filter_traces([tone(500.0)], FS, settings)


def test_describe():
    assert tf.describe({"kind": tf.KIND_OFF}) == "no filter"
    assert tf.describe({"kind": tf.KIND_BANDPASS, "low_hz": 2e3, "high_hz": 200e3, "order": 4}) == \
        "Butterworth band-pass 2-200 kHz, order 4, zero phase"
    assert tf.describe({"kind": tf.KIND_HIGHPASS, "low_hz": 500.0, "zero_phase": False}) == \
        "Butterworth high-pass above 0.5 kHz, order 4, causal"
