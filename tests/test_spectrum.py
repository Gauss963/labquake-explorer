"""Tests for labquake_explorer.analysis.spectrum against synthetic signals."""
import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis import spectrum as sp

FS = 1e6
N = 4000
T0 = 2e-3          # pulse centre (s) - sample 2000 of the 4000-sample trace
PRE = POST = 1e-3  # 2000-sample analysis window centred on the pulse


def make_pulse(fc, omega0=1.0):
    return sp.brune_pulse(FS, N, omega0, fc, T0)


def spectrum_of(t, x, **kw):
    return sp.compute_spectrum(sp.extract_window(t, x, T0, PRE, POST), **kw)


def wobble():
    """A slow 300 Hz wobble at 100 kHz, no transient (for the noise-window tests)."""
    t = np.arange(0, 0.05, 1 / 100e3)
    return t, 0.01 * np.sin(2 * np.pi * 300 * t)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------
def test_sampling_rate_and_window_by_sample_counts():
    t = np.arange(N) / FS
    assert sp.sampling_rate(t) == pytest.approx(FS)
    assert sp.sampling_rate(t + np.random.default_rng(1).normal(0, 1e-9, t.size)) == pytest.approx(FS, rel=1e-2)
    for bad in ([0.0], [1.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            sp.sampling_rate(bad)
    x = np.arange(N, dtype=float)
    w = sp.extract_window(t, x, T0, 3e-4, 7e-4, baseline="none")
    assert (w.n_pre, w.n_post, w.n, w.i_trigger) == (300, 700, 1000, 2000) and w.fs == pytest.approx(FS)
    np.testing.assert_allclose(w.signal, x[1700:2700])
    np.testing.assert_allclose(w.t_rel, (np.arange(1000) - 300) / FS)
    assert w.t_rel[w.n_pre] == 0.0
    # equal-length noise window directly before the signal, never overlapping
    assert w.noise_available and w.noise_fraction == 1.0
    np.testing.assert_allclose(w.noise, x[700:1700])
    # the trigger snaps to the nearest sample
    assert sp.extract_window(t, x, T0 + 0.4 / FS, 3e-4, 7e-4, baseline="none").i_trigger == 2000
    for args, kw in (((2.5e-3, 1e-3), {}), ((1e-3, 2.5e-3), {}), ((-1e-3, 1e-3), {}),
                     ((1e-3, 1e-3), {"baseline": "bogus"}),
                     ((1e-3, 1e-3), {"remove_step": True, "step_centre": "bogus"})):
        with pytest.raises(ValueError):
            sp.extract_window(t, x, T0, *args, **kw)
    with pytest.raises(ValueError):
        sp.extract_window(t, x[:-1], T0, 1e-3, 1e-3)


def test_noise_window_shortened_then_after_the_signal_then_unavailable():
    t = np.arange(N) / FS
    x = np.random.default_rng(0).normal(size=N)
    # signal window of 1000 samples starting at sample 500: only 500 noise samples (50%)
    w = sp.extract_window(t, x, 800 / FS, 3e-4, 7e-4, baseline="none")
    assert w.noise_available and w.noise.size == 500 and w.noise_fraction == pytest.approx(0.5)
    assert w.noise_region == "pre" and w.i_noise1 == w.i_trigger - w.n_pre
    np.testing.assert_allclose(w.noise, x[0:500])
    # exactly 25% is still accepted
    w = sp.extract_window(t, x, 550 / FS, 3e-4, 7e-4, baseline="none")
    assert w.noise_available and w.noise.size == 250 and w.noise_region == "pre"
    # below 25% before the signal: the stretch after it, when there is one
    w = sp.extract_window(t, x, 500 / FS, 3e-4, 7e-4, baseline="none")
    assert w.noise_available and w.noise_region == "post" and w.noise.size == 1000 and w.noise_fraction == 1.0
    assert w.i_noise0 == w.i_trigger + w.n_post
    np.testing.assert_allclose(w.noise, x[1200:2200])
    assert sp.compute_spectrum(w).meta["noise_region"] == "post"
    # and none on either side: noise is NaN and flagged
    w = sp.extract_window(t[:1300], x[:1300], 500 / FS, 3e-4, 7e-4, baseline="none")
    assert not w.noise_available and w.noise_region == "" and w.noise_fraction == 0.0 and w.i_noise0 == -1
    assert np.all(np.isnan(w.noise))
    s = sp.compute_spectrum(w)
    assert np.all(np.isnan(s.amp_noise)) and np.all(np.isnan(s.noise_binned))
    assert s.meta["noise_available"] is False and s.meta["noise_region"] == ""


def test_pre_linear_and_highpass_baselines():
    t, pulse = make_pulse(20e3)
    trend = 0.3 + 40.0 * t
    tol = 1e-6 * pulse.max()
    w = sp.extract_window(t, pulse + trend, T0, PRE, POST)
    assert w.baseline_mode == "pre_linear" and w.baseline_region == "noise"
    np.testing.assert_allclose(w.signal, pulse[1000:3000], atol=tol)
    np.testing.assert_allclose(w.noise, pulse[0:1000], atol=tol)
    # no noise window: the line is fitted over the pre-trigger samples
    w = sp.extract_window(t, pulse + trend, 1000 / FS, 1e-3, 1e-3)
    assert w.baseline_region == "pre_trigger"
    np.testing.assert_allclose(w.signal, pulse[0:2000], atol=tol)
    # 'none' keeps the raw samples
    w = sp.extract_window(t, pulse + trend, T0, PRE, POST, baseline="none")
    np.testing.assert_allclose(w.signal, (pulse + trend)[1000:3000])
    assert w.baseline_region == "none"
    # highpass: needs a corner below Nyquist, removes a DC offset
    x = pulse + 5.0
    for kw in ({}, {"highpass_hz": FS}):
        with pytest.raises(ValueError):
            sp.extract_window(t, x, T0, PRE, POST, baseline="highpass", **kw)
    w = sp.extract_window(t, x, T0, PRE, POST, baseline="highpass", highpass_hz=200.0)
    assert w.highpass_hz == 200.0 and w.baseline_region == "highpass"
    w_ref = sp.extract_window(t, pulse, T0, PRE, POST, baseline="highpass", highpass_hz=200.0)
    np.testing.assert_allclose(w.signal, w_ref.signal, atol=tol)
    np.testing.assert_allclose(w.noise, w_ref.noise, atol=tol)
    assert int(np.argmax(w.signal)) == w.n_pre                 # zero-phase: the peak stays at the trigger
    # well above the corner the spectrum is untouched
    fit = sp.fit_omega_n(sp.compute_spectrum(w), snr_min=None, fmin=2e3)
    assert fit.valid and fit.fc == pytest.approx(20e3, rel=0.03) and fit.omega0 == pytest.approx(1.0, rel=0.03)


def test_explicit_noise_window_before_or_after_the_signal():
    from scipy import signal as sps
    t, x = wobble()
    trigger, pre_s, post_s = 0.030, 0.0005, 0.002
    default = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="none")
    n = default.n_pre + default.n_post
    assert default.i_noise1 == default.i_trigger - default.n_pre and default.i_noise1 - default.i_noise0 == n
    # a noise window of its own, with a gap before the signal
    before = (-0.0040, -0.0025)
    w = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="none", noise_window=before)
    assert w.noise_available and w.i_noise0 == w.i_trigger - 400 and w.i_noise1 == w.i_trigger - 250
    np.testing.assert_allclose(w.noise, x[w.i_noise0:w.i_noise1])
    assert w.noise_fraction == pytest.approx(150 / n)
    # the pre-linear baseline is fitted over that window at its true position
    wb = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="pre_linear", noise_window=before)
    assert wb.baseline_region == "noise"
    i0 = wb.i_trigger - wb.n_pre
    k = np.arange(wb.i_noise0 - i0, wb.i_noise1 - i0, dtype=float)
    slope, intercept = np.polyfit(k, x[wb.i_noise0:wb.i_noise1], 1)
    np.testing.assert_allclose(wb.noise, x[wb.i_noise0:wb.i_noise1] - (slope * k + intercept), atol=1e-12)
    # too short to count; badly ordered or overlapping the signal on either side is refused
    short = sp.extract_window(t, x, trigger, pre_s, post_s, noise_window=(-0.0012, -0.0010))
    assert not short.noise_available and short.i_noise0 == -1
    with pytest.raises(ValueError):
        sp.extract_window(t, x, trigger, pre_s, post_s, noise_window=(-0.001, -0.002))
    for bad in ((-0.001, 0.001), (0.001, 0.003), (-0.003, 0.0)):
        with pytest.raises(ValueError, match="before the signal window .* or after it"):
            sp.extract_window(t, x, trigger, pre_s, post_s, noise_window=bad)
    # a gap with the highpass baseline filters the two stretches on their own
    hp = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="highpass", highpass_hz=1e3, noise_window=before)
    assert hp.noise_available and hp.noise.size == 150 and np.all(np.isfinite(hp.noise))
    # right after the signal window, or after a gap
    after = (0.002, 0.0035)
    w = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="none", noise_window=after)
    assert w.noise_available and w.noise_region == "explicit"
    assert w.i_noise0 == w.i_trigger + w.n_post and w.i_noise1 == w.i_trigger + 350
    np.testing.assert_allclose(w.noise, x[w.i_noise0:w.i_noise1])
    assert w.noise_fraction == pytest.approx(150 / w.n) and sp.compute_spectrum(w).meta["noise_region"] == "explicit"
    gap = sp.extract_window(t, x, trigger, pre_s, post_s, noise_window=(0.003, 0.0045))
    assert gap.noise_available and gap.i_noise0 == gap.i_trigger + 300
    # the pre-linear baseline then comes from the pre-trigger samples, not the coda, and is taken off the noise too
    wb = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="pre_linear", noise_window=after)
    assert wb.baseline_region == "pre_trigger"
    slope, intercept = np.polyfit(np.arange(wb.n_pre, dtype=float), x[i0:wb.i_trigger], 1)
    k = np.arange(wb.i_noise0 - i0, wb.i_noise1 - i0)
    np.testing.assert_allclose(wb.noise, x[wb.i_noise0:wb.i_noise1] - (slope * k + intercept), atol=1e-12)
    np.testing.assert_allclose(wb.signal, x[i0:wb.i_trigger + wb.n_post] - (slope * np.arange(wb.n) + intercept), atol=1e-12)
    # the highpass baseline filters an adjacent noise window jointly with the signal, no edge at the join
    hp = sp.extract_window(t, x, trigger, pre_s, post_s, baseline="highpass", highpass_hz=1e3, noise_window=after)
    assert hp.noise_available and hp.noise.size == 150 and np.all(np.isfinite(hp.noise))
    joint = sps.sosfiltfilt(sps.butter(2, 1e3, btype="highpass", fs=100e3, output="sos"), x[i0:hp.i_noise1], padlen=12)
    np.testing.assert_allclose(hp.signal, joint[:hp.n], atol=1e-12)
    np.testing.assert_allclose(hp.noise, joint[hp.n:], atol=1e-12)
    # automatic: 50 samples before a 250-sample window are under a quarter, so the stretch after it is used
    w = sp.extract_window(t, x, 0.003, 0.0025, 0.0, baseline="none")
    assert w.noise_available and w.noise_region == "post" and w.i_noise0 == w.i_trigger + w.n_post
    assert w.noise.size == w.n and w.noise_fraction == 1.0


def test_taper_window():
    w = sp.taper_window(1000, "tukey", 0.25)
    assert w.size == 1000 and w[0] == 0.0 and w[-1] == 0.0 and np.all(w[130:870] == 1.0)
    assert np.all(sp.taper_window(100, "tukey", 0.0) == 1.0)
    bh = sp.taper_window(1001, "bh4")
    assert bh[500] == pytest.approx(1.0) and bh[0] < 1e-4
    for args in ((10, "hann"), (10, "tukey", 2.0), (0,)):
        with pytest.raises(ValueError):
            sp.taper_window(*args)


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("taper", ["tukey", "bh4"])
def test_unit_impulse_gives_amplitude_dt(taper):
    t = np.arange(N) / FS
    x = np.zeros(N)
    x[2000] = 1.0
    s = spectrum_of(t, x, taper=taper)
    assert s.f[0] == 0.0 and s.f[-1] == pytest.approx(FS / 2)
    # (an even-length bh4 taper peaks between samples: 1 - 1.4e-6 at the trigger)
    np.testing.assert_allclose(s.amp_signal, 1.0 / FS, rtol=1e-5)
    np.testing.assert_allclose(s.amp_binned, 1.0 / FS, rtol=1e-5)
    np.testing.assert_allclose(s.amp_noise, 0.0, atol=1e-30)
    assert s.amp_units == "V*s" and not s.calibrated
    assert s.counts.sum() == np.count_nonzero((s.f >= s.meta["fmin"]) & (s.f <= s.meta["fmax"])) and np.all(s.counts > 0)
    assert s.meta["taper"] == taper and s.meta["n"] == 2000


def test_nfft_padding_and_log_binning():
    t, x = make_pulse(20e3)
    w = sp.extract_window(t, x, T0, PRE, POST)
    s = sp.compute_spectrum(w, nfft=4096)
    assert s.f.size == 4096 // 2 + 1 and s.meta["nfft"] == 4096
    s0 = sp.compute_spectrum(w)                            # denser grid, same underlying transform
    assert s0.f.size == 1001
    assert np.interp(20e3, s.f, s.amp_signal) == pytest.approx(np.interp(20e3, s0.f, s0.amp_signal), rel=1e-3)
    assert sp.compute_spectrum(w, nfft=16).f.size == 1001  # a smaller nfft is ignored
    with pytest.raises(ValueError):
        sp.compute_spectrum(w, taper="hann")
    # binning averages power and drops empty bins
    s = sp.compute_spectrum(w, fmin=1e3, fmax=4e5, bins_per_decade=10)
    assert s.meta["fmin"] == 1e3 and s.meta["fmax"] == 4e5 and s.meta["bins_per_decade"] == 10
    assert s.n_bins <= math.ceil(math.log10(400) * 10) and np.all(s.counts > 0)
    assert np.all((s.f_binned >= 1e3) & (s.f_binned <= 4e5)) and np.all(np.diff(s.f_binned) > 0)
    for fb, ab in zip(s.f_binned, s.amp_binned):           # sqrt(mean(power)) of the analytic spectrum near the centre
        assert ab == pytest.approx(sp.brune_spectrum(fb, 1.0, 20e3), rel=0.15)
    s2 = sp.bin_spectrum(s, 1e3, 1.4e3, 100)               # narrower than the raw grid: empty bins dropped
    assert s2.n_bins < 30 and np.all(s2.counts > 0)
    for args in ((0.0, 1e3), (1e4, 1e3), (1e3, 1e4, 0)):
        with pytest.raises(ValueError):
            sp.bin_spectrum(s, *args)


def test_brune_pulse_matches_analytic_spectrum():
    t, x = make_pulse(20e3, omega0=3.0)
    dt = 1.0 / FS
    # centred, two-sided exponential with area omega0
    assert t[np.argmax(x)] == pytest.approx(T0) and np.sum(x) * dt == pytest.approx(3.0, rel=1e-6)
    f = np.fft.rfftfreq(N, dt)
    np.testing.assert_allclose(np.abs(np.fft.rfft(x)) * dt, sp.brune_spectrum(f, 3.0, 20e3), rtol=1e-8)
    _, v = sp.brune_pulse(FS, N, 3.0, 20e3, T0, derivative=1)
    inner = slice(1, -1)   # DC is 0 and the imaginary Nyquist term cannot survive a real irfft
    np.testing.assert_allclose((np.abs(np.fft.rfft(v)) * dt)[inner],
                               (sp.brune_spectrum(f, 3.0, 20e3) * 2 * np.pi * f)[inner], rtol=1e-8)


def test_noise_spectrum_is_scaled_to_signal_length():
    sigma = 0.7
    t = np.arange(20000) / FS
    x = np.random.default_rng(3).normal(0, sigma, t.size)
    expected = sigma / FS * math.sqrt(2000)     # |X| dt of white noise over n samples
    w_full = sp.extract_window(t, x, 10000 / FS, PRE, POST, baseline="none")
    w_half = sp.extract_window(t, x, 2000 / FS, PRE, POST, baseline="none")  # only 1000 noise samples
    assert w_half.noise.size == 1000
    incoherent_gain = math.sqrt(np.mean(sp.taper_window(2000, "tukey", 0.25) ** 2))
    for w in (w_full, w_half):
        s = sp.compute_spectrum(w)
        sel = s.f > 1e4
        # noise: incoherent-gain corrected (and scaled to the signal length for the short window)
        assert np.sqrt(np.mean(s.amp_noise[sel] ** 2)) == pytest.approx(expected, rel=0.05)
        # signal: a transient estimate, no gain division, so pure noise reads lower by sqrt(mean(w^2))
        assert np.sqrt(np.mean(s.amp_signal[sel] ** 2)) == pytest.approx(expected * incoherent_gain, rel=0.05)
        assert np.nanmedian(s.snr) == pytest.approx(incoherent_gain, abs=0.15)


def test_t_star_and_attenuation_table_corrections():
    t, x = make_pulse(20e3)
    w = sp.extract_window(t, x + np.random.default_rng(0).normal(0, 1e-3, N), T0, PRE, POST)
    s0 = sp.compute_spectrum(w)
    s1 = sp.compute_spectrum(w, t_star_s=2e-6)
    corr = np.exp(np.pi * s0.f * 2e-6)
    np.testing.assert_allclose(s1.amp_signal, s0.amp_signal * corr, rtol=1e-12)
    np.testing.assert_allclose(s1.amp_noise, s0.amp_noise * corr, rtol=1e-12)
    assert s1.t_star_s == 2e-6 and s1.meta["t_star_s"] == 2e-6
    assert sp.t_star_from_q(0.1, 50.0, 5000.0) == pytest.approx(0.1 / (50 * 5000))
    with pytest.raises(ValueError):
        sp.t_star_from_q(0.1, 0.0, 5000.0)
    with pytest.raises(ValueError):
        sp.compute_spectrum(w, t_star_s=-1.0)
    # a Q(f) table: the same factor per frequency, signal and noise; not together with t*
    n = 4096
    t, x = sp.brune_pulse(1e6, n, 1e-9, 2e4, n / 1e6 / 2)
    w = sp.extract_window(t, x, t[n // 2], 0.001, 0.002, baseline="none")
    plain = sp.compute_spectrum(w, nfft=8192)
    att = sp.Attenuation.from_arrays([1e4, 1e5], [0.1, 0.01], 5e-5)
    corrected = sp.compute_spectrum(w, nfft=8192, attenuation=att)
    ok = np.isfinite(plain.amp_signal) & (plain.amp_signal > 0)
    np.testing.assert_allclose(corrected.amp_signal[ok] / plain.amp_signal[ok], att.factor(plain.f)[ok], rtol=1e-12)
    np.testing.assert_allclose(corrected.amp_noise[ok] / plain.amp_noise[ok], att.factor(plain.f)[ok], rtol=1e-12)
    assert corrected.meta["attenuation"]["travel_time_s"] == 5e-5
    with pytest.raises(ValueError):
        sp.compute_spectrum(w, t_star_s=1e-5, attenuation=att)


def test_q_csv_table_and_held_ends(tmp_path):
    f_khz = np.array([20.0, 40.0, 80.0, 160.0])
    q_inv = np.array([0.3, 0.15, 0.075, 0.0375])
    path = _csv(tmp_path / "q.csv", "freq_kHz,Qp_inv_median", zip(f_khz, q_inv))
    att = sp.load_q_csv(path, travel_time_s=5.5e-5)
    assert att.fmin == 2e4 and att.fmax == 1.6e5 and att.q_inv_at(40e3) == pytest.approx(0.15)
    # log-log interpolation of a power law is exact
    assert att.q_inv_at(np.sqrt(20e3 * 40e3)) == pytest.approx(np.sqrt(0.3 * 0.15))
    assert att.factor(40e3) == pytest.approx(np.exp(np.pi * 40e3 * 5.5e-5 * 0.15))
    np.testing.assert_allclose(att.factor(np.array([1e3, 1e6])), 1.0)   # outside the table: unchanged
    np.testing.assert_allclose(sp.Attenuation.from_dict(att.as_dict()).q_inv, att.q_inv)
    with pytest.raises(ValueError):
        sp.Attenuation.from_arrays(f_khz * 1e3, q_inv, -1.0)
    # outside the table: NaN and no correction, unless the end values are held
    att = sp.Attenuation.from_arrays([1e3, 1e4], [0.1, 0.01], 1e-4)
    assert att.hold_ends is False and np.isnan(att.q_inv_at([500.0, 2e4])).all() and att.factor([500.0])[0] == 1.0
    held = sp.Attenuation.from_arrays([1e3, 1e4], [0.1, 0.01], 1e-4, hold_ends=True)
    assert held.q_inv_at([500.0])[0] == pytest.approx(0.1) and held.q_inv_at([2e4])[0] == pytest.approx(0.01)
    assert held.factor([2e4])[0] == pytest.approx(np.exp(np.pi * 2e4 * 1e-4 * 0.01))
    np.testing.assert_allclose(held.q_inv_at([1e3, 3e3, 1e4]), att.q_inv_at([1e3, 3e3, 1e4]))   # the same inside
    assert held.as_dict()["hold_ends"] is True and sp.Attenuation.from_dict(held.as_dict()).hold_ends
    assert sp.Attenuation.from_dict(att.as_dict()).hold_ends is False
    loaded = sp.load_q_csv(_csv(tmp_path / "q2.csv", "freq_kHz,Qp_inv_median", [(1, 0.1), (10, 0.01)]), 2e-4,
                           hold_ends=True)
    assert loaded.hold_ends and loaded.travel_time_s == 2e-4 and loaded.fmin == 1e3


def test_bin_estimators():
    t, x = make_pulse(20e3)
    w = sp.extract_window(t, x + np.random.default_rng(3).normal(0, 1e-3 * np.abs(x).max(), x.size), T0, PRE, POST)
    rms = sp.compute_spectrum(w)
    logmean = sp.compute_spectrum(w, estimator="logmean")
    median = sp.compute_spectrum(w, estimator="median")
    assert rms.meta["estimator"] == "rms"
    for s in (logmean, median):
        assert s.n_bins == rms.n_bins and np.allclose(s.f_binned, rms.f_binned)
        assert np.all(np.isfinite(s.noise_binned)) and s.meta["noise_subtracted"] is False
    assert np.all(logmean.amp_binned <= rms.amp_binned * (1 + 1e-9))      # a geometric mean never exceeds the rms
    # a bin's logmean is the geometric mean of its raw amplitudes, its median their median
    f, a = rms.f, rms.amp_signal
    lo, hi, bpd = rms.meta["fmin"], rms.meta["fmax"], rms.meta["bins_per_decade"]
    edges = np.logspace(np.log10(lo), np.log10(hi), int(math.ceil((np.log10(hi) - np.log10(lo)) * bpd)) + 1)
    edges[0], edges[-1] = lo, hi                                       # the edges bin_spectrum uses
    b = np.digitize(logmean.f_binned[3], edges) - 1
    inside = (f >= lo) & (f <= hi) & (np.clip(np.digitize(f, edges) - 1, 0, edges.size - 2) == b) & np.isfinite(a) & (a > 0)
    assert logmean.amp_binned[3] == pytest.approx(np.exp(np.mean(np.log(a[inside]))), rel=1e-6)
    assert median.amp_binned[3] == pytest.approx(np.median(a[inside]), rel=1e-6)
    # the fit re-bins with the same estimator; a bad name is refused; subtraction only applies to rms
    assert sp.fit_omega_n(logmean, bins_per_decade=20, snr_min=None).valid
    with pytest.raises(ValueError):
        sp.compute_spectrum(w, estimator="mean")
    assert sp.compute_spectrum(w, estimator="median", noise_subtract=True).meta["noise_subtracted"] is False
    assert sp.bin_spectrum(rms, estimator="logmean").meta["estimator"] == "logmean"


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def gain_displacement(f):
    """A frequency-dependent displacement sensitivity (V/m) for the tests."""
    return 100.0 * (np.asarray(f, dtype=float) / 1e4) ** 0.3


def make_volts(fc, gain_fn, omega0=1.0):
    """Voltage trace whose spectrum is omega0/(1+(f/fc)^2) * gain_fn(f)."""
    dt = 1.0 / FS
    f = np.fft.rfftfreq(N, dt)
    g = gain_fn(f)
    g[0] = g[1]
    spec = sp.brune_spectrum(f, omega0, fc) * g * np.exp(-2j * np.pi * f * T0)
    return np.arange(N) * dt, np.fft.irfft(spec, N) / dt


def write_csv(path, fcol, gcol, f, g):
    return _csv(path, f"{fcol},{gcol}", [(f"{a:.10g}", f"{b:.10g}") for a, b in zip(f, g)])


def _csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    return path


def test_calibration_csv_reading(tmp_path):
    f = np.array([1e3, 1e4, 1e5])
    g = np.array([1.0, 10.0, 100.0])
    cal = sp.load_calibration_csv(write_csv(tmp_path / "c.csv", "Frequency_Hz", "GAIN", f, g), "V/m")
    assert cal.unit == "V/m" and cal.k == 0 and cal.fmin == 1e3 and cal.fmax == 1e5
    np.testing.assert_allclose(cal.interp(f), g)
    assert cal.interp(math.sqrt(1e3 * 1e4)) == pytest.approx(math.sqrt(10.0))   # log-log midpoint
    assert np.isnan(cal.interp(999.0)) and np.isnan(cal.interp(1e5 + 1)) and np.isnan(cal.interp(0.0))
    assert cal.as_dict()["n_rows"] == 3 and cal.as_dict()["unit"] == "V/m"
    # dB gains, alias column names, unsorted rows
    cal_db = sp.load_calibration_csv(write_csv(tmp_path / "d.csv", "freq", "response", f[::-1], 20 * np.log10(g[::-1])),
                                     "V/(m/s)", gain_is_db=True)
    np.testing.assert_allclose(cal_db.interp(f), g)
    assert cal_db.k == 1 and sp.Calibration.from_arrays(f, g, "V/(m/s^2)").k == 2
    assert sp.Calibration.from_arrays(f, g, "V/V").output_units == "V*s"
    # a bare 'hz' column is a frequency and the only other column is the gain; a kHz column is scaled
    assert sp.load_calibration_csv(write_csv(tmp_path / "b.csv", "hz", "volts", f[:2], f[:2]), "V/m").gain[0] == 1e3
    khz = sp.load_calibration_csv(_csv(tmp_path / "k.csv", "freq_kHz,gain", [(1.0, 1), (10.0, 1), (100.0, 1)]), "V/V")
    np.testing.assert_allclose(khz.frequency_hz, [1e3, 1e4, 1e5])
    # headers of the lab's tools: a '(dB)' unit in the header is detected, an explicit flag wins
    fl = np.logspace(3, 5, 25)
    db_path = _csv(tmp_path / "bac.csv", "Frequency (Hz),Average (dB)", zip(fl, np.full(25, 20 * np.log10(2.0))))
    cal = sp.load_calibration_csv(db_path, "V/V", gain_is_db=None)
    assert cal.frequency_hz.size == 25 and sp.csv_gain_is_db(db_path) is True
    np.testing.assert_allclose(cal.gain, 2.0)
    np.testing.assert_allclose(sp.load_calibration_csv(db_path, "V/V", gain_is_db=False).gain, 20 * np.log10(2.0))
    pw = _csv(tmp_path / "pw.csv", "Frequency (Hz),Piecewise Fit 3-seg (dB)", zip(fl, np.full(25, 20 * np.log10(2.0))))
    assert sp.load_calibration_csv(pw, "V/m", None).gain[0] == pytest.approx(2.0)
    # a linear Psi column with an empty tail
    fp = np.logspace(2.3, 6.4, 40)
    rows = [(fi, 1e-11) for fi in fp[:-3]] + [(fi, "") for fi in fp[-3:]]
    psi = _csv(tmp_path / "egf.csv", "Frequency_Hz,Mean_Inst_App_Psi", rows)
    cal = sp.load_calibration_csv(psi, "V/m", gain_is_db=None)
    assert cal.frequency_hz.size == 37 and cal.fmax == pytest.approx(fp[-4]) and sp.csv_gain_is_db(psi) is False
    np.testing.assert_allclose(cal.gain, 1e-11)
    # errors: missing file, no frequency-like column, ambiguous gain, unknown unit, a single row
    with pytest.raises(FileNotFoundError):
        sp.load_calibration_csv(tmp_path / "missing.csv", "V/m")
    for path, unit in ((write_csv(tmp_path / "a.csv", "x", "gain", f, f), "V/m"),
                       (_csv(tmp_path / "amb.csv", "frequency,a,b", [(1e3, 1, 2), (1e4, 1, 2)]), "V/V"),
                       (write_csv(tmp_path / "u.csv", "frequency", "gain", f, f), "counts/m")):
        with pytest.raises(ValueError):
            sp.load_calibration_csv(path, unit)
    with pytest.raises(ValueError):
        sp.Calibration.from_arrays([1e3], [1.0], "V/m")


def test_velocity_and_displacement_calibrations_agree(tmp_path):
    fc = 20e3
    t, volts = make_volts(fc, gain_displacement)
    f_cal = np.geomspace(300.0, 5e5, 60)
    g_d = gain_displacement(f_cal)
    cal_d = sp.load_calibration_csv(write_csv(tmp_path / "disp.csv", "frequency_hz", "gain", f_cal, g_d), "V/m")
    cal_v = sp.load_calibration_csv(write_csv(tmp_path / "vel.csv", "f", "amplitude", f_cal, g_d / (2 * np.pi * f_cal)),
                                    "V/(m/s)")
    cal_a = sp.Calibration.from_arrays(f_cal, 20 * np.log10(g_d / (2 * np.pi * f_cal) ** 2), "V/(m/s^2)", gain_is_db=True)
    w = sp.extract_window(t, volts, T0, PRE, POST)
    results = {}
    for name, cal in (("disp", cal_d), ("vel", cal_v), ("acc", cal_a)):
        s = sp.compute_spectrum(w, calibration=cal)
        assert s.amp_units == "m*s" and s.calibrated and s.meta["calibration"]["unit"] == cal.unit
        fit = sp.fit_omega_n(s, snr_min=None)
        assert fit.valid, fit.reason
        assert fit.omega0 == pytest.approx(1.0, rel=0.02) and fit.fc == pytest.approx(fc, rel=0.02)
        results[name] = s
    np.testing.assert_allclose(results["vel"].amp_binned, results["disp"].amp_binned, rtol=1e-3)
    np.testing.assert_allclose(results["acc"].amp_binned, results["disp"].amp_binned, rtol=1e-3)
    # uncalibrated: volts in, volts*s out, and the raw plateau is the voltage plateau
    s_raw = sp.compute_spectrum(w)
    assert s_raw.amp_units == "V*s" and not s_raw.calibrated
    assert sp.fit_omega_n(s_raw, snr_min=None).omega0 != pytest.approx(1.0, rel=0.1)
    # 'V/V' passes the numbers through but stays uncalibrated
    s_vv = sp.compute_spectrum(w, calibration=sp.Calibration.from_arrays(f_cal, np.ones_like(f_cal), "V/V"))
    assert s_vv.amp_units == "V*s" and not s_vv.calibrated


def test_calibration_masks_outside_table_range(tmp_path):
    t, volts = make_volts(20e3, gain_displacement)
    f_cal = np.geomspace(2e3, 2e5, 30)
    cal = sp.load_calibration_csv(write_csv(tmp_path / "c.csv", "frequency", "gain", f_cal, gain_displacement(f_cal)), "V/m")
    s = spectrum_of(t, volts, calibration=cal)
    outside = (s.f < 2e3) | (s.f > 2e5)
    assert np.all(np.isnan(s.amp_signal[outside])) and np.all(np.isnan(s.amp_noise[outside]))
    assert np.all(np.isfinite(s.amp_signal[~outside]))
    assert np.all((s.f_binned >= 2e3) & (s.f_binned <= 2e5)) and s.f_binned.min() < 2.5e3 and s.f_binned.max() > 1.6e5
    fit = sp.fit_omega_n(s, snr_min=None)
    assert fit.valid and fit.band_used[0] >= 2e3 and fit.band_used[1] <= 2e5
    assert fit.omega0 == pytest.approx(1.0, rel=0.03) and fit.fc == pytest.approx(20e3, rel=0.03)


def test_calibration_record_hash_and_moment_unit(tmp_path):
    f_cal = np.geomspace(300.0, 5e5, 25)
    g = gain_displacement(f_cal)
    cal = sp.load_calibration_csv(write_csv(tmp_path / "c.csv", "frequency_hz", "gain", f_cal, g), "V/m")
    d = cal.as_dict()
    json.dumps(d)
    assert d["frequency_hz"] == pytest.approx(f_cal.tolist()) and d["gain"] == pytest.approx(g.tolist())
    assert d["sha256"] == cal.sha256 and len(d["sha256"]) == 16 and d["source"].endswith("c.csv")
    # the record regenerates the calibration without the CSV
    again = sp.Calibration.from_dict(d)
    f_test = np.geomspace(400.0, 4e5, 50)
    np.testing.assert_allclose(again.interp(f_test), cal.interp(f_test))
    assert again.sha256 == cal.sha256 and again.unit == "V/m" and again.k == 0
    # a different table or unit hashes differently
    assert sp.Calibration.from_arrays(f_cal, 2 * g, "V/m").sha256 != cal.sha256
    assert sp.Calibration.from_arrays(f_cal, g, "V/(m/s)").sha256 != cal.sha256
    # and it travels with the spectrum meta
    s = spectrum_of(*make_volts(20e3, gain_displacement), calibration=cal)
    assert s.meta["calibration"]["sha256"] == cal.sha256 and len(s.meta["calibration"]["gain"]) == 25
    json.dumps(s.as_dict())
    # a moment sensitivity, stored already divided by the gain divisor
    cal = sp.Calibration.from_arrays([100.0, 1e5], [4628.0, 4628.0], "V/(N m)", gain_divisor=4628.0)
    assert cal.output_units == "N*m*s" and cal.k == 0 and sp.CALIBRATION_UNITS["V/(N m)"] == 0
    np.testing.assert_allclose(cal.gain, [1.0, 1.0])
    np.testing.assert_allclose(sp.Calibration.from_dict(cal.as_dict()).gain, [1.0, 1.0])
    s = spectrum_of(*make_pulse(20e3), calibration=cal)
    assert s.amp_units == "N*m*s" and s.calibrated and sp.fit_omega_n(s, snr_min=None).omega0 == pytest.approx(1.0, rel=0.02)
    csv = tmp_path / "psi.csv"
    csv.write_text("Frequency (Hz),Mean_Inst_App_Psi\n100,9256\n100000,9256\n")
    np.testing.assert_allclose(sp.load_calibration_csv(csv, "V/(N m)", False, gain_divisor=4628.0).gain, [2.0, 2.0])
    assert sp.load_calibration_csv(csv, "V/m").gain[0] == 9256.0
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            sp.Calibration.from_arrays([1.0, 10.0], [1.0, 1.0], "V/m", gain_divisor=bad)


# ---------------------------------------------------------------------------
# omega-n fit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fc", [5e3, 20e3, 80e3])
def test_brune_recovery_clean_and_with_noise_30db_below_plateau(fc):
    t, x = make_pulse(fc, omega0=2.5e-9)
    s = spectrum_of(t, x)
    fit = sp.fit_omega_n(s, snr_min=None)
    assert fit.valid, fit.reason
    assert fit.omega0 == pytest.approx(2.5e-9, rel=0.02) and fit.fc == pytest.approx(fc, rel=0.02)
    assert fit.n == 2.0 and fit.n_fixed and fit.converged and not fit.band_limited
    assert not any(fit.at_bounds.values()) and fit.rms < 0.05
    assert fit.n_bins >= 8 and fit.band_used[0] < fit.band_used[1] and fit.amp_units == "V*s"
    np.testing.assert_allclose(fit.model([fit.fc]), [fit.omega0 / 2], rtol=1e-12)
    # free n on clean data comes back near 2
    fit_free = sp.fit_omega_n(s, snr_min=None, n_fixed=False)
    assert fit_free.valid and not fit_free.n_fixed and fit_free.n == pytest.approx(2.0, abs=0.1)
    assert fit_free.fc == pytest.approx(fc, rel=0.05)
    # the SNR gate on a noise-free trace (SNR = inf) does not drop bins
    fit_gated = sp.fit_omega_n(s)
    assert fit_gated.valid and fit_gated.n_bins == fit.n_bins
    # white noise whose amplitude spectral density (|X| dt over the window) is 30 dB below the plateau
    t, x = make_pulse(fc)
    sigma = 1.0 / math.sqrt(1000.0) * FS / math.sqrt(int(round((PRE + POST) * FS)))
    s = spectrum_of(t, x + np.random.default_rng(0).normal(0, sigma, N))
    fit = sp.fit_omega_n(s)
    assert fit.valid, fit.reason
    assert fit.snr_min == 3.0 and fit.converged
    assert fit.omega0 == pytest.approx(1.0, rel=0.05) and fit.fc == pytest.approx(fc, rel=0.05)
    # measured noise level is as designed, and the band stops before the noise crossing
    assert np.nanmedian(s.noise_binned) == pytest.approx(1.0 / math.sqrt(1000.0), rel=0.15)
    f_cross = fc * math.sqrt(1000.0 - 1.0)          # analytic S = N
    f_snr3 = fc * math.sqrt(1000.0 / 3.0 - 1.0)     # analytic S = 3 N
    assert fit.band_used[1] < min(f_cross, FS / 2) and fit.band_used[1] < min(1.6 * f_snr3, FS / 2)
    used = (s.f_binned >= fit.band_used[0]) & (s.f_binned <= fit.band_used[1])
    assert np.count_nonzero(s.snr[used] >= 3.0) == fit.n_bins and fit.n_bins < s.n_bins
    assert sp.fit_omega_n(s, snr_min=10.0).n_bins < fit.n_bins        # a stricter gate uses fewer bins


def test_fixed_and_free_n_flag_fc_above_band():
    s = spectrum_of(*make_pulse(300e3), fmax=100e3)
    fixed = sp.fit_omega_n(s, snr_min=None, n_fixed=True)
    free = sp.fit_omega_n(s, snr_min=None, n_fixed=False)
    for fit in (fixed, free):
        assert fit.valid and fit.band_limited and fit.band_used[1] <= 100e3 and fit.fc > fit.band_used[1] / 2
    assert (free.at_bounds["n"] or free.at_bounds["ln_fc"]) and fixed.at_bounds["ln_fc"]
    # a well-resolved corner is not flagged
    assert not sp.fit_omega_n(spectrum_of(*make_pulse(20e3)), snr_min=None).band_limited


def test_fit_invalid_cases_never_raise():
    t, x = make_pulse(20e3)
    s = spectrum_of(t, x)
    # too few bins
    fit = sp.fit_omega_n(s, fmin=1e4, fmax=1.2e4, snr_min=None)
    assert not fit.valid and "bins" in fit.reason and fit.n_bins < 8
    assert math.isnan(fit.omega0) and math.isnan(fit.fc) and np.all(np.isnan(fit.model([1e3, 1e4])))
    assert not fit.converged and not fit.band_limited
    # noise unavailable (the window fills the trace) but SNR gating requested
    w_nn = sp.extract_window(t[:2000], x[:2000], 1000 / FS, 1e-3, 1e-3)
    assert not w_nn.noise_available
    fit = sp.fit_omega_n(sp.compute_spectrum(w_nn))
    assert not fit.valid and "snr_min" in fit.reason
    assert sp.fit_omega_n(sp.compute_spectrum(w_nn), snr_min=None).valid
    # all-NaN / zero spectrum; the SNR gate removes everything on pure noise
    s_nan = sp.SpectrumResult(f=s.f, amp_signal=np.full(s.f.shape, np.nan), amp_noise=s.amp_noise,
                              amp_units="V*s", f_binned=np.array([]), amp_binned=np.array([]),
                              noise_binned=np.array([]), snr=np.array([]), counts=np.array([], dtype=int),
                              calibrated=False, t_star_s=None, meta={})
    zero = spectrum_of(t, np.zeros(N))
    for spec, kw in ((s_nan, {}), (s_nan, {"bins_per_decade": 30}), (zero, {})):
        fit = sp.fit_omega_n(spec, snr_min=None, **kw)
        assert not fit.valid and fit.reason
    noise_only = spectrum_of(t, np.random.default_rng(1).normal(size=N))
    fit = sp.fit_omega_n(noise_only, snr_min=50.0)
    assert not fit.valid and fit.n_bins < 8
    with pytest.raises(ValueError):
        sp.fit_omega_n(s, loss="huber")


def test_fit_options_soft_l1_rebinning_and_n_bounds():
    s = spectrum_of(*make_pulse(20e3))
    fit = sp.fit_omega_n(s, snr_min=None, loss="soft_l1")
    assert fit.valid and fit.loss == "soft_l1"
    assert fit.omega0 == pytest.approx(1.0, rel=0.02) and fit.fc == pytest.approx(20e3, rel=0.02)
    fit10 = sp.fit_omega_n(s, snr_min=None, bins_per_decade=10)
    assert fit10.valid and fit10.n_bins < fit.n_bins and fit10.fc == pytest.approx(20e3, rel=0.03)
    fit_band = sp.fit_omega_n(s, fmin=2e3, fmax=2e5, snr_min=None)
    assert 2e3 <= fit_band.band_used[0] and fit_band.band_used[1] <= 2e5
    # free-n bounds are a parameter
    default = sp.fit_omega_n(s, snr_min=None, n_fixed=False)
    assert default.n_bounds == sp.N_BOUNDS == (1.0, 4.0) and default.n == pytest.approx(2.0, abs=0.05)
    wide = sp.fit_omega_n(s, snr_min=None, n_fixed=False, n_bounds=(1, 8))
    assert wide.n_bounds == (1.0, 8.0) and wide.n == pytest.approx(2.0, abs=0.05) and not wide.at_bounds["n"]
    assert wide.as_dict()["n_bounds"] == [1.0, 8.0]
    # a floor above the truth: n sits on it, and at_bounds refers to the bounds given
    high = sp.fit_omega_n(s, snr_min=None, n_fixed=False, n_bounds=(3, 8))
    assert high.valid and high.n == pytest.approx(3.0, abs=0.05) and high.at_bounds["n"]
    assert any("n at bound" in msg for msg in high.warnings)
    # a fixed n ignores them; an invalid fit still reports them; bad bounds are refused
    assert sp.fit_omega_n(s, snr_min=None, n_fixed=True, n_bounds=(3, 8)).n == 2.0
    assert sp.fit_omega_n(s, snr_min=None, fmin=1e9, n_bounds=(2, 6)).n_bounds == (2.0, 6.0)
    for bad in ((4, 1), (0, 4), (1,), (1, float("nan")), "ab"):
        with pytest.raises(ValueError):
            sp.fit_omega_n(s, snr_min=None, n_bounds=bad)


def test_results_meta_and_json():
    t, x = make_pulse(20e3)
    w = sp.extract_window(t, x, T0 + 0.3 / FS, PRE, POST)
    assert w.trigger_time == T0 + 0.3 / FS and w.i_trigger == 2000
    s = sp.compute_spectrum(w)
    assert s.meta["i_trigger"] == 2000 and s.meta["trigger_time"] == T0 + 0.3 / FS
    assert s.meta["step_width_fraction"] is None and math.isnan(s.meta["step_residual"])
    s = sp.compute_spectrum(sp.extract_window(t, x, T0, PRE, POST, remove_step=True))
    fit = sp.fit_omega_n(s, snr_min=None)
    d = s.as_dict()
    assert d["version"] == sp.RESULT_VERSION and len(d["f_binned"]) == s.n_bins and "f" not in d   # no raw grid
    json.dumps(d)
    fd = fit.as_dict()
    assert fd["valid"] and fd["band_used"] == list(fit.band_used) and fd["version"] == sp.RESULT_VERSION
    assert fd["warnings"] == list(fit.warnings) == []
    json.dumps(fd)
    for key in ("n", "nfft", "fs", "taper", "baseline_mode", "step_removed", "step_centre_method",
                "noise_available", "calibration", "amplitude_convention", "fmin", "fmax",
                "i_trigger", "trigger_time", "taper_energy_fraction", "transient_in_flat_region",
                "step_width_fraction", "step_residual", "noise_subtract", "noise_subtracted"):
        assert key in s.meta
    assert s.meta["step_centre_method"] == "antisymmetric" and s.meta["step_width_fraction"] == sp.STEP_WIDTH_FRACTION
    assert sp.extract_window(t, x, T0, PRE, POST, remove_step=True).step_width_fraction == sp.STEP_WIDTH_FRACTION
    # a fitted step width is recorded as None
    w_fit = sp.extract_window(t, x, T0, PRE, POST, remove_step=True, step_width_fraction=None)
    assert w_fit.step_width_fraction is None and w_fit.step_removed
    d = sp.compute_spectrum(w_fit).as_dict()
    json.dumps(d)
    assert d["meta"]["step_width_fraction"] is None and d["meta"]["step_removed"] is True


def test_taper_energy_fraction_flags_off_centre_transient():
    t, x = make_pulse(20e3)
    # pulse peak at 5% of a 2000-sample window: inside the tukey ramp (alpha/2 = 12.5%)
    w_edge = sp.extract_window(t, x, T0, 1e-4, 1.9e-3)
    assert int(np.argmax(w_edge.signal)) == 100
    s_edge = sp.compute_spectrum(w_edge)
    frac = s_edge.meta["taper_energy_fraction"]
    assert frac == pytest.approx(np.sum(sp.taper_window(2000) * w_edge.signal ** 2) / np.sum(w_edge.signal ** 2))
    assert frac < 0.95 and s_edge.meta["transient_in_flat_region"] is False
    fit_edge = sp.fit_omega_n(s_edge, snr_min=None)
    assert fit_edge.valid and fit_edge.omega0 < 0.9        # attenuated, and flagged
    assert any("taper" in wmsg for wmsg in fit_edge.warnings) and fit_edge.as_dict()["warnings"] == list(fit_edge.warnings)
    # peak at 50%: flat region, no warning, plateau intact
    s_mid = spectrum_of(t, x)
    assert s_mid.meta["taper_energy_fraction"] > 0.99 and s_mid.meta["transient_in_flat_region"] is True
    fit_mid = sp.fit_omega_n(s_mid, snr_min=None)
    assert fit_mid.omega0 == pytest.approx(1.0, rel=0.02) and fit_mid.warnings == ()
    # a zero window has no defined fraction and is not "in the flat region"
    s0 = spectrum_of(t, np.zeros(N))
    assert math.isnan(s0.meta["taper_energy_fraction"]) and s0.meta["transient_in_flat_region"] is False


def test_converged_false_when_optimizer_stops_early(monkeypatch):
    t, x = make_pulse(20e3)
    sigma = 1.0 / math.sqrt(1000.0) * FS / math.sqrt(2000)
    s = spectrum_of(t, x + np.random.default_rng(0).normal(0, sigma, N))
    good = sp.fit_omega_n(s)
    assert good.valid and good.converged and good.status in (1, 2, 4)
    used = (s.f_binned >= good.band_used[0]) & (s.f_binned <= good.band_used[1]) & (s.snr >= 3.0)
    resid_norm = good.rms * math.sqrt(np.sum(s.counts[used]))
    assert good.optimality <= sp.CONVERGED_REL_OPTIMALITY * resid_norm
    # the same fit stopped after two evaluations is reported as not converged
    orig = sp.least_squares
    monkeypatch.setattr(sp, "least_squares", lambda *a, **kw: orig(*a, **dict(kw, max_nfev=2)))
    early = sp.fit_omega_n(s)
    assert early.valid and not early.converged and early.status == 0
    assert early.optimality > sp.CONVERGED_REL_OPTIMALITY * resid_norm
    assert any("not converged" in wmsg for wmsg in early.warnings)
    # clean data: gradient far below the absolute floor, converged
    monkeypatch.setattr(sp, "least_squares", orig)
    clean = sp.fit_omega_n(spectrum_of(t, x), snr_min=None)
    assert clean.converged and clean.optimality < sp.CONVERGED_ABS_OPTIMALITY


@pytest.mark.parametrize("fc", [20e3, 80e3])
def test_noise_bias_direction_and_noise_subtraction(fc):
    """Noise adds in quadrature in the bins near the SNR cut: fc reads high and
    Omega0 low by a few percent at 30 dB.  Pinned so a change is visible;
    ``noise_subtract`` removes most of it."""
    t, x = make_pulse(fc)
    sigma = 1.0 / math.sqrt(1000.0) * FS / math.sqrt(2000)
    ratios, ratios_sub, o0, o0_sub = [], [], [], []
    for seed in range(12):
        w = sp.extract_window(t, x + np.random.default_rng(seed).normal(0, sigma, N), T0, PRE, POST)
        s = sp.compute_spectrum(w)
        s_sub = sp.compute_spectrum(w, noise_subtract=True)
        assert s.meta["noise_subtract"] is False and s.meta["noise_subtracted"] is False
        assert s_sub.meta["noise_subtract"] is True and s_sub.meta["noise_subtracted"] is True
        assert np.all(s_sub.amp_binned <= s.amp_binned) and np.all(s_sub.amp_binned >= 0)
        np.testing.assert_allclose(s_sub.noise_binned, s.noise_binned)
        fit, fit_sub = sp.fit_omega_n(s), sp.fit_omega_n(s_sub)
        assert fit.valid and fit_sub.valid
        ratios.append(fit.fc / fc); ratios_sub.append(fit_sub.fc / fc)
        o0.append(fit.omega0); o0_sub.append(fit_sub.omega0)
    bias, bias_sub = np.mean(ratios) - 1.0, np.mean(ratios_sub) - 1.0
    assert 0.005 < bias < 0.06                    # systematic +2..+4% (max ~+12% per seed)
    assert np.mean(o0) < 1.0                      # and Omega0 reads low
    assert abs(bias_sub) < abs(bias)              # subtraction shrinks the bias
    assert abs(np.mean(o0_sub) - 1.0) < abs(np.mean(o0) - 1.0)
    assert sp.fit_omega_n(s_sub, bins_per_decade=20).valid    # re-binning inside the fit keeps the subtraction
    # without a noise window (the window fills the trace) the request is recorded but nothing is subtracted
    w_nn = sp.extract_window(t[:2000], x[:2000], 1000 / FS, 1e-3, 1e-3)
    s_nn = sp.compute_spectrum(w_nn, noise_subtract=True)
    assert s_nn.meta["noise_subtracted"] is False
    np.testing.assert_allclose(s_nn.amp_binned, sp.compute_spectrum(w_nn).amp_binned)


# ---------------------------------------------------------------------------
# coseismic step removal
# ---------------------------------------------------------------------------
def logistic_step(t, amplitude, width_s, t0=T0):
    return amplitude / (1.0 + np.exp(-(t - t0) / width_s))


def step_trace(n_win, fs, width_fraction=None, shape="logistic", rise_time=None):
    """A Brune pulse (fc = 20 kHz at 1 MHz, 5 kHz at 200 kHz) centred in a trace of
    ``2 * n_win`` samples plus a co-located step of 3x the pulse peak: a logistic of
    scale ``width_fraction * n_win`` samples, or the tanh step of
    ``synthetic.add_strain`` (a logistic of scale ``rise_time / 2``).
    Returns ``(t, pulse, step, fc, t0, half_window_s, true_scale_samples)``."""
    fc = 20e3 if fs >= 1e6 else 5e3
    t0 = n_win / fs
    t, x = sp.brune_pulse(fs, 2 * n_win, 1.0, fc, t0)
    amp = 3.0 * x.max()
    if shape == "logistic":
        scale = width_fraction * n_win
        step = logistic_step(t, amp, scale / fs, t0)
    else:
        scale = rise_time * fs / 2.0
        step = amp * 0.5 * (1.0 + np.tanh((t - t0) / rise_time))
    return t, x, step, fc, t0, n_win / 2.0 / fs, scale


def fit_ok(fit, fc, rel):
    return fit.valid and abs(fit.omega0 - 1.0) < rel and abs(fit.fc - fc) < rel * fc


# (window samples, fs): the 1 MHz reference and the strain-block rate of
# tests/synthetic.py add_strain (fs = 2e5) at the sizes that broke the old search
@pytest.mark.parametrize("n_win, fs", [(2000, 1e6), (5000, 2e5), (20000, 2e5)])
@pytest.mark.parametrize("width_fraction", [0.003, 0.01, 0.03])
def test_step_removal_recovers_brune_parameters(n_win, fs, width_fraction):
    t, x, step, fc, t0, half, true_scale = step_trace(n_win, fs, width_fraction)
    kw = {} if width_fraction == sp.STEP_WIDTH_FRACTION else {"step_width_fraction": width_fraction}
    # without removal the step dominates the spectrum
    w_off = sp.extract_window(t, x + step, t0, half, half)
    assert not w_off.step_removed and math.isnan(w_off.step_amplitude) and math.isnan(w_off.step_residual)
    assert not fit_ok(sp.fit_omega_n(sp.compute_spectrum(w_off), snr_min=None), fc, 0.10)
    # with removal at the step's own width (the default 1% for the middle case)
    w_on = sp.extract_window(t, x + step, t0, half, half, remove_step=True, **kw)
    assert w_on.step_removed and w_on.step_centre_method == "antisymmetric"
    assert w_on.step_amplitude == pytest.approx(step.max(), rel=0.02)
    assert abs(w_on.step_centre - w_on.n_pre) < 0.5                # centre error under a sample
    assert w_on.step_scale == pytest.approx(true_scale) and w_on.step_fit_scale == pytest.approx(true_scale, rel=0.05)
    assert w_on.step_residual < 0.01
    fit_on = sp.fit_omega_n(sp.compute_spectrum(w_on), snr_min=None)
    assert fit_on.valid, fit_on.reason
    assert fit_on.omega0 == pytest.approx(1.0, rel=0.10) and fit_on.fc == pytest.approx(fc, rel=0.10)
    assert np.mean(w_on.signal[-n_win // 20:]) == pytest.approx(0.0, abs=0.02 * x.max())
    # with the width fitted
    w_fit = sp.extract_window(t, x + step, t0, half, half, remove_step=True, step_width_fraction=None)
    assert w_fit.step_width_fraction is None and w_fit.step_scale == w_fit.step_fit_scale
    assert abs(w_fit.step_centre - w_fit.n_pre) < 0.5
    assert w_fit.step_fit_scale == pytest.approx(true_scale, rel=0.05) and w_fit.step_residual < 0.01
    fit = sp.fit_omega_n(sp.compute_spectrum(w_fit), snr_min=None)
    assert fit.omega0 == pytest.approx(1.0, rel=0.05) and fit.fc == pytest.approx(fc, rel=0.05)
    if kw and n_win == 2000:
        # the fixed default width mismatches the step shape: recovery fails and the residual says so
        w_fixed = sp.extract_window(t, x + step, t0, half, half, remove_step=True)
        assert w_fixed.step_fit_scale == pytest.approx(true_scale, rel=0.05)
        assert w_fixed.step_scale == pytest.approx(sp.STEP_WIDTH_FRACTION * n_win) and w_fixed.step_residual > 0.05
        assert not fit_ok(sp.fit_omega_n(sp.compute_spectrum(w_fixed), snr_min=None), fc, 0.10)


def test_tanh_step_at_strain_block_rate():
    """The synthetic strain step (tanh, rise_time 20 us at 200 kHz) is only a
    2-sample logistic; the fitted-width path must handle it."""
    t, x, step, fc, t0, half, true_scale = step_trace(5000, 2e5, shape="tanh", rise_time=2e-5)
    assert true_scale == pytest.approx(2.0)
    w = sp.extract_window(t, x + step, t0, half, half, remove_step=True, step_width_fraction=None)
    assert abs(w.step_centre - w.n_pre) < 0.5 and w.step_fit_scale == pytest.approx(2.0, rel=0.10) and w.step_residual < 0.01
    fit = sp.fit_omega_n(sp.compute_spectrum(w), snr_min=None)
    assert fit.omega0 == pytest.approx(1.0, rel=0.05) and fit.fc == pytest.approx(fc, rel=0.05)
    # the default 1% width (50 samples) is far too wide for it and is flagged
    w_fixed = sp.extract_window(t, x + step, t0, half, half, remove_step=True)
    assert w_fixed.step_residual > 0.3
    assert not fit_ok(sp.fit_omega_n(sp.compute_spectrum(w_fixed), snr_min=None), fc, 0.10)


def test_step_centre_trigger_method():
    t, x, step, fc, t0, half, true_scale = step_trace(2000, 1e6, 0.01)
    # trigger on the step: deterministic centre at n_pre, scale still fitted
    w = sp.extract_window(t, x + step, t0, half, half, remove_step=True, step_centre="trigger")
    assert w.step_centre_method == "trigger" and w.step_centre == float(w.n_pre)
    assert w.step_fit_scale == pytest.approx(true_scale, rel=0.05) and w.step_residual < 0.01
    fit = sp.fit_omega_n(sp.compute_spectrum(w), snr_min=None)
    assert fit.omega0 == pytest.approx(1.0, rel=0.05) and fit.fc == pytest.approx(fc, rel=0.05)
    w_fit = sp.extract_window(t, x + step, t0, half, half, remove_step=True, step_centre="trigger",
                              step_width_fraction=None)
    assert w_fit.step_scale == w_fit.step_fit_scale == pytest.approx(true_scale, rel=0.05)
    # trigger 5 samples late: 'trigger' keeps the pick (and reports the mismatch),
    # 'antisymmetric' finds the true centre 5 samples before it
    late = t0 + 5 / 1e6
    w_t = sp.extract_window(t, x + step, late, half, half, remove_step=True, step_centre="trigger")
    w_a = sp.extract_window(t, x + step, late, half, half, remove_step=True)
    assert w_t.step_centre == float(w_t.n_pre) and w_t.step_residual > 0.05
    assert w_a.step_centre == pytest.approx(w_a.n_pre - 5, abs=0.5) and w_a.step_residual < 0.01


def test_step_centre_methods_and_fitted_scale_floor():
    t, x = make_pulse(20e3)
    peak = x.max()
    sharp = 3.0 * peak * (t >= T0)
    smooth = logistic_step(t, 3.0 * peak, 0.01 * 2000 / FS)
    # sharp step at the pulse peak: both methods locate it (within a sample)
    w_a = sp.extract_window(t, x + sharp, T0, PRE, POST, remove_step=True)
    w_p = sp.extract_window(t, x + sharp, T0, PRE, POST, remove_step=True, step_centre="peak")
    assert w_a.step_centre == pytest.approx(w_a.n_pre, abs=1.0) and w_a.step_fit_scale == pytest.approx(sp.STEP_MIN_SCALE)
    assert w_p.step_centre == pytest.approx(w_p.n_pre, abs=1.0) and w_p.step_centre_method == "peak"
    # smooth step larger than the pulse: peak-|x| lands on the plateau, antisymmetric does not
    w_a = sp.extract_window(t, x + smooth, T0, PRE, POST, remove_step=True)
    w_p = sp.extract_window(t, x + smooth, T0, PRE, POST, remove_step=True, step_centre="peak")
    assert w_a.step_centre == pytest.approx(w_a.n_pre, abs=1.0) and w_p.step_centre > w_p.n_pre + 100
    # a step-free pulse is left essentially untouched
    w0 = sp.extract_window(t, x, T0, PRE, POST, remove_step=True)
    assert abs(w0.step_amplitude) < 1e-6 * peak and math.isnan(w0.step_residual)
    np.testing.assert_allclose(w0.signal, x[1000:3000], atol=1e-6 * peak)
    # a one-sample (Heaviside) step cannot be matched by a logistic of scale >= 0.5 samples:
    # the fit must not collapse below the floor, and the residual says the shape did not fit
    w_fit = sp.extract_window(t, x + sharp, T0, PRE, POST, remove_step=True, step_width_fraction=None)
    assert w_fit.step_fit_scale >= sp.STEP_MIN_SCALE and w_fit.step_scale == w_fit.step_fit_scale
    assert w_fit.step_residual > 0.1
    w_fixed = sp.extract_window(t, x + sharp, T0, PRE, POST, remove_step=True)
    assert w_fixed.step_residual > 0.3
    for w in (w_fit, w_fixed):                # never wider than the edge region, never narrower than the floor
        assert sp.STEP_MIN_SCALE <= w.step_fit_scale <= 0.1 * w.n
    # a matched smooth step leaves no antisymmetric residual
    w_ok = sp.extract_window(t, x + smooth, T0, PRE, POST, remove_step=True, step_width_fraction=None)
    assert w_ok.step_residual < 1e-3 and sp.compute_spectrum(w_ok).meta["step_residual"] == w_ok.step_residual
