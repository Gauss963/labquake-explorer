"""Automatic arrival picks: difference from the noise, slope, peak."""
import numpy as np
import pytest

from labquake_explorer.analysis import arrival_picks as ap

N, ONSET, NOISE_END = 4000, 2500, 2000


def trace(onset=ONSET, amp=1.0, noise=0.01, rise=20, seed=0, spike_at=None, spike=0.0, oscillate=False):
    """Noise, then from ``onset`` a rise over ``rise`` samples (or a decaying oscillation)."""
    rng = np.random.default_rng(seed)
    k = np.arange(N, dtype=float)
    after = np.clip(k - onset, 0.0, None)
    if oscillate:
        x = amp * np.sin(2 * np.pi * after / 40.0) * (1.0 - np.exp(-after / rise)) * np.exp(-after / 800.0)
    else:
        x = amp * (1.0 - np.exp(-after / rise))
    x = x + noise * rng.standard_normal(N)
    if spike_at is not None:
        x[spike_at] += spike
    return x


@pytest.mark.parametrize("method", [ap.METHOD_DIFFERENCE, ap.METHOD_SLOPE])
def test_the_onset_is_found_within_a_few_samples(method):
    for seed in range(5):
        (pick,) = ap.auto_pick([trace(seed=seed)], method, NOISE_END)
        assert pick.flag == "" and abs(pick.index - ONSET) <= 6
        (pick,) = ap.auto_pick([-trace(seed=seed, oscillate=True)], method, NOISE_END)
        assert pick.flag == "" and abs(pick.index - ONSET) <= 6        # the sign and the shape do not matter


def test_a_noise_spike_before_the_arrival_does_not_take_the_pick():
    x = trace(spike_at=ONSET - 300, spike=0.3)                         # well above threshold, but not the largest
    (pick,) = ap.auto_pick([x], ap.METHOD_DIFFERENCE, NOISE_END)
    assert abs(pick.index - ONSET) <= 6
    first_above = NOISE_END + int(np.flatnonzero(np.abs(x[NOISE_END:]) > 5 * 0.01)[0])
    assert first_above < ONSET - 200                                   # a first-crossing pick would take the spike


def test_flags_and_bad_input():
    # 2 noise spreads of arrival: its largest swing with the noise stays below 8 spreads
    (pick,) = ap.auto_pick([trace(amp=0.02)], ap.METHOD_DIFFERENCE, NOISE_END, threshold=8.0)
    assert pick.index is None and pick.flag == ap.FLAG_WEAK
    (pick,) = ap.auto_pick([trace(amp=0.02)], ap.METHOD_DIFFERENCE, NOISE_END, threshold=1.0)
    assert pick.index is not None                                      # a lower threshold accepts it
    (pick,) = ap.auto_pick([trace(onset=1990)], ap.METHOD_DIFFERENCE, NOISE_END)   # arrival inside the noise
    assert pick.index == NOISE_END and pick.flag == ap.FLAG_AT_NOISE_END
    (pick,) = ap.auto_pick([trace()], ap.METHOD_SLOPE, 5)
    assert pick.index is None and pick.flag == ap.FLAG_NO_NOISE
    for method, threshold in (("nope", 5.0), (ap.METHOD_DIFFERENCE, 0.0)):
        with pytest.raises(ValueError):
            ap.auto_pick([trace()], method, NOISE_END, threshold=threshold)


def test_an_oscillation_is_picked_at_its_start_not_at_its_largest_cycle():
    x = trace(oscillate=True, rise=200)                                 # the largest swing comes cycles later
    assert int(np.argmax(np.abs(x))) - ONSET > 100
    (pick,) = ap.auto_pick([x], ap.METHOD_DIFFERENCE, NOISE_END)
    assert abs(pick.index - ONSET) <= 12
    (pick,) = ap.auto_pick([x], ap.METHOD_DIFFERENCE, NOISE_END, quiet=1)  # every dip ends the search: too late
    assert pick.index - ONSET > 30


def test_peak_and_noiseless_traces():
    x = trace(noise=0.0)
    (pick,) = ap.auto_pick([x], ap.METHOD_PEAK, NOISE_END)
    assert pick.index == int(np.argmax(x))
    pulse = np.exp(-0.5 * ((np.arange(N) - 3000) / 30.0) ** 2) + 0.01 * np.random.default_rng(1).standard_normal(N)
    (pick,) = ap.auto_pick([5.0 - pulse], ap.METHOD_PEAK, NOISE_END)   # a negative pulse on an offset: its trough
    assert abs(pick.index - 3000) <= 10 and pick.peak == pytest.approx(1.0, abs=0.05)
    (pick,) = ap.auto_pick([x], ap.METHOD_DIFFERENCE, NOISE_END)      # zero noise: still an onset, no division by 0
    assert pick.flag == "" and ONSET <= pick.index <= ONSET + 2


def test_robust_spread_ignores_outliers():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(5000)
    v[:20] = 100.0
    centre, spread = ap.robust_spread(v)
    assert abs(centre) < 0.1 and spread == pytest.approx(1.0, rel=0.1)


def test_a_search_window_picks_the_burst_wanted():
    x = trace(onset=3200) + trace(onset=2300, amp=3.0, seed=1, oscillate=True) * (np.arange(N) < 2600)
    for method in (ap.METHOD_DIFFERENCE, ap.METHOD_SLOPE, ap.METHOD_PEAK):
        (whole,) = ap.auto_pick([x], method, NOISE_END)
        assert whole.index < 2700                                         # the larger, earlier burst
        (pick,) = ap.auto_pick([x], method, NOISE_END, search=(3000, 3800))
        assert 3195 <= pick.index <= (3800 if method == ap.METHOD_PEAK else 3210) and pick.flag == ""
    # a window starting inside the rise: the onset is its start, flagged
    (pick,) = ap.auto_pick([trace()], ap.METHOD_DIFFERENCE, NOISE_END, search=(ONSET + 5, None))
    assert pick.index == ONSET + 5 and pick.flag == ap.FLAG_AT_SEARCH_START
    # a window before the noise's end shortens the noise; one with too little noise before it is refused
    (pick,) = ap.auto_pick([trace()], ap.METHOD_DIFFERENCE, 3500, search=(2400, None))
    assert abs(pick.index - ONSET) <= 6
    (pick,) = ap.auto_pick([trace()], ap.METHOD_DIFFERENCE, NOISE_END, search=(5, None))
    assert pick.index is None and pick.flag == ap.FLAG_NO_NOISE


def test_peak_flags_a_weak_largest_value():
    noise = 0.01 * np.random.default_rng(2).standard_normal(N)
    (pick,) = ap.auto_pick([noise], ap.METHOD_PEAK, NOISE_END)                 # nothing above 5 x the noise
    assert pick.index is None and pick.flag == ap.FLAG_WEAK and pick.peak < 5 * pick.noise
    (pick,) = ap.auto_pick([noise], ap.METHOD_PEAK, NOISE_END, threshold=2.0)
    assert pick.index is not None and pick.flag == ""
    assert ap.pick_peak(noise, NOISE_END, threshold=None).index is not None   # no check asked for
    assert ap.pick_peak(noise, 0).index is not None                            # no noise window: unchecked
