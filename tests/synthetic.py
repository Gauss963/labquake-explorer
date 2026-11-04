"""Synthetic experiments with known answers for tests.

The mechanical data mimic a biaxial stick-slip run: shear stress rises
linearly with load-point displacement at a known stiffness, drops by a known
amount at each event, and the on-fault displacement jumps by a known slip.
Every generated quantity is returned so tests can assert against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StickSlipRun:
    time: np.ndarray
    shear_stress: np.ndarray
    displacement: np.ndarray
    LP_displacement: np.ndarray
    LP_velocity: np.ndarray
    normal_stress: np.ndarray
    event_indices: list[int]
    event_times: np.ndarray
    stiffness: float          # d(tau)/d(LP_displacement) while locked, MPa/um
    stress_drop: float        # MPa, same for every event
    slip: float               # um, same for every event
    lp_velocity: float        # um/s
    post_slope: float         # MPa/s, stress rate after each event (healing/relaxation)
    creep_fraction: float     # interseismic fault slip rate / lp_velocity
    extra: dict = field(default_factory=dict)

    @property
    def loading_rate(self) -> float:
        """d(tau)/dt while locked, MPa/s."""
        return self.stiffness * self.lp_velocity * (1.0 - self.creep_fraction)

    @property
    def stiffness_lp(self) -> float:
        """d(tau)/d(LP_displacement) while locked, MPa/um."""
        return self.stiffness * (1.0 - self.creep_fraction)

    @property
    def stiffness_fault(self) -> float:
        """d(tau)/d(displacement) while locked (creep only), MPa/um."""
        if self.creep_fraction == 0:
            return float("inf")
        return self.stiffness * (1.0 - self.creep_fraction) / self.creep_fraction

    def as_dict(self) -> dict:
        d = {
            "time": self.time,
            "shear_stress": self.shear_stress,
            "displacement": self.displacement,
            "LP_displacement": self.LP_displacement,
            "LP_velocity": self.LP_velocity,
            "normal_stress": self.normal_stress,
            "friction": self.shear_stress / self.normal_stress,
            "event_extraction": {"version": 1, "event_indices": list(self.event_indices)},
        }
        d.update(self.extra)
        return d


def make_stick_slip_run(
    n_events: int = 4,
    recurrence: float = 4.0,
    fs: float = 1000.0,
    stiffness: float = 0.01,
    stress_drop: float = 0.4,
    slip: float = 30.0,
    lp_velocity: float = 10.0,
    post_slope: float = 0.0,
    creep_fraction: float = 0.05,
    tau0: float = 5.0,
    normal_stress: float = 10.0,
    noise: float = 0.0,
    seed: int = 0,
    lead: float = 2.0,
) -> StickSlipRun:
    """Build a sawtooth stick-slip run.

    Between events the fault creeps at ``creep_fraction * lp_velocity`` while
    LP advances at ``lp_velocity``; shear stress follows
    ``tau0 + stiffness * (LP_displacement - displacement)`` so it rises at
    ``stiffness * lp_velocity * (1 - creep_fraction)`` per second.  At each
    event displacement jumps by ``slip`` and stress drops by ``stress_drop``
    in one sample (the drop is imposed, so it need not equal stiffness*slip).  ``post_slope`` adds a linear
    stress trend for ``recurrence/4`` seconds after each event so that the
    pre/post trends differ.
    """
    rng = np.random.default_rng(seed)
    duration = lead + n_events * recurrence + lead
    n = int(round(duration * fs)) + 1
    time = np.arange(n) / fs
    lp = lp_velocity * time
    event_times = lead + recurrence * np.arange(1, n_events + 1)
    event_indices = [int(round(t * fs)) for t in event_times]
    event_times = time[event_indices]

    disp = creep_fraction * lp_velocity * time
    tau = tau0 + stiffness * (lp - disp)
    for k, idx in enumerate(event_indices):
        tau[idx:] -= stress_drop
        disp[idx:] += slip
        if post_slope:
            span = int(round(recurrence / 4 * fs))
            seg = slice(idx, min(idx + span, n))
            t_rel = time[seg] - time[idx]
            tau[seg] += post_slope * t_rel
            # keep continuity after the segment ends
            if seg.stop < n:
                tau[seg.stop:] += post_slope * t_rel[-1]
    if noise:
        tau = tau + rng.normal(0.0, noise, n)
    sigma_n = np.full(n, normal_stress)
    return StickSlipRun(
        time=time,
        shear_stress=tau,
        displacement=disp,
        LP_displacement=lp,
        LP_velocity=np.full(n, lp_velocity),
        normal_stress=sigma_n,
        event_indices=event_indices,
        event_times=event_times,
        stiffness=stiffness,
        stress_drop=stress_drop,
        slip=slip,
        lp_velocity=lp_velocity,
        post_slope=post_slope,
        creep_fraction=creep_fraction,
    )


def extract_events(run: dict, window: float) -> list[dict]:
    """Slice a run dict into event dicts the way EventProcessor does (no strain)."""
    events = []
    time = run["time"]
    for idx in run["event_extraction"]["event_indices"]:
        event_time = time[idx]
        beg = int(np.argmin(np.abs(event_time - window - time)))
        end = int(np.argmin(np.abs(event_time + window - time)))
        event = {"event_time": event_time, "time": time[beg:end]}
        for key, value in run.items():
            if key in ("time", "events", "event_extraction"):
                continue
            if isinstance(value, np.ndarray) and value.shape == time.shape:
                event[key] = value[beg:end]
        events.append(event)
    return events


def make_experiment(name: str = "p0001", n_runs: int = 1, window: float = 5.0, **kwargs) -> tuple[dict, list[StickSlipRun]]:
    """Return (experiment dict, [StickSlipRun ...]) with events already extracted."""
    runs = []
    truth = []
    for i in range(n_runs):
        ssr = make_stick_slip_run(seed=i, **kwargs)
        run = ssr.as_dict()
        run["name"] = f"run{i:02d}"
        run["events"] = extract_events(run, window)
        run["event_extraction"].update({"start_s": -float(window), "end_s": float(window), "x_field": "time",
                                        "y_field": "shear_stress", "n_events": len(run["events"])})
        runs.append(run)
        truth.append(ssr)
    return {"name": name, "runs": runs}, truth


# ----------------------------------------------------------------------------
# strain-gauge array (for CZM fitter / arrival picker tests)
# ----------------------------------------------------------------------------
def add_strain(
    event: dict,
    n_channels: int = 16,
    fs: float = 2.0e5,
    half_window: float = 0.01,
    rupture_speed: float = 1000.0,
    rise_time: float = 2.0e-5,
    amplitude: float = 0.5,
    locations_mm: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> dict:
    """Attach a synthetic ``strain`` block to an event dict and return it.

    Each channel shows a smoothed step (tanh) arriving at
    ``event_time + (location - location[0]) / rupture_speed``; channel 14 has
    the opposite sign so it can stand in for Eyy.  Layout matches what
    EventProcessor produces: ``strain.original.time/raw`` at full rate and a
    100x downsampled ``strain.time/raw``.
    """
    rng = np.random.default_rng(seed)
    if locations_mm is None:
        locations_mm = 10.5 + 12.0 * (np.arange(n_channels) // 2)
    locations_mm = np.asarray(locations_mm, dtype=float)
    event_time = float(event["event_time"])
    n = int(round(2 * half_window * fs)) + 1
    tt = event_time - half_window + np.arange(n) / fs
    arrivals = event_time + (locations_mm - locations_mm[0]) * 1e-3 / rupture_speed
    raw = np.zeros((n_channels, n))
    for i in range(n_channels):
        sign = -1.0 if i == 14 else 1.0
        raw[i] = sign * amplitude * 0.5 * (1.0 + np.tanh((tt - arrivals[i]) / rise_time))
        if noise:
            raw[i] += rng.normal(0.0, noise, n)
    event["strain"] = {
        "filename": "synthetic.tpc5",
        "filename_downsampled": "",
        "time": tt[::100].copy(),
        "raw": raw[:, ::100].copy(),
        "original": {"time": tt, "raw": raw},
    }
    event["strain_truth"] = {
        "arrival_times": arrivals,
        "locations_mm": locations_mm,
        "rupture_speed": rupture_speed,
    }
    return event["strain"]


def make_experiment_with_strain(**kwargs) -> tuple[dict, list[StickSlipRun]]:
    data, truth = make_experiment(**kwargs)
    for run in data["runs"]:
        for event in run["events"]:
            add_strain(event)
    return data, truth


# ----------------------------------------------------------------------------
# Elsys TranAX tpc5 file (ECR dual mode) for reader tests
# ----------------------------------------------------------------------------
def write_tpc5(path, signals, fs_continuous: float = 2000.0, duration: float = 2.0,
               fs_trigger: float = 200_000.0, trigger_times=(0.5, 1.4), pre: float = 0.002,
               post: float = 0.008, volt_range: float = 0.512, start_time: str = "2026-01-01T00:00:00.0000000+08:00",
               marker_bits: int = 0) -> dict:
    """Write a dual-mode tpc5 file with one continuous block and trigger blocks.

    ``signals`` maps channel name -> callable ``f(t) -> volts`` evaluated on
    every block's own time axis, so the trigger blocks are consistent with the
    continuous record.  Block 1 is the continuous record (trigger sample at its
    last sample, as TranAX writes it); blocks 2.. are the trigger records.
    Returns the block table used (seconds on the file clock).
    """
    import h5py

    analog_mask = (0xFFFF >> marker_bits) if marker_bits else 0xFFFF
    marker_mask = 0xFFFF & ~analog_mask
    factor = 2 * volt_range / (analog_mask + 1)
    constant = -volt_range

    n_cont = int(round(duration * fs_continuous))
    blocks = [{"block": 1, "fs": fs_continuous, "n": n_cont, "trigger_sample": n_cont - 1,
               "trigger_time": (n_cont - 1) / fs_continuous}]
    n_pre, n_post = int(round(pre * fs_trigger)), int(round(post * fs_trigger))
    for k, tt in enumerate(trigger_times, start=2):
        blocks.append({"block": k, "fs": fs_trigger, "n": n_pre + n_post, "trigger_sample": n_pre,
                       "trigger_time": float(tt)})

    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = "TransAsData"
        f.attrs["creator"] = "synthetic"
        meas = f.create_group("/measurements/00000001")
        meas.attrs["name"] = "M1"
        channels = meas.create_group("channels")
        for number, (name, func) in enumerate(signals.items(), start=1):
            cg = channels.create_group(f"{number:08d}")
            cg.attrs["name"] = name
            cg.attrs["ChannelName"] = name
            cg.attrs["physicalUnit"] = "V"
            cg.attrs["binToVoltFactor"] = np.float64(factor)
            cg.attrs["binToVoltConstant"] = np.float64(constant)
            cg.attrs["analogMask"] = np.int32(analog_mask)
            cg.attrs["markerMask"] = np.int32(marker_mask)
            cg.attrs["rangeMin"] = np.float64(constant)
            cg.attrs["rangeMax"] = np.float64(constant + factor * analog_mask)
            cg.attrs["voltToPhysicalFactor"] = np.float64(1.0)
            cg.attrs["voltToPhysicalConstant"] = np.float64(0.0)
            bgs = cg.create_group("blocks")
            for b in blocks:
                bg = bgs.create_group(f"{b['block']:08d}")
                bg.attrs["sampleRateHertz"] = np.float64(b["fs"])
                bg.attrs["triggerSample"] = np.int64(b["trigger_sample"])
                bg.attrs["triggerTimeSeconds"] = np.float64(b["trigger_time"])
                bg.attrs["startTime"] = start_time
                bg.attrs["relativeDivisor"] = np.int32(128)
                t = b["trigger_time"] + (np.arange(b["n"]) - b["trigger_sample"]) / b["fs"]
                volts = np.asarray(func(t), dtype=float)
                raw = np.clip(np.round((volts - constant) / factor), 0, analog_mask).astype(np.uint16)
                if marker_bits:
                    raw = raw | np.uint16(marker_mask)   # marker bits set on every sample
                bg.create_dataset("raw", data=raw, chunks=(min(32768, b["n"]),))
    return {"blocks": blocks, "factor": factor, "constant": constant}


# ----------------------------------------------------------------------------
# National Instruments .npz recording for reader tests
# ----------------------------------------------------------------------------
def write_ni_npz(path, signals, fs: float = 10_000.0, duration: float = 2.0,
                 trigger_sample_index=None, compressed: bool = False, dtype=np.float64) -> dict:
    """Write an NI-style npz: ``aiN.npy`` per channel plus ``sample_rate``,
    ``channels`` and ``trigger_sample_index``.  ``signals`` maps a label to a
    callable ``f(t) -> volts``.  Returns the arrays written (for assertions)."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    arrays = {f"ai{i}": np.asarray(func(t), dtype=dtype) for i, func in enumerate(signals.values())}
    meta = {"sample_rate": np.array(fs), "channels": np.array(list(signals.keys())),
            "trigger_sample_index": np.array(-1 if trigger_sample_index is None else int(trigger_sample_index))}
    if trigger_sample_index is None:
        meta.pop("trigger_sample_index")
    (np.savez_compressed if compressed else np.savez)(path, **arrays, **meta)
    return {"t": t, **arrays}


# ----------------------------------------------------------------------------
# slip sensor array along the fault (for the coseismic slip analysis)
# ----------------------------------------------------------------------------
SLIP_X_MM = (31.0, 93.0, 155.0, 217.0, 279.0, 341.0, 403.0, 465.0)


def add_slip_array(event: dict, x_mm=SLIP_X_MM, x0_mm: float = 279.0, t0_s: float = 0.02,
                   front_speed_m_s: float = 50.0, slips_um=None, rise_s: float = 0.002,
                   creep_um_s: float = 3.0, noise_um=0.05, seed: int = 0) -> dict:
    """Give an event a ``slip`` channel array of sensors along the fault.

    The rupture starts at ``x0_mm`` at ``event_time + t0_s`` and reaches each
    sensor at ``t_i = event_time + t0_s + |x - x0| / front_speed``; there the
    slip steps up by ``slips_um[i]`` along a tanh of rise time ``rise_s`` on top
    of a steady creep ``creep_um_s`` (the same rate before and after, so two
    lines extrapolated to ``t_i`` differ by exactly the step).  Gaussian noise
    of ``noise_um`` (one value, or one per sensor) is added.  Returns the
    truth: ``t_i`` (absolute), ``slips_um``, ``noise_um``, ``x_mm``.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(event["time"], dtype=float)
    te = float(event["event_time"])
    x = np.asarray(x_mm, dtype=float)
    if slips_um is None:
        slips_um = np.maximum(0.0, 100.0 * (1.0 - ((x - x0_mm) / 300.0) ** 2))
    slips = np.asarray(slips_um, dtype=float)
    noise = np.broadcast_to(np.asarray(noise_um, dtype=float), x.shape)
    t_i = te + t0_s + np.abs(x - x0_mm) * 1e-3 / front_speed_m_s
    data = np.empty((x.size, t.size))
    for i in range(x.size):
        step = 0.5 * slips[i] * (1.0 + np.tanh((t - t_i[i]) / rise_s))
        data[i] = creep_um_s * (t - t[0]) + step + rng.normal(0.0, noise[i], t.size)
    names = [f"slip_{i + 1}" for i in range(x.size)]
    event["slip"] = {
        "data": data.astype(np.float64),
        "channels": names,
        "unit": "um",
        "positions": {"x": x.tolist(), "y": [0.0] * x.size, "z": [50.0] * x.size, "unit": "mm", "frame": "test"},
        "source": [f"eddy_{i + 1}" for i in range(x.size)],
    }
    return {"t_i": t_i, "slips_um": slips, "noise_um": noise.copy(), "x_mm": x}
