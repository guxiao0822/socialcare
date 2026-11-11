"""Fixed-horizon outcome labels.

For each landmark t0 and horizon H:

  mortality-binary-{H}yr  1 if death in (t0, t0+H], 0 if alive at t0+H.
  cp-binary-{H}yr         1 if a new long-term care plan starts in (t0, t0+H].
  ha-binary-{H}yr         1 if an admission (stay > 3 days) starts in (t0, t0+H].

For care plans and admissions, death before the event is treated as censored,
and landmarks inside an active care plan / admission are not used. Labels are
NaN (excluded) when follow-up does not reach t0+H and no event occurred.
"""

import numpy as np
import pandas as pd

from .config import CP_END, CP_START, DATA_CUTOFF_DATE, HORIZONS_DAYS, KEYS, MIN_ADMISSION_LOS, REF
from .utils import episode_lists, to_dt


def first_event_after(t0, dates):
    later = [d for d in dates if pd.notna(d) and d > t0]
    return min(later) if later else pd.NaT


def binary_label(t0, event_dates, last_obs, horizon_days):
    end = t0 + pd.Timedelta(days=horizon_days)
    t = first_event_after(t0, event_dates)
    if pd.notna(t) and t <= end:
        return 1
    if pd.notna(last_obs) and last_obs >= end:
        return 0
    return np.nan


def competing_label(t0, event_dates, death, last_obs, horizon_days):
    """0 = neither, 1 = event first, 2 = death first, NaN = follow-up too short."""
    end = t0 + pd.Timedelta(days=horizon_days)
    t = first_event_after(t0, event_dates)
    event_in = pd.notna(t) and t <= end
    death_in = pd.notna(death) and t0 < death <= end
    if death_in and (not event_in or death < t):
        return 2
    if event_in and (not death_in or death >= t):
        return 1
    if pd.isna(last_obs) or last_obs < end:
        return np.nan
    return 0


def _to_binary(c):
    # death before the event (class 2) is censored
    return 1 if c == 1 else (0 if c == 0 else np.nan)


def _active_at(t0, starts, ends):
    return any(s <= t0 <= e for s, e in zip(starts, ends))


def _add_competing_targets(out, prefix, t0s, event_lists, deaths, last_obs, usable):
    out[f"{prefix}-Reference Used for Prediction"] = usable.astype(int).values
    for w in HORIZONS_DAYS:
        h = w // 365
        cls = pd.Series([competing_label(t, ev, d, lo, w)
                         for t, ev, d, lo in zip(t0s, event_lists, deaths, last_obs)])
        cls[~usable.values] = np.nan
        out[f"{prefix}-class-{h}yr"] = cls.values
        out[f"{prefix}-binary-{h}yr"] = cls.apply(_to_binary).values


def build_targets(sheets, instances, verbose=True):
    base = instances[KEYS + ["death_date", "last_obs_date"]]
    out = base[KEYS].copy()

    # Care plan initiation (an open-ended plan is treated as ongoing until the cutoff)
    cp = sheets["Care Plan - LT"].copy()
    cp[CP_START] = to_dt(cp[CP_START])
    cp[CP_END] = to_dt(cp[CP_END]).fillna(DATA_CUTOFF_DATE)
    x = episode_lists(base, cp, [CP_START, CP_END])
    usable = ~x.apply(lambda r: _active_at(r[REF], r[CP_START], r[CP_END]), axis=1)
    _add_competing_targets(out, "cp", x[REF], x[CP_START], x["death_date"], x["last_obs_date"], usable)

    # Hospital admission
    ha = sheets["Hospital Admissions"].copy()
    ha["Admission Date"] = to_dt(ha["Admission Date"])
    ha["Actual Discharge Date"] = to_dt(ha["Actual Discharge Date"]).fillna(DATA_CUTOFF_DATE)
    x = episode_lists(base, ha, ["Admission Date", "Actual Discharge Date"])
    usable = ~x.apply(lambda r: _active_at(r[REF], r["Admission Date"], r["Actual Discharge Date"]), axis=1)
    admissions = [[a for a, d in zip(A, D) if d - a > MIN_ADMISSION_LOS]
                  for A, D in zip(x["Admission Date"], x["Actual Discharge Date"])]
    _add_competing_targets(out, "ha", x[REF], admissions, x["death_date"], x["last_obs_date"], usable)

    # Mortality
    for w in HORIZONS_DAYS:
        out[f"mortality-binary-{w // 365}yr"] = [
            binary_label(t, [d] if pd.notna(d) else [], lo, w)
            for t, d, lo in zip(base[REF], base["death_date"], base["last_obs_date"])]

    if verbose:
        print("targets (n labelled / n events):")
        for c in sorted(c for c in out.columns if "binary" in c):
            print(f"  {c:22s} {int(out[c].notna().sum()):7d} / {int(np.nansum(out[c])):6d}")
    return out
