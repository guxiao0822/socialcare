"""Predictors at each landmark, built only from records dated on or before it.

  demographics        age, gender                                     2
  services            12 groups x (count, duration sum/mean/sd, active) 60
  hospital admissions count, LOS sum/mean/sd, active                   5
  health conditions   20 conditions x (count, duration x3, active)    100
  other condition     mean BiomedBERT embedding (PCA'd to 10 later)   10
  care assessments    count per type                                  13
  care plans          count, duration sum/mean/sd, active              5
  OT equipment        count per assessment type                        2
  reablement          number of Home First forms                       1
                                                                     ---
                                                                     198
"""

import numpy as np
import pandas as pd

from . import config as C
from .config import FORM_DATE, FORM_TYPE, ID, KEYS, REF
from .utils import days_until_landmark, history, sanitize, to_dt


def _episode_summary(h, prefix, group_col=None):
    """count / duration sum, mean, sd / active-at-landmark, optionally per category."""
    agg = (h.groupby(KEYS + ([group_col] if group_col else []))
             .agg(**{f"{prefix}_count": ("dur", "size"),
                     f"{prefix}_dur_sum_days": ("dur", "sum"),
                     f"{prefix}_dur_mean_days": ("dur", "mean"),
                     f"{prefix}_dur_std_days": ("dur", "std"),
                     f"{prefix}_active": ("active", "max")})
             .reset_index())
    if group_col is None:
        return agg
    values = [c for c in agg.columns if c.startswith(prefix)]
    wide = agg.pivot_table(index=KEYS, columns=group_col, values=values, fill_value=0, aggfunc="first")
    wide.columns = [f"{m}__{sanitize(c)}" for m, c in wide.columns]
    return wide.reset_index()


def _type_counts(h, prefix, expected=()):
    wide = (h.groupby(KEYS + [FORM_TYPE]).size().reset_index(name="n")
              .pivot_table(index=KEYS, columns=FORM_TYPE, values="n", fill_value=0))
    wide.columns = [f"{prefix}__{sanitize(c)}" for c in wide.columns]
    wide = wide.reset_index()
    for c in expected:
        if c not in wide.columns:
            wide[c] = 0
    return wide


def _forms(sheet):
    df = sheet.copy()
    df[FORM_DATE] = to_dt(df[FORM_DATE])
    df[FORM_TYPE] = df[FORM_TYPE].astype(str).str.strip()
    return df[[ID, FORM_DATE, FORM_TYPE]]


def services(sheets, landmarks):
    s = sheets["Services"].copy()
    s["Start Date"], s["End Date"] = to_dt(s["Start Date"]), to_dt(s["End Date"])
    s = s[s["Service Group Class"].isin(C.SERVICE_GROUPS)]
    h = history(landmarks, s, "Start Date")
    h["dur"] = days_until_landmark(h, "Start Date", "End Date")
    h["active"] = (h["End Date"].isna() | (h["End Date"] >= h[REF])).astype(int)
    # Left as NaN when a person has no service history; imputed (median) in training.
    return _episode_summary(h, "svc", "Service Group Class"), False


def hospital_admissions(sheets, landmarks):
    ha = sheets["Hospital Admissions"].copy()
    ha["Admission Date"] = to_dt(ha["Admission Date"])
    ha["Actual Discharge Date"] = to_dt(ha["Actual Discharge Date"])
    h = history(landmarks, ha, "Admission Date")
    h["active"] = (h["Actual Discharge Date"].isna() | (h["Actual Discharge Date"] >= h[REF])).astype(int)
    h["dur"] = days_until_landmark(h, "Admission Date", "Actual Discharge Date")
    h = h[h["dur"] > C.MIN_ADMISSION_LOS.days]
    f = _episode_summary(h, "ha")
    f.columns = [c.replace("ha_dur_", "ha_los_") for c in f.columns]
    return f, True


def health_conditions(sheets, landmarks):
    hc = sheets["Health Conditions"].copy()
    hc["Health Condition Start Date"] = to_dt(hc["Health Condition Start Date"])
    hc["Health Condition End Date"] = to_dt(hc["Health Condition End Date"])
    hc = hc[hc[ID].notna() & hc["Condition Description"].isin(C.HEALTH_CONDITIONS)]
    h = history(landmarks, hc, "Health Condition Start Date")
    h["dur"] = days_until_landmark(h, "Health Condition Start Date", "Health Condition End Date")
    h["active"] = (h["Health Condition End Date"].isna() | (h["Health Condition End Date"] >= h[REF])).astype(int)
    return _episode_summary(h, "hc", "Condition Description"), True


def other_condition_embedding(sheets, landmarks, codebook):
    """Mean embedding of all free-text 'Other Condition' entries recorded so far."""
    hc = sheets["Health Conditions"].copy()
    hc["Health Condition Start Date"] = to_dt(hc["Health Condition Start Date"])
    text = hc["Other Condition"].astype(str).str.strip()
    hc["Other Condition"] = text.where(~text.isin(["", "nan", "NaN", "None"]), np.nan)
    h = history(landmarks, hc[[ID, "Other Condition", "Health Condition Start Date"]], "Health Condition Start Date")
    h = h[h["Other Condition"].notna()]
    dim = len(next(iter(codebook.values())))

    def mean_embedding(texts):
        vecs = [codebook[t] for t in texts.astype(str) if t in codebook]
        return np.mean(np.stack(vecs), axis=0) if vecs else np.full(dim, np.nan)

    f = h.groupby(KEYS)["Other Condition"].apply(mean_embedding).rename(C.EMB_COL).reset_index()
    return f, False


def care_assessments(sheets, landmarks):
    h = history(landmarks, _forms(sheets["Care Assessment"]), FORM_DATE)
    h = h[h[FORM_TYPE].isin(C.CARE_ASSESSMENTS)]
    # Left as NaN when there is no prior assessment; imputed (median) in training.
    return _type_counts(h, "ca_count"), False


def care_plans(sheets, landmarks):
    cp = sheets["Care Plan - LT"].copy()
    cp[C.CP_START], cp[C.CP_END] = to_dt(cp[C.CP_START]), to_dt(cp[C.CP_END])
    h = history(landmarks, cp[[ID, C.CP_START, C.CP_END]], C.CP_START)
    h["dur"] = days_until_landmark(h, C.CP_START, C.CP_END)
    h["active"] = (h[C.CP_END].isna() | (h[C.CP_END] >= h[REF])).astype(int)
    return _episode_summary(h, "cp"), True


def ot_equipment(sheets, landmarks):
    h = history(landmarks, _forms(sheets["OT Equipment"]), FORM_DATE)
    h = h[h[FORM_TYPE].isin(C.OT_ASSESSMENTS)]
    expected = [f"ot_count__{sanitize(c)}" for c in C.OT_ASSESSMENTS]
    return _type_counts(h, "ot_count", expected), True


def reablement(sheets, landmarks):
    h = history(landmarks, _forms(sheets["Reablement"]), FORM_DATE)
    h["re_home_first_count"] = h[FORM_TYPE].isin(C.HOME_FIRST_FORMS).astype(int)
    return h.groupby(KEYS)["re_home_first_count"].sum().reset_index(), True


def build_features(sheets, landmarks, instances, codebook):
    """Add all predictor columns to `instances` (one row per landmark)."""
    df = instances
    blocks = [services, hospital_admissions, health_conditions, care_assessments,
              care_plans, ot_equipment, reablement]
    for block in blocks:
        f, fill_zero = block(sheets, landmarks)
        df = df.merge(f, on=KEYS, how="left")
        if fill_zero:
            cols = [c for c in f.columns if c not in KEYS]
            df[cols] = df[cols].fillna(0)
        if block is health_conditions:
            f, _ = other_condition_embedding(sheets, landmarks, codebook)
            df = df.merge(f, on=KEYS, how="left")
    df["Gender"] = df["Gender Description"].map(C.GENDER_CODES)
    return df
