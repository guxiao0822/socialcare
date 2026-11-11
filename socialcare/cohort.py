"""Cohort selection and landmark (prediction time) construction."""

import numpy as np
import pandas as pd

from .config import (CP_START, CURRENT_YEAR, DATA_CUTOFF_DATE, FORM_DATE, ID, KEYS,
                     LANDMARK_INTERVAL_MONTHS, REF)
from .utils import to_dt

# First record date of each person in each domain
FIRST_DATE_SOURCES = {
    "Services First Date": ("Services", "Start Date"),
    "Care Assessment First Date": ("Care Assessment", FORM_DATE),
    "Care Plan - LT First Date": ("Care Plan - LT", CP_START),
    "Hospital Admissions First Date": ("Hospital Admissions", "Admission Date"),
    "Health Conditions First Date": ("Health Conditions", "Health Condition Start Date"),
    "Reablement First Date": ("Reablement", FORM_DATE),
    "OT Equipment First Date": ("OT Equipment", FORM_DATE),
}


def approx_death_date(year_of_death):
    """Only the year of death is recorded; we use 1 July, capped at the cutoff."""
    y = pd.to_numeric(year_of_death, errors="coerce")
    if pd.isna(y) or not (1900 <= int(y) <= CURRENT_YEAR):
        return pd.NaT
    return min(pd.Timestamp(int(y), 7, 1), DATA_CUTOFF_DATE)


def build_cohort(sheets, verbose=True):
    """Apply the person-level exclusion criteria.

    Excludes people aged < 18 (now or at death, and at their first record) and
    people with no non-demographic record before the end of follow-up.
    """
    log = print if verbose else (lambda *a: None)
    demo = sheets["Person Demographics"].drop_duplicates(subset=ID, keep="first").copy()
    log(f"people in extract: {len(demo)}")

    for new_col, (sheet, col) in FIRST_DATE_SOURCES.items():
        t = sheets[sheet][[ID, col]].copy()
        t[col] = to_dt(t[col])
        first = t.groupby(ID, as_index=False)[col].min().rename(columns={col: new_col})
        demo = demo.merge(first, on=ID, how="left")
    demo["Earliest First Date"] = demo[list(FIRST_DATE_SOURCES)].min(axis=1)

    demo = demo[demo["Current Age in Years or Age at Death"] >= 18]
    log(f"  aged >= 18 now / at death: {len(demo)}")

    demo["Year of Death"] = pd.to_numeric(demo["Year of Death"], errors="coerce")
    demo["Approx Death Date"] = demo["Year of Death"].apply(approx_death_date)
    anchor = demo["Approx Death Date"].fillna(DATA_CUTOFF_DATE)
    age = demo["Current Age in Years or Age at Death"].astype(float)
    demo["Approx DOB"] = anchor - pd.to_timedelta(age * 365.25, unit="D")
    demo["Earliest Age"] = (demo["Earliest First Date"] - demo["Approx DOB"]).dt.days / 365.25
    demo = demo[demo["Earliest Age"].notna() & (demo["Earliest Age"] >= 18)]
    log(f"  aged >= 18 at first record: {len(demo)}")

    end_of_followup = demo["Approx Death Date"].fillna(DATA_CUTOFF_DATE)
    demo = demo[demo["Earliest First Date"].notna() & (demo["Earliest First Date"] < end_of_followup)]
    log(f"  with records before end of follow-up: {len(demo)}")
    return demo.copy()


def build_landmarks(demo, verbose=True):
    """One row per person and landmark, every 6 months from the first record
    until death or the data cutoff.

    Returns (landmarks, instances): `landmarks` holds only the keys,
    `instances` adds demographics, age at landmark and follow-up dates.
    """
    rows = []
    for _, r in demo.iterrows():
        end = r["Approx Death Date"] if pd.notna(r["Approx Death Date"]) else DATA_CUTOFF_DATE
        t = r["Earliest First Date"]
        while pd.notna(t) and t < end:
            rows.append((r[ID], t))
            t += pd.DateOffset(months=LANDMARK_INTERVAL_MONTHS)
    landmarks = pd.DataFrame(rows, columns=KEYS)

    inst = landmarks.merge(demo, on=ID, how="left")
    inst["Age"] = (inst[REF] - inst["Approx DOB"]).dt.days / 365.25
    inst = inst.sort_values(KEYS).reset_index(drop=True)
    inst["pred_idx"] = inst.groupby(ID).cumcount()
    inst["death_date"] = inst["Approx Death Date"]
    # Alive people are followed to the cutoff; death is an event, not censoring.
    inst["last_obs_date"] = DATA_CUTOFF_DATE
    if verbose:
        print(f"landmarks: {len(landmarks)} from {landmarks[ID].nunique()} people")
    return landmarks, inst
