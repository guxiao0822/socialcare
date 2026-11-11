"""Random synthetic data with the same sheets and columns as the source extract.

Nothing here comes from real records. People, dates and free text are drawn
from simple distributions driven by a latent "frailty" score, so that the
outcomes are somewhat predictable. The data are only meant to exercise the
pipeline; they say nothing about the real cohort.
"""

import numpy as np
import pandas as pd

from .config import (CARE_ASSESSMENTS, DATA_CUTOFF_DATE, HEALTH_CONDITIONS, HOME_FIRST_FORMS,
                     OT_ASSESSMENTS, SERVICE_GROUPS)

FIRST_RECORD_START = pd.Timestamp("2004-01-01")
FIRST_RECORD_END = pd.Timestamp("2022-12-31")
REABLEMENT_TYPES = HOME_FIRST_FORMS + ["Other Reablement Form"]
OTHER_CONDITION_TEXT = [
    "arthritis", "osteoporosis", "diabetes", "heart failure", "hypertension",
    "chronic kidney disease", "asthma", "depression", "anxiety",
    "falls", "frailty", "incontinence", "macular degeneration", "epilepsy",
    "multiple sclerosis", "obesity", "atrial fibrillation", "leg ulcer",
]
ETHNICITIES = ["White - British", "White - Any Other Background", "Asian - Any",
               "Black - Any", "Mixed - Any", "Not Known"]


def rand_dates(rng, start, end, n):
    """n uniform random dates in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = max((end - start).days, 1)
    return start + pd.to_timedelta(rng.integers(0, span, size=n), unit="D")


def maybe_end(rng, starts, mean_days, p_missing, cap):
    """Episode end dates (exponential duration), some left missing (ongoing)."""
    dur = rng.exponential(mean_days, size=len(starts)).astype(int) + 1
    ends = pd.Series(starts + pd.to_timedelta(dur, unit="D"))
    ends = ends.where(ends <= cap, pd.NaT)
    ends[rng.random(len(starts)) < p_missing] = pd.NaT
    return ends.values


def generate(n_people: int, seed: int):
    rng = np.random.default_rng(seed)
    tables = {k: [] for k in [
        "Person Demographics", "Care Assessment", "Care Plan - LT", "OT Equipment",
        "Reablement", "Services", "Hospital Admissions", "Health Conditions"]}

    for pid in range(1, n_people + 1):
        frailty = rng.normal()                       # latent need, drives everything
        first = rand_dates(rng, FIRST_RECORD_START, FIRST_RECORD_END, 1)[0]
        age_first = float(np.clip(rng.normal(72 + 6 * frailty, 14), 18.5, 100))

        # death: yearly hazard increasing with age and frailty
        death_date = pd.NaT
        t, age = first, age_first
        while t < DATA_CUTOFF_DATE:
            h = 1 / (1 + np.exp(-(-6.0 + 0.06 * (age - 60) + 0.8 * frailty)))
            if rng.random() < h:
                death_date = t + pd.Timedelta(days=int(rng.integers(0, 365)))
                break
            t += pd.DateOffset(years=1)
            age += 1
        if pd.notna(death_date) and death_date > DATA_CUTOFF_DATE:
            death_date = pd.NaT
        yod = death_date.year if pd.notna(death_date) else np.nan
        end = death_date if pd.notna(death_date) else DATA_CUTOFF_DATE
        anchor = min(pd.Timestamp(int(yod), 7, 1), DATA_CUTOFF_DATE) if pd.notna(yod) else DATA_CUTOFF_DATE
        current_age = int(age_first + (anchor - first).days / 365.25)
        years = max((end - first).days / 365.25, 0.1)

        tables["Person Demographics"].append({
            "Dummy LAS ID": pid,
            "Year of Death": yod,
            "Ethnicity Description": rng.choice(ETHNICITIES, p=[.8, .06, .04, .03, .02, .05]),
            "Gender Description": rng.choice(["Female", "Male", "Unknown"], p=[.58, .41, .01]),
            "Nationality Description": "Synthetic",
            "Religion Description": "Not provided",
            "Current Age in Years or Age at Death": current_age,
            "Language Description": "English",
            "Funding (current services)": rng.choice(["No current service", "Council Funded"]),
        })

        def n_events(base, beta=0.6):
            return rng.poisson(max(base * np.exp(beta * frailty) * years, 0.05))

        # Services
        k = n_events(0.5)
        if k:
            s = rand_dates(rng, first, end, k)
            for st, en, cls in zip(s, maybe_end(rng, s, 300, 0.2, end),
                                   rng.choice(SERVICE_GROUPS + ["Other Service"], size=k)):
                tables["Services"].append({"Dummy LAS ID": pid, "Service Group Class": cls,
                                           "Service Group": f"{cls} (synthetic)",
                                           "Start Date": st, "End Date": en})
        # Care assessments (the first record is often an assessment)
        k = n_events(0.4) + 1
        s = rand_dates(rng, first, end, k)
        s = s.insert(0, first)[:k]
        for st, typ in zip(s, rng.choice(CARE_ASSESSMENTS + ["Other Form"], size=k)):
            tables["Care Assessment"].append({
                "Dummy LAS ID": pid, "Person Form Completion or Cancellation Date and Time": st,
                "Person Form Type Description": typ})
        # Long-term care plans: non-overlapping sequence
        k = n_events(0.15)
        t = first
        for _ in range(k):
            gap = int(rng.exponential(900 * np.exp(-1.2 * frailty))) + 30
            st = t + pd.Timedelta(days=gap)
            if st >= end:
                break
            en = st + pd.Timedelta(days=int(rng.exponential(400)) + 30)
            ongoing = en >= end or rng.random() < 0.1
            tables["Care Plan - LT"].append({
                "Dummy LAS ID": pid, "Care Plan Effective From Date and Time": st,
                "Care Plan Effective To Date and Time": pd.NaT if ongoing else en,
                "Plan Status Description": "Active" if ongoing else "Ended",
                "Plan Type Description": "Long Term"})
            if ongoing:
                break
            t = en
        # OT equipment
        k = n_events(0.15)
        for st, typ in zip(rand_dates(rng, first, end, k), rng.choice(OT_ASSESSMENTS + ["Other OT Form"], size=k)):
            tables["OT Equipment"].append({
                "Dummy LAS ID": pid, "Person Form Type Description": typ,
                "Person Form Completion or Cancellation Date and Time": st,
                "Person Forms Answer (Ans1)": "Yes"})
        # Reablement
        k = n_events(0.1)
        for st, typ in zip(rand_dates(rng, first, end, k), rng.choice(REABLEMENT_TYPES, size=k)):
            tables["Reablement"].append({
                "Dummy LAS ID": pid, "Person Form Type Description": typ,
                "Person Form Completion or Cancellation Date and Time": st})
        # Hospital admissions (more frequent shortly before death)
        k = n_events(0.15, beta=1.5)
        adm = list(rand_dates(rng, first, end, k))
        if pd.notna(death_date) and rng.random() < 0.6:
            adm.append(death_date - pd.Timedelta(days=int(rng.integers(5, 300))))
        for a in adm:
            if a < first:
                continue
            los = int(rng.exponential(8))
            dis = a + pd.Timedelta(days=los)
            tables["Hospital Admissions"].append({
                "Dummy LAS ID": pid, "Admission Date": a,
                "Expected Discharge Date": a + pd.Timedelta(days=max(los - 1, 0)),
                "Actual Discharge Date": pd.NaT if (dis > DATA_CUTOFF_DATE or rng.random() < 0.02) else dis,
                "Hospital Name": rng.choice(["Hospital A", "Hospital B", "Hospital C"])})
        # Health conditions
        p = 1 / (1 + np.exp(-(-2.5 + 0.7 * frailty)))
        for cond in HEALTH_CONDITIONS:
            if rng.random() < p * (2.0 if cond in ("Other Condition", "Other Physical Health Condition") else 0.5):
                st = rand_dates(rng, first - pd.Timedelta(days=365), end, 1)[0]
                tables["Health Conditions"].append({
                    "Dummy LAS ID": pid, "Health Condition Start Date": st,
                    "Health Condition End Date": pd.NaT if rng.random() < 0.9 else st + pd.Timedelta(days=365),
                    "Condition Type Description": "Physical" if "Physical" in cond else "Other",
                    "Condition Description": cond,
                    "Other Condition": rng.choice(OTHER_CONDITION_TEXT) if cond == "Other Condition" else np.nan})

    return {k: pd.DataFrame(v) for k, v in tables.items()}


def random_codebook(conditions, dim=768, seed=0):
    """Stand-in for BiomedBERT embeddings: one random vector per text."""
    rng = np.random.default_rng(seed)
    return {"conditions": list(conditions), "embeddings": pd.DataFrame(rng.normal(size=(len(conditions), dim)))}
