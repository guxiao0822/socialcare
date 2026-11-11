import pandas as pd

from .config import ID, KEYS, REF


def to_dt(s):
    """Parse UK-style (day-first) dates; unparseable values become NaT."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def sanitize(name) -> str:
    """Category label -> column-name suffix, e.g. 'Day Services' -> 'day_services'."""
    return (str(name).lower().strip().replace("&", "and").replace("/", "_")
            .replace("-", "_").replace(" ", "_").replace("__", "_"))


def history(landmarks, records, date_col):
    """Pair each landmark with the person's records dated on or before it."""
    m = landmarks.merge(records, on=ID, how="left")
    return m[m[date_col].notna() & (m[date_col] <= m[REF])].copy()


def days_until_landmark(h, start_col, end_col):
    """Episode duration counted up to the landmark (open or later end -> landmark)."""
    end = h[end_col].where(h[end_col].notna() & (h[end_col] <= h[REF]), h[REF])
    return (end - h[start_col]).dt.days.clip(lower=0)


def episode_lists(instances, records, cols):
    """Attach the person's full list of dates in `cols` to every landmark."""
    m = instances[KEYS].merge(records[[ID] + cols], on=ID, how="left")
    g = m.groupby(KEYS).agg({c: (lambda x: [d for d in x if pd.notna(d)]) for c in cols}).reset_index()
    out = instances.merge(g, on=KEYS, how="left")
    for c in cols:
        out[c] = out[c].apply(lambda v: v if isinstance(v, list) else [])
    return out
