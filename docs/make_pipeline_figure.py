"""Draw docs/pipeline.pdf and docs/pipeline.png.

The figure follows the Nature Portfolio artwork guidelines: 183 mm wide
(double column), Arial, 5-7 pt text, 8 pt bold lowercase panel labels,
editable text in the PDF. Outcome colours match those used in the paper.

It uses one person from the synthetic data, so run the demo first:

    bash run_demo.sh
    python docs/make_pipeline_figure.py
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

from socialcare.config import (CARE_ASSESSMENTS, CP_END, CP_START, DATA_CUTOFF_DATE, EMB_COL,
                               FORM_DATE, FORM_TYPE, HEALTH_CONDITIONS, HOME_FIRST_FORMS, ID,
                               MIN_ADMISSION_LOS, OT_ASSESSMENTS, REF, SERVICE_GROUPS, SHEETS)

MM = 1 / 25.4
FIG_W, FIG_H = 183, 134          # mm

# colours: outcomes as in the paper; everything else grey
CP_COL, HA_COL, DEATH_COL = "#8ECCC7", "#69A0F1", "#EC776C"
INK = "#000000"
DARK = "#4d4d4d"                 # records before the landmark
LIGHT = "#cccccc"                # records after the landmark
BAND = "#efefef"                 # observation window
EDGE = "#3a3a3a"                 # outline on coloured marks
MISSING = "#e3e3e3"
VALUE_CMAP = LinearSegmentedColormap.from_list("value", ["#ffffff", "#262626"])

LABEL_X = 31                     # right edge of row labels (mm)
PLOT_X, PLOT_W = 33, 100         # shared time axis of panels a and b (mm)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "xtick.major.size": 2,
    "xtick.labelsize": 6,
    "pdf.fonttype": 42,          # editable TrueType text
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
})


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def panel_label(fig, letter, x, y):
    fig.text(x / FIG_W, y / FIG_H, letter, fontsize=8, fontweight="bold", va="top", ha="left")


def years(d, origin):
    return (pd.Timestamp(d) - origin).days / 365.25


# ---------------------------------------------------------------------------
# example person
# ---------------------------------------------------------------------------
DOMAIN_SHEETS = ["Services", "Care Assessment", "Care Plan - LT", "OT Equipment",
                 "Reablement", "Hospital Admissions", "Health Conditions"]


def used_records(sheets):
    """Keep only the record types that feed into the predictors."""
    keep = {
        "Services": lambda d: d["Service Group Class"].isin(SERVICE_GROUPS),
        "Care Assessment": lambda d: d[FORM_TYPE].isin(CARE_ASSESSMENTS),
        "OT Equipment": lambda d: d[FORM_TYPE].isin(OT_ASSESSMENTS),
        "Reablement": lambda d: d[FORM_TYPE].isin(HOME_FIRST_FORMS),
        "Health Conditions": lambda d: d["Condition Description"].isin(HEALTH_CONDITIONS),
    }
    return {name: d[keep[name](d)] if name in keep else d for name, d in sheets.items()}


def _next_event(dates, t0):
    later = [d for d in dates if d > t0]
    return min(later) if later else pd.NaT


def pick_example(sheets, df):
    """Choose a person and landmark that show every part of the figure: records
    in all domains, and a care plan and an admission starting 2 months to 3
    years after the landmark. Prefer someone who also dies 1-3 years after it."""
    counts = pd.DataFrame({s: sheets[s].groupby(ID).size() for s in DOMAIN_SHEETS}).fillna(0)
    people = counts.index[(counts > 0).all(axis=1)]
    has_text = set(sheets["Health Conditions"].dropna(subset=["Other Condition"])[ID])
    all_ha = sheets["Hospital Admissions"]
    ha = all_ha[all_ha["Actual Discharge Date"].fillna(DATA_CUTOFF_DATE) - all_ha["Admission Date"] > MIN_ADMISSION_LOS]
    cp = sheets["Care Plan - LT"]

    def clear_of_t0(pid, t0):
        gaps = [_next_event(ha.loc[ha[ID] == pid, "Admission Date"], t0) - t0,
                _next_event(cp.loc[cp[ID] == pid, CP_START], t0) - t0]
        return all(pd.notna(g) and g.days >= 60 for g in gaps)

    # an admission without a discharge date is drawn as a stay up to the cutoff; avoid those
    open_stay = set(all_ha.loc[all_ha["Actual Discharge Date"].isna(), ID])

    best = None
    for pid in people:
        rows = df[df[ID] == pid].reset_index(drop=True)
        if not 8 <= len(rows) <= 22 or pid in open_stay:
            continue
        for k in range(4, len(rows)):
            r = rows.loc[k]
            if r["cp-binary-3yr"] == 1 and r["ha-binary-3yr"] == 1 and clear_of_t0(pid, r[REF]):
                dies = r["mortality-binary-1yr"] == 0 and r["mortality-binary-3yr"] == 1
                score = (dies, pid in has_text, counts.loc[pid].sum(), -k)
                if best is None or score > best[0]:
                    best = (score, pid, r[REF])
    if best is None:
        raise RuntimeError("no suitable example person in the synthetic data")
    return best[1], best[2]


# ---------------------------------------------------------------------------
# panel a: the record over time
# ---------------------------------------------------------------------------
ROWS = ["Services", "Care assessments", "OT equipment", "Reablement", "Health conditions",
        "Care plans", "Hospital admissions", "Death", "Landmarks"]


def draw_records(ax, sheets, person, t0, origin, end):
    """Episodes as bars and single events as dots, dark before t0 and light after.
    The first care plan, admission and death after t0 are shown in colour."""
    row_y = {name: len(ROWS) - 1 - i for i, name in enumerate(ROWS)}
    x0 = years(t0, origin)
    h = 0.36

    def bar(row, s, e, colour=None):
        s, e = max(s, 0), min(e, years(end, origin))
        if e <= s:
            return
        y = row_y[row] - h / 2
        if colour:                                   # outcome episode
            ax.add_patch(Rectangle((s, y), e - s, h, fc=colour, ec=EDGE, lw=0.3, zorder=4))
            return
        if s < x0:
            ax.add_patch(Rectangle((s, y), min(e, x0) - s, h, fc=DARK, lw=0, zorder=3))
        if e > x0:
            ax.add_patch(Rectangle((max(s, x0), y), e - max(s, x0), h, fc=LIGHT, lw=0, zorder=3))

    def dot(row, x, colour=None, hollow=False, size=2.6):
        if colour:
            ax.plot(x, row_y[row], "o", ms=3.6, mfc=colour, mec=EDGE, mew=0.4, zorder=6)
        elif hollow:
            ax.plot(x, row_y[row], "o", ms=size, mfc="white", mec=DARK if x < x0 else "#a6a6a6",
                    mew=0.5, zorder=5)
        else:
            ax.plot(x, row_y[row], "o", ms=size, mfc=DARK if x < x0 else LIGHT, mec="none", zorder=5)

    def records(sheet, start):
        r = sheets[sheet][sheets[sheet][ID] == person].copy()
        r = r[r[start].notna() & (r[start] <= end)]
        return r.sort_values(start)

    horizon_end = t0 + pd.DateOffset(years=3)

    for _, r in records("Services", "Start Date").iterrows():
        e = r["End Date"] if pd.notna(r["End Date"]) else end
        bar("Services", years(r["Start Date"], origin), years(e, origin))
    for sheet, row in [("Care Assessment", "Care assessments"), ("OT Equipment", "OT equipment"),
                       ("Reablement", "Reablement")]:
        for d in records(sheet, FORM_DATE)[FORM_DATE]:
            dot(row, years(d, origin))
    for _, r in records("Health Conditions", "Health Condition Start Date").iterrows():
        e = r["Health Condition End Date"] if pd.notna(r["Health Condition End Date"]) else end
        bar("Health conditions", years(r["Health Condition Start Date"], origin), years(e, origin))

    # care plans: the first plan starting after t0 is the outcome event
    first_cp = True
    for _, r in records("Care Plan - LT", CP_START).iterrows():
        s, e = r[CP_START], r[CP_END] if pd.notna(r[CP_END]) else end
        is_event = first_cp and t0 < s <= horizon_end
        if s > t0:
            first_cp = False
        bar("Care plans", years(s, origin), years(e, origin), CP_COL if is_event else None)

    # admissions: stays of 3 days or less are not counted (open circles)
    first_ha = True
    for _, r in records("Hospital Admissions", "Admission Date").iterrows():
        a = r["Admission Date"]
        d = r["Actual Discharge Date"] if pd.notna(r["Actual Discharge Date"]) else DATA_CUTOFF_DATE
        counted = d - a > MIN_ADMISSION_LOS
        x = years(a, origin)
        if counted and first_ha and t0 < a <= horizon_end:
            dot("Hospital admissions", x, colour=HA_COL)
        elif not counted:
            dot("Hospital admissions", x, hollow=True)
        elif (d - a).days > 30:
            bar("Hospital admissions", x, years(d, origin))
        else:
            dot("Hospital admissions", x)
        if counted and a > t0:
            first_ha = False

    return row_y


def panel_a(fig, sheets, df, person, t0):
    rows = df[df[ID] == person].sort_values(REF)
    origin = rows[REF].min()
    death = rows["death_date"].iloc[0]
    end = death if pd.notna(death) else DATA_CUTOFF_DATE
    x0 = years(t0, origin)
    xmax = max(years(end, origin), x0 + 3) + 0.35

    ax = mm_axes(fig, PLOT_X, 79, PLOT_W, 46)
    ax.set_xlim(-0.35, xmax)
    ax.set_ylim(-0.6, 12.2)
    ax.axis("off")

    row_y = draw_records(ax, sheets, person, t0, origin, end)
    if pd.notna(death):
        ax.plot(years(death, origin), row_y["Death"], "o", ms=3.6, mfc=DEATH_COL, mec=EDGE, mew=0.4, zorder=6)

    # observation window
    ax.add_patch(Rectangle((0, row_y["Death"] - 0.5), x0, len(ROWS) - 1, fc=BAND, lw=0, zorder=0))

    # landmarks
    for x in (rows[REF].apply(lambda d: years(d, origin))):
        is_t0 = abs(x - x0) < 1e-6
        ax.plot([x, x], [-0.28, 0.28], color=INK if is_t0 else "#8c8c8c", lw=0.9 if is_t0 else 0.5,
                solid_capstyle="butt", zorder=3)

    # t0 and prediction horizons
    ax.plot([x0, x0], [-0.45, 11.0], color=INK, lw=0.6, zorder=7, solid_capstyle="butt")
    ax.text(x0, 11.25, "Landmark $t_0$", ha="center", va="bottom")
    for i, h in enumerate([1, 2, 3]):
        y = 8.95 + 0.6 * i
        ax.plot([x0, x0 + h], [y, y], color=INK, lw=0.5, solid_capstyle="butt")
        ax.plot([x0 + h, x0 + h], [y - 0.18, y + 0.18], color=INK, lw=0.5)
        ax.text(x0 + h + 0.07, y, f"{h} year" + ("s" if h > 1 else ""), va="center", ha="left")
    ax.text(x0 + 0.08, 10.65, "Prediction horizons (outcomes)", ha="left", va="center")
    ax.text(x0 / 2, 9.55, "Observation window (predictors)", ha="center", va="center")

    trans = ax.get_yaxis_transform()
    label_x = (LABEL_X - PLOT_X) / PLOT_W
    for name, y in row_y.items():
        ax.text(label_x, y, name, ha="right", va="center", transform=trans)

    panel_label(fig, "a", 2, FIG_H - 2)
    return ax, origin, xmax


# ---------------------------------------------------------------------------
# panel b: the same person's landmarks as model input
# ---------------------------------------------------------------------------
def predictor_blocks(df):
    cols = df.columns
    hc = [c for c in cols if c.startswith(("hc_active__", "hc_count__", "hc_dur_"))]
    return [
        ("Age, gender", ["Age", "Gender"]),
        ("Services", [c for c in cols if c.startswith(("svc_active__", "svc_count__", "svc_dur_"))]),
        ("Hospital admissions", ["ha_count", "ha_los_sum_days", "ha_los_mean_days", "ha_los_std_days", "ha_active"]),
        ("Health conditions", hc),
        ("Free text", [EMB_COL]),
        ("Care assessments", [c for c in cols if c.startswith("ca_count__")]),
        ("Care plans", ["cp_count", "cp_dur_sum_days", "cp_dur_mean_days", "cp_dur_std_days", "cp_active"]),
        ("OT equipment", [c for c in cols if c.startswith("ot_count__")]),
        ("Reablement", [c for c in cols if c.startswith("re_")]),
    ]


def display_values(df, cols, person_rows):
    """Scale each predictor to [0, 1] across all landmarks, for display only."""
    if cols == [EMB_COL]:
        has = df[EMB_COL].apply(lambda v: isinstance(v, np.ndarray))
        pca = PCA(n_components=10, random_state=0).fit(np.stack(df.loc[has, EMB_COL].values))
        z_all = pca.transform(np.stack(df.loc[has, EMB_COL].values))
        lo, hi = z_all.min(axis=0), z_all.max(axis=0)
        out = np.zeros((10, len(person_rows)))
        for j, v in enumerate(person_rows[EMB_COL]):
            if isinstance(v, np.ndarray):
                out[:, j] = (pca.transform(v[None])[0] - lo) / (hi - lo)
        return out
    out = []
    for c in cols:
        v = person_rows[c].astype(float).fillna(0).values
        if c == "Age":
            lo, hi = df[c].min(), df[c].max()
            out.append((v - lo) / (hi - lo))
        else:
            pos = df[c][df[c] > 0]
            scale = pos.quantile(0.99) if len(pos) else 1.0
            out.append(np.clip(v / scale, 0, 1))
    return np.array(out)


OUTCOMES = [("Care plan initiation", "cp", CP_COL), ("Hospital admission", "ha", HA_COL),
            ("Death", "mortality", DEATH_COL)]


def panel_b(fig, df, person, t0, origin, xmax):
    rows = df[df[ID] == person].sort_values(REF)
    xs = rows[REF].apply(lambda d: years(d, origin)).values
    half = 0.25
    left, right = xs[0] - half, xs[-1] + half

    ax = mm_axes(fig, PLOT_X, 13, PLOT_W, 61)
    ax.set_xlim(-0.35, xmax)
    ax.set_ylim(0, 61)
    trans = ax.get_yaxis_transform()
    label_x = (LABEL_X - PLOT_X) / PLOT_W

    # outcome labels (bottom): 9 rows
    cell_h, group_gap = 2.2, 1.0
    y = 22.4
    for name, prefix, colour in OUTCOMES:
        top = y
        for hz in (1, 2, 3):
            for x, v in zip(xs, rows[f"{prefix}-binary-{hz}yr"].values):
                if pd.isna(v):
                    kw = dict(fc=MISSING, ec="none", lw=0)
                elif v == 1:
                    kw = dict(fc=colour, ec=EDGE, lw=0.3)
                else:
                    kw = dict(fc="white", ec="#9a9a9a", lw=0.3)
                ax.add_patch(Rectangle((x - half + 0.03, y - cell_h + 0.2), 2 * half - 0.06,
                                       cell_h - 0.4, zorder=2, **kw))
            ax.text(label_x, y - cell_h / 2, f"{hz} y", ha="right", va="center", transform=trans)
            y -= cell_h
        ax.text(label_x - 0.045, (top + y) / 2, name, ha="right", va="center", transform=trans)
        y -= group_gap
    outcome_bottom = y + group_gap

    # predictors (top): rows that are zero at every landmark of this person are left out
    blocks = []
    for name, cols in predictor_blocks(df):
        vals = display_values(df, cols, rows)
        keep = vals.max(axis=1) > 0
        blocks.append((name, 10 if cols == [EMB_COL] else len(cols), vals[keep] if keep.any() else vals[:1] * 0))
    top_y, bottom_y, gap, min_h = 60.5, 25.0, 0.7, 2.2
    avail = top_y - bottom_y - gap * (len(blocks) - 1)
    unit = 1.0
    while sum(max(min_h, len(v) * unit) for _, _, v in blocks) > avail:
        unit -= 0.01
    y = top_y
    for name, n, vals in blocks:
        h = max(min_h, len(vals) * unit)
        ax.imshow(vals, extent=(left, right, y - h, y), aspect="auto", cmap=VALUE_CMAP,
                  vmin=0, vmax=1, interpolation="none", zorder=1)
        ax.add_patch(Rectangle((left, y - h), right - left, h, fill=False, ec="#9a9a9a", lw=0.3, zorder=3))
        ax.text(label_x, y - h / 2, f"{name} ({n})", ha="right", va="center", transform=trans)
        y -= h + gap
    for x in xs[:-1]:                                  # gaps between landmark columns
        ax.plot([x + half, x + half], [y + gap, top_y], color="white", lw=0.6, zorder=2)

    # side labels
    side_x = (6 - PLOT_X) / PLOT_W
    for text, lo, hi in [("Predictors", y + gap, top_y), ("Outcome labels", outcome_bottom, 22.4)]:
        ax.plot([side_x + 0.012] * 2, [lo, hi], color=INK, lw=0.5, transform=trans, clip_on=False)
        ax.text(side_x, (lo + hi) / 2, text, rotation=90, ha="right", va="center", transform=trans)

    # the landmark t0
    x0 = years(t0, origin)
    ax.add_patch(Rectangle((x0 - half, outcome_bottom - 0.1), 2 * half, top_y + 0.3 - outcome_bottom,
                           fill=False, ec=INK, lw=0.75, zorder=5, clip_on=False))

    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 2))
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, int(xmax) + 1))
    ax.set_xlabel("Years since first record", labelpad=2)
    panel_label(fig, "b", 2, 77)


# ---------------------------------------------------------------------------
# panel c: cross-validation and the key to panel b
# ---------------------------------------------------------------------------
def panel_c(fig):
    x, w = 141, 40
    ax = mm_axes(fig, x, 0, w, FIG_H)
    ax.set_xlim(0, w)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    panel_label(fig, "c", x - 3, FIG_H - 2)

    train, test = "#d9d9d9", "#4d4d4d"
    ax.text(0, 124.6, "Five-fold cross-validation,\ngrouped by person", va="top")
    bar_x, bar_w, bar_h = 4.2, 35.5, 2.2
    top = 112
    for i in range(5):
        yy = top - i * 3.1
        ax.text(bar_x - 1.2, yy + bar_h / 2, f"{i + 1}", ha="right", va="center")
        for j in range(5):
            ax.add_patch(Rectangle((bar_x + j * bar_w / 5 + 0.25, yy), bar_w / 5 - 0.5, bar_h,
                                   fc=test if i == j else train, lw=0))
    yy = top - 4 * 3.1 - 3.4
    for k, (lab, c) in enumerate([("Training folds", train), ("Test fold", test)]):
        ax.add_patch(Rectangle((bar_x + k * 18, yy - 0.9), 2.2, 1.8, fc=c, lw=0))
        ax.text(bar_x + k * 18 + 3.0, yy, lab, va="center")
    ax.text(0, yy - 3.6, "All landmarks of a person are in the\nsame fold.", va="top")

    def steps(y_top, rule_colour, title, items):
        bottom = y_top - 3.0 * len(items) - 1.2
        ax.plot([0.6, 0.6], [y_top + 1.0, bottom], color=rule_colour, lw=1.6, solid_capstyle="butt")
        ax.text(2.2, y_top, title, va="center")
        for k, item in enumerate(items, start=1):
            ax.text(3.6, y_top - 3.0 * k, item, va="center", color=DARK)
        return bottom

    y = steps(83, train, "Fitted on the training folds",
              ["Median imputation", "PCA of free text (10 components)", "LR, RF or XGBoost",
               "Sigmoid calibration"])
    y = steps(y - 3.2, test, "Evaluated on the test fold", ["AUROC, AUPRC, Brier score"])
    ax.text(0, y - 3.0, "Repeated for 9 targets and 10 seeds.", va="center")

    # key to panel b, level with the matrix it describes
    ky = 45
    ax.text(0, ky, "Predictor value (scaled)", va="center")
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(0.3, 22, ky - 3.9, ky - 2.1), cmap=VALUE_CMAP,
              aspect="auto", interpolation="bilinear")
    ax.add_patch(Rectangle((0.3, ky - 3.9), 21.7, 1.8, fill=False, ec="#9a9a9a", lw=0.3))
    ax.text(0.3, ky - 5.6, "Low", va="center")
    ax.text(22, ky - 5.6, "High", va="center", ha="right")

    ky = 27
    ax.text(0, ky, "Outcome label", va="center")
    for k, colour in enumerate([CP_COL, HA_COL, DEATH_COL]):
        ax.add_patch(Rectangle((0.3 + k * 2.3, ky - 3.8), 1.9, 1.6, fc=colour, ec=EDGE, lw=0.3))
    ax.text(7.6, ky - 3.0, "Event", va="center")
    ax.add_patch(Rectangle((0.3, ky - 6.7), 1.9, 1.6, fc="white", ec="#9a9a9a", lw=0.3))
    ax.text(3.0, ky - 5.9, "No event", va="center")
    ax.add_patch(Rectangle((0.3, ky - 9.6), 1.9, 1.6, fc=MISSING, lw=0))
    ax.text(3.0, ky - 8.8, "Not labelled", va="center")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic/social_care_synthetic.xlsx")
    ap.add_argument("--landmarks", default="data/synthetic/landmarks.pkl")
    ap.add_argument("--out", type=Path, default=Path("docs/pipeline.pdf"))
    args = ap.parse_args()

    sheets = used_records(pd.read_excel(args.data, sheet_name=SHEETS))
    df = pd.read_pickle(args.landmarks)
    person, t0 = pick_example(sheets, df)

    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    _, origin, xmax = panel_a(fig, sheets, df, person, t0)
    panel_b(fig, df, person, t0, origin, xmax)
    panel_c(fig)

    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"saved {args.out} and .png (synthetic person {person})")


if __name__ == "__main__":
    main()
