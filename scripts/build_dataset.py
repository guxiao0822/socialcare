"""Turn the raw social care tables into one row per person per landmark.

    python scripts/build_dataset.py \
        --data data/synthetic/social_care_synthetic.xlsx \
        --embeddings data/synthetic/other_condition_embeddings.pkl \
        --out data/synthetic/landmarks.pkl
"""

import argparse
import pickle

import pandas as pd

from socialcare.cohort import build_cohort, build_landmarks
from socialcare.config import ID, KEYS, REF, SHEETS
from socialcare.features import build_features
from socialcare.targets import build_targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Excel workbook with the 8 sheets (docs/data_format.md)")
    ap.add_argument("--embeddings", required=True, help="codebook from embed_other_conditions.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sheets = pd.read_excel(args.data, sheet_name=SHEETS)
    with open(args.embeddings, "rb") as f:
        cb = pickle.load(f)
    codebook = dict(zip(cb["conditions"], cb["embeddings"].values))

    demo = build_cohort(sheets)
    landmarks, instances = build_landmarks(demo)
    targets = build_targets(sheets, instances)
    df = build_features(sheets, landmarks, instances, codebook).merge(targets, on=KEYS, how="left")
    df["instance_id"] = df[ID].astype(str) + "__" + df[REF].dt.strftime("%Y-%m-%d")

    df.to_pickle(args.out)
    print(f"saved {args.out}: {len(df)} landmarks, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
