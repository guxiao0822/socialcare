"""Write a synthetic workbook that can stand in for the restricted data.

    python scripts/generate_synthetic_data.py --n-people 1000 --out-dir data/synthetic

Creates social_care_synthetic.xlsx (8 sheets), unique_conditions.txt and
other_condition_embeddings.pkl (random vectors, so torch is not needed).
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from socialcare.synthetic import generate, random_codebook


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-people", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sheets = generate(args.n_people, args.seed)
    with pd.ExcelWriter(args.out_dir / "social_care_synthetic.xlsx") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            print(f"{name:22s} {len(df):7d} rows")

    texts = sorted(sheets["Health Conditions"]["Other Condition"].dropna().unique())
    (args.out_dir / "unique_conditions.txt").write_text("\n".join(texts))
    with open(args.out_dir / "other_condition_embeddings.pkl", "wb") as f:
        pickle.dump(random_codebook(texts, seed=args.seed + 1), f)
    print(f"written to {args.out_dir}/")


if __name__ == "__main__":
    main()
