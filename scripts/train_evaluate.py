"""Cross-validated training and evaluation of the fixed-horizon models.

    # quick run: XGBoost, one seed, all 9 targets
    python scripts/train_evaluate.py --data data/synthetic/landmarks.pkl --out-dir results/synthetic

    # as in the paper
    python scripts/train_evaluate.py --data landmarks.pkl --out-dir results \
        --models xgb rf lr --seeds 42 7 13 21 37 101 123 2023 2024 2025 --device cuda --shap

Writes per-run predictions and fold metrics, plus summary.csv (mean and SD
over folds and seeds).
"""

import argparse
from pathlib import Path

import pandas as pd

from socialcare.config import EMB_COL, PAPER_SEEDS, PCA_DIM, TARGETS
from socialcare.modelling import METRICS, cross_validate, predictor_columns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="output of build_dataset.py")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--models", nargs="+", default=["xgb"], choices=["xgb", "rf", "lr"],
                    help="logistic regression is slow (saga, max_iter=5000)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="the paper used: " + " ".join(map(str, PAPER_SEEDS)))
    ap.add_argument("--device", default="cpu", help="XGBoost device (cpu or cuda)")
    ap.add_argument("--shap", action="store_true", help="also save SHAP importance (XGBoost only)")
    args = ap.parse_args()

    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(args.data)
    n_pred = len(predictor_columns(df)) - 1 + PCA_DIM if EMB_COL in df else len(predictor_columns(df))
    print(f"{len(df)} landmarks, {n_pred} predictors\n")

    all_folds = []
    for target in args.targets:
        for model in args.models:
            for seed in args.seeds:
                run = f"{target}__{model}__seed{seed}"
                folds, preds, shap = cross_validate(df, target, model, seed, device=args.device, with_shap=args.shap)
                preds.to_pickle(runs_dir / f"preds__{run}.pkl")
                folds.to_csv(runs_dir / f"metrics__{run}.csv", index=False)
                if shap is not None:
                    shap.to_csv(runs_dir / f"shap__{run}.csv", index=False)
                m = folds[METRICS].mean()
                print(f"{run:40s} AUROC {m.AUROC:.3f}  AUPRC {m.AUPRC:.3f}  Brier {m.Brier:.3f}")
                all_folds.append(folds.assign(target=target, model=model, seed=seed))

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    g = folds.groupby(["target", "model"])[METRICS]
    summary = g.mean().round(3).astype(str) + " (" + g.std().round(3).astype(str) + ")"
    summary.reset_index().to_csv(args.out_dir / "summary.csv", index=False)
    print("\nmean (SD) over folds and seeds\n" + summary.to_string())


if __name__ == "__main__":
    main()
