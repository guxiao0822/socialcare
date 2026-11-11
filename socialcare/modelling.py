"""Models, fold-internal preprocessing, calibration and evaluation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import EMB_COL, EMB_DIM, ID, PCA_DIM

METRICS = ["AUROC", "AUPRC", "ACC", "F1", "Brier", "LogLoss"]


def predictor_columns(df):
    """The full a priori predictor set, in a fixed order."""
    def starts(*p):
        return [c for c in df.columns if c.startswith(p)]

    def present(cols):
        return [c for c in cols if c in df.columns]

    cols = (present(["Age", "Gender"])
            + starts("svc_active__", "svc_count__", "svc_dur_")
            + present(["ha_count", "ha_los_sum_days", "ha_los_mean_days", "ha_los_std_days", "ha_active"])
            + starts("hc_active__", "hc_count__", "hc_dur_")
            + present([EMB_COL])
            + starts("ca_count__")
            + present(["cp_count", "cp_dur_sum_days", "cp_dur_mean_days", "cp_dur_std_days", "cp_active"])
            + starts("ot_count__")
            + starts("re_"))
    return list(dict.fromkeys(cols))


class EmbeddingToMatrix(BaseEstimator, TransformerMixin):
    """A column holding 768-d arrays (NaN if no text) -> (n, 768) float matrix."""

    def __init__(self, dim=EMB_DIM):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = X.iloc[:, 0].values if isinstance(X, pd.DataFrame) else np.asarray(X)[:, 0]
        out = np.full((len(col), self.dim), np.nan, dtype=np.float32)
        for i, v in enumerate(col):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[i] = np.asarray(v, dtype=np.float32).reshape(-1)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"other_condition_emb{i}" for i in range(self.dim)], dtype=object)


def make_preprocessor(feature_cols):
    """Median imputation; embedding -> zero-fill -> PCA(10). Fitted per training fold."""
    numeric = [c for c in feature_cols if c != EMB_COL]
    steps = [("num", SimpleImputer(strategy="median"), numeric)]
    if EMB_COL in feature_cols:
        steps.append(("emb", Pipeline([
            ("to_matrix", EmbeddingToMatrix()),
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("pca", PCA(n_components=PCA_DIM, random_state=42)),
        ]), [EMB_COL]))
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=0.3)


def make_model(name, seed, device="cpu"):
    """Pre-specified hyperparameters; no tuning was done."""
    if name == "xgb":
        return XGBClassifier(n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
                             eval_metric="auc", tree_method="hist", random_state=seed, device=device)
    if name == "rf":
        return RandomForestClassifier(n_estimators=600, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if name == "lr":
        return Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5, C=1.0,
                                      class_weight="balanced", max_iter=5000, random_state=seed)),
        ])
    raise ValueError(f"unknown model {name!r}; use xgb, rf or lr")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class SigmoidCalibrator:
    """Platt scaling on the logit of the model's probabilities."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(solver="lbfgs", max_iter=2000).fit(_logit(p), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p))[:, 1]


def evaluate(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "ACC": accuracy_score(y, pred), "F1": f1_score(y, pred, zero_division=0),
            "Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, np.c_[1 - p, p])}


def shap_importance(pipe, X_train, X_test, seed, fold, n_background=2000, n_explain=5000):
    """Mean |SHAP| (XGBoost), background from the training fold, explained on the test fold."""
    import shap
    rng = np.random.RandomState(seed + 1000 * fold)
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    A, B = pre.transform(X_train), pre.transform(X_test)
    A = A[rng.choice(len(A), min(n_background, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n_explain, len(B)), replace=False)]
    sv = shap.TreeExplainer(model, data=A, feature_perturbation="interventional",
                            model_output="raw").shap_values(B)
    sv = sv[1] if isinstance(sv, list) else sv
    return pd.DataFrame({"feature": pre.get_feature_names_out(), "mean_abs_shap": np.abs(sv).mean(axis=0)})


def cross_validate(df, target, model_name, seed, n_splits=5, device="cpu", with_shap=False):
    """Subject-wise k-fold CV for one target / model / seed.

    GroupKFold does not shuffle, so every seed sees the same folds; the seed
    only changes the model's own randomness (seed + fold).

    Returns (fold_metrics, predictions, shap or None).
    """
    d = df.loc[df[target].notna()]
    feats = predictor_columns(d)
    X, y, groups = d[feats], d[target].astype(int).values, d[ID].values

    metrics, preds, shaps = [], [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        pipe = Pipeline([("pre", make_preprocessor(feats)),
                         ("model", make_model(model_name, seed + fold, device))])
        pipe.fit(X.iloc[tr], y[tr])
        cal = SigmoidCalibrator().fit(pipe.predict_proba(X.iloc[tr])[:, 1], y[tr])
        p = cal.predict(pipe.predict_proba(X.iloc[te])[:, 1])

        metrics.append({"fold": fold, "n_test": len(te), **evaluate(y[te], p)})
        preds.append(pd.DataFrame({"fold": fold, "row_index": d.index[te], ID: groups[te],
                                   "y_true": y[te], "y_prob": p}))
        if with_shap and model_name == "xgb":
            shaps.append(shap_importance(pipe, X.iloc[tr], X.iloc[te], seed, fold).assign(fold=fold))

    shap_df = None
    if shaps:
        shap_df = (pd.concat(shaps).groupby("feature")["mean_abs_shap"].agg(["mean", "std"])
                     .sort_values("mean", ascending=False).reset_index())
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True), shap_df
