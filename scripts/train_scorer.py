"""Train a model that predicts the experts' pronunciation score from our features.

The model is CHOSEN with cross-validation on the training speakers only (GroupKFold by
speaker, so no speaker is in both parts of a fold) and evaluated ONCE on the official
test split. The main metric is the Pearson correlation with the expert scores (PCC),
the standard metric for speechocean762.

Usage:
    python scripts/train_scorer.py                    # target: sentence accuracy (0-10)
    python scripts/train_scorer.py --target total
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pronunciation.assess import FEATURE_NAMES  # noqa: E402


def candidate_models() -> dict:
    return {
        "Baseline: phone accuracy only": make_pipeline(
            ColumnTransformer([("phone_accuracy", "passthrough", ["phone_accuracy"])]), LinearRegression()),
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    }


def pcc(y_true, y_pred) -> float:
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--target", default="accuracy", choices=["accuracy", "total", "fluency", "completeness", "prosodic"])
    parser.add_argument("--out", type=Path, default=Path("models/scorer.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()

    train = pd.read_csv(args.features_dir / "train.csv")
    test = pd.read_csv(args.features_dir / "test.csv")
    X_train, y_train = train[FEATURE_NAMES], train[args.target].to_numpy(float)
    X_test, y_test = test[FEATURE_NAMES], test[args.target].to_numpy(float)

    n_splits = max(2, min(5, len(train) // 2))
    if "speaker" in train and train["speaker"].nunique() >= n_splits:
        cv, groups = GroupKFold(n_splits=n_splits), train["speaker"]
    else:
        cv, groups = KFold(n_splits=n_splits, shuffle=True, random_state=0), None

    rows = []
    for name, model in candidate_models().items():
        cv_pred = cross_val_predict(model, X_train, y_train, cv=cv, groups=groups)
        model.fit(X_train, y_train)
        test_pred = np.clip(model.predict(X_test), 0, 10)
        rows.append({
            "model": name,
            "cv_pcc": pcc(y_train, cv_pred),
            "test_pcc": pcc(y_test, test_pred),
            "test_spearman": float(spearmanr(y_test, test_pred)[0]) if np.std(test_pred) > 0 else 0.0,
            "test_mse": float(np.mean((y_test - test_pred) ** 2)),
            "fitted": model,
        })

    # choose on cross-validation only; the test set is never used for choosing
    best = max((r for r in rows if not r["model"].startswith("Baseline")), key=lambda r: r["cv_pcc"])

    print(f"\nTarget: sentence {args.target} (0-10), train {len(train)} / test {len(test)} utterances\n")
    print("| Model | CV PCC (train speakers) | Test PCC | Test Spearman | Test MSE |")
    print("| --- | --- | --- | --- | --- |")
    for r in rows:
        marker = " **(selected)**" if r is best else ""
        print(f"| {r['model']}{marker} | {r['cv_pcc']:.3f} | {r['test_pcc']:.3f} | {r['test_spearman']:.3f} | {r['test_mse']:.2f} |")

    metrics = {
        "target": args.target,
        "n_train": len(train),
        "n_test": len(test),
        "selected_model": best["model"],
        "models": [{k: v for k, v in r.items() if k != "fitted"} for r in rows],
    }

    words_path = args.features_dir / "test_words.csv"
    if words_path.exists():
        words = pd.read_csv(words_path)
        if len(words) > 1:
            metrics["word_level_pcc"] = pcc(words["human_accuracy"], words["our_score"])
            print(f"\nWord level: PCC between our word scores and expert word accuracy = "
                  f"{metrics['word_level_pcc']:.3f} ({len(words)} words, no training involved)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["fitted"], "features": FEATURE_NAMES, "target": args.target,
                 "sklearn_version": sklearn.__version__, "test_pcc": best["test_pcc"]}, args.out)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved the selected model to {args.out} and metrics to {args.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
