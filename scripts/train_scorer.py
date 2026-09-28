"""Train a model that predicts the experts' pronunciation score from our features.

The model is CHOSEN with cross-validation on the training speakers only (GroupKFold by
speaker, so no speaker is in both parts of a fold) and evaluated ONCE on the official
test split. The main metric is the Pearson correlation with the expert scores (PCC),
the standard metric for speechocean762.

Word-level error detection on the test split is reported too (no training involved, and no
threshold or rule is tuned on it): do we flag the words the experts did not find perfect?

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
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pronunciation.assess import FEATURE_NAMES  # noqa: E402
from pronunciation.metrics import SPEECHOCEAN_WRONG_BELOW, pearson, word_detection_metrics  # noqa: E402


def candidate_models() -> dict:
    return {
        "Baseline: phone accuracy only": make_pipeline(
            ColumnTransformer([("phone_accuracy", "passthrough", ["phone_accuracy"])]), LinearRegression()),
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    }


def spearman(y_true, y_pred) -> float:
    """Spearman correlation: the Pearson correlation of the ranks."""
    return pearson(rankdata(y_true), rankdata(y_pred))


def word_detection(words: pd.DataFrame) -> dict:
    """Word-level detection on extract_features' *_words.csv (same definition as evaluate_words.py)."""
    result = word_detection_metrics(words["human_accuracy"] < SPEECHOCEAN_WRONG_BELOW, words["n_issues"] > 0)
    result["expert_threshold"] = SPEECHOCEAN_WRONG_BELOW
    result["pcc"] = pearson(words["our_score"], words["human_accuracy"])
    return result


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
            "cv_pcc": pearson(y_train, cv_pred),
            "test_pcc": pearson(y_test, test_pred),
            "test_spearman": spearman(y_test, test_pred),
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
            d = word_detection(words)
            metrics["word_detection"] = d
            metrics["word_level_pcc"] = d["pcc"]
            c = d["confusion"]
            print(f"\nWord-level error detection on the test split ({d['words']} words; really wrong = expert "
                  f"score below {d['expert_threshold']}; no training or tuning involved)\n")
            print("| Words | Expert: wrong | False alarm | Recall | Precision | F1 | Word score PCC |")
            print("| --- | --- | --- | --- | --- | --- | --- |")
            print(f"| {d['words']} | {d['expert_wrong']} | {d['false_alarm']:.1%} | {d['recall']:.1%} | "
                  f"{d['precision']:.1%} | {d['f1']:.3f} | {d['pcc']:.3f} |")
            print(f"\nConfusion matrix: tp {c['tp']}, fp {c['fp']}, fn {c['fn']}, tn {c['tn']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["fitted"], "features": FEATURE_NAMES, "target": args.target,
                 "sklearn_version": sklearn.__version__, "test_pcc": best["test_pcc"]}, args.out)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved the selected model to {args.out} and metrics to {args.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
