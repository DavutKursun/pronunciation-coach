"""Train the learned error detector (v2) and choose its red/yellow thresholds. One command, from scratch.

1. Labels: speechocean762 phone scores of the "fit" speakers, moved onto our expected phonemes
   (pronunciation/speechocean.py). Two definitions of an error: "strict" (the experts' average
   rounds to 0: wrong or missing) and "lenient" (rounds to 0 or 1: also a heavy accent).
2. Features: pronunciation/detector.py, from the cached recognizer output.
3. Model: gradient boosting, probabilities calibrated with isotonic regression; both with 5-fold
   cross-validation grouped by speaker on "fit".
4. The label definition and the thresholds are chosen on the SAA dev half (never training data):
   red = at least 85% precision on Turkish speakers; yellow = the most recall with red + yellow
   precision at least 70% and at most 2% false alarms for native speakers. An honest estimate
   chooses the thresholds on some dev speakers and measures on the others.
5. The "val" speakers are only measured (phone-level AUC and average precision, and permutation
   importance of the features). The SAA test half and the speechocean762 test split stay locked.

Writes models/detector.joblib (with the recognizer id, the feature list and the scikit-learn
version) and results/detector_training.json.

Usage:
    python scripts/train_detector.py
    python scripts/train_detector.py --recognizer path/to/fine-tuned-model     # v2-3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_saa  # noqa: E402
import evaluate_words  # noqa: E402
from pronunciation.assess import analyze, prepare_expected  # noqa: E402
from pronunciation.detector import (FEATURES, Detector, choose_thresholds, cross_validate_thresholds,  # noqa: E402
                                    feature_matrix, gop_stats, phone_rows)
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.metrics import word_detection_metrics  # noqa: E402
from pronunciation.phonemes import FUNCTION_WORDS  # noqa: E402
from pronunciation.recognizer import DEFAULT_MODEL  # noqa: E402
from pronunciation.speechocean import LABELS, is_error, transfer_labels  # noqa: E402
from pronunciation.systems import System  # noqa: E402

NO_THRESHOLDS = {"red": 1.01, "yellow": 1.01}   # only probabilities (and the v1 rule for added sounds)


def phone_table(decoder, utterances):
    """Feature rows, expert scores and speakers for every labelled expected phoneme."""
    rows, scores, speakers, stats = [], [], [], Counter()
    for info, log_probs, seconds in utterances:
        words = tokenize(info["text"])
        if len(words) != len(info["words"]):
            stats["utterances skipped (word lists differ)"] += 1
            continue
        raw = phonemize_words(words)
        recognition = decoder(log_probs, seconds)
        result = analyze(info["text"], raw, recognition, decoder.token_to_id, decoder.blank_id)
        utterance_rows = phone_rows(words, raw, result, recognition, decoder.token_to_id, decoder.blank_id)
        labels = []
        for text, phones, so_word in zip(words, prepare_expected(words, raw)[0], info["words"]):
            word_labels, word_stats = transfer_labels(phones, so_word["phones"], so_word["phones-accuracy"],
                                                      text.lower() in FUNCTION_WORDS)
            labels += word_labels
            stats.update(word_stats)
        for row, score in zip(utterance_rows, labels):
            if score is not None:
                rows.append(row)
                scores.append(score)
                speakers.append(info["speaker"])
    return rows, np.array(scores), np.array(speakers), stats


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def train(X, y, groups) -> tuple[CalibratedClassifierCV, dict]:
    splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    out_of_fold = cross_val_predict(new_model(), X, y, cv=splits, method="predict_proba")[:, 1]
    model = CalibratedClassifierCV(new_model(), method="isotonic", cv=splits).fit(X, y)
    return model, {"errors": int(y.sum()), "phones": len(y), "cv_auc": roc_auc_score(y, out_of_fold),
                   "cv_average_precision": average_precision_score(y, out_of_fold)}


def dev_tables(rows):
    """Per group: word error probability, expert label and the added-sound flag (for choose_thresholds)."""
    rows = rows[rows.expert_wrong.notna()]
    return {g: {"p": part.error_prob.to_numpy(float), "wrong": part.expert_wrong.astype(bool).to_numpy(),
                "extra": part.flagged.to_numpy(bool)} for g, part in rows.groupby("group")}


def measure(tables: dict, thresholds: dict) -> dict:
    out = {}
    for g, t in tables.items():
        flagged = (t["p"] >= thresholds["yellow"]) | t["extra"]
        out[g] = word_detection_metrics(t["wrong"], flagged)
    out["turkish_red"] = word_detection_metrics(tables["turkish"]["wrong"], tables["turkish"]["p"] >= thresholds["red"])
    return out


def select_on_dev(detector: Detector, dev) -> dict:
    """Thresholds chosen on SAA dev, their (optimistic) dev result and the speaker cross-validated estimate."""
    decoder, data = dev
    rows = evaluate_saa.evaluate(decoder, data, System("probe", decision="detector",
                                                      settings={"detector": detector, "thresholds": NO_THRESHOLDS}))
    tables = dev_tables(rows)
    thresholds = choose_thresholds(tables["turkish"], tables["english"])
    labelled = rows[rows.expert_wrong.notna()]
    cv = cross_validate_thresholds(labelled.speaker, labelled.group, labelled.error_prob,
                                   labelled.expert_wrong.astype(bool), labelled.flagged)
    wrong, group = labelled.expert_wrong.astype(bool).to_numpy(), labelled.group.to_numpy()
    turkish = group == "turkish"
    cv_result = {"turkish": word_detection_metrics(wrong[turkish], cv["flagged"][turkish]),
                 "english": word_detection_metrics(wrong[~turkish], cv["flagged"][~turkish]),
                 "turkish_red": word_detection_metrics(wrong[turkish], cv["red"][turkish]),
                 "fold_thresholds": cv["thresholds"]}
    return {"thresholds": thresholds, "dev_in_sample": measure(tables, thresholds), "dev_cross_validated": cv_result,
            "dev_average_precision": average_precision_score(tables["turkish"]["wrong"], tables["turkish"]["p"])}


def line(name: str, m: dict) -> str:
    return f"{name:34} precision {m['precision']:6.1%}  recall {m['recall']:6.1%}  false alarm {m['false_alarm']:6.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recognizer", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "detector.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "detector_training.json")
    args = parser.parse_args()

    print("Features and labels on speechocean762 fit...")
    decoder, fit = evaluate_words.load("fit", args.recognizer)
    rows, scores, speakers, stats = phone_table(decoder, fit)
    print(f"  {len(rows)} labelled phonemes from {len(set(speakers))} speakers; label transfer: {dict(stats)}")
    stats_table = gop_stats(rows, list(scores >= 2.0))
    X = feature_matrix(rows, stats_table)

    dev = evaluate_saa.load("dev", args.recognizer)
    candidates = {}
    for label in LABELS:
        y = np.array([is_error(s, label) for s in scores])
        model, cv_fit = train(X, y, speakers)
        bundle = {"model": model, "features": FEATURES, "gop_stats": stats_table, "recognizer": args.recognizer,
                  "sklearn_version": sklearn.__version__, "label": label, "thresholds": None}
        selection = select_on_dev(Detector(bundle), dev)
        candidates[label] = {"bundle": bundle, "fit_cv": cv_fit, **selection}
        cv = selection["dev_cross_validated"]
        print(f"\n[{label}] fit: {cv_fit['errors']} errors of {cv_fit['phones']} phonemes, grouped CV AUC "
              f"{cv_fit['cv_auc']:.3f}, AP {cv_fit['cv_average_precision']:.3f}; SAA dev word AP "
              f"{selection['dev_average_precision']:.3f}; thresholds {selection['thresholds']}")
        print("  " + line("SAA dev Turkish (chosen on dev)", selection["dev_in_sample"]["turkish"]))
        print("  " + line("SAA dev Turkish (cross-validated)", cv["turkish"]))
        print("  " + line("SAA dev English (cross-validated)", cv["english"]))

    # the definition that keeps more recall under the constraints on held-out dev speakers
    chosen = max(candidates, key=lambda k: (candidates[k]["dev_cross_validated"]["turkish"]["recall"],
                                            candidates[k]["dev_average_precision"]))
    best = candidates[chosen]
    best["bundle"]["thresholds"] = best["thresholds"]
    print(f"\nChosen label definition: {chosen} (more recall on held-out dev speakers)")

    print("Measuring on speechocean762 val (no tuning)...")
    val_decoder, val = evaluate_words.load("val", args.recognizer)
    val_rows, val_scores, _, _ = phone_table(val_decoder, val)
    X_val = feature_matrix(val_rows, stats_table)
    y_val = np.array([is_error(s, chosen) for s in val_scores])
    p_val = best["bundle"]["model"].predict_proba(X_val)[:, 1]
    importance = permutation_importance(best["bundle"]["model"], X_val, y_val, scoring="average_precision",
                                        n_repeats=5, random_state=0)
    ranked = sorted(zip(FEATURES, importance.importances_mean, importance.importances_std), key=lambda x: -x[1])
    val_result = {"phones": len(y_val), "errors": int(y_val.sum()), "auc": roc_auc_score(y_val, p_val),
                  "average_precision": average_precision_score(y_val, p_val)}
    print(f"  val phonemes: AUC {val_result['auc']:.3f}, average precision {val_result['average_precision']:.3f} "
          f"({val_result['errors']} errors of {val_result['phones']})")
    print("  permutation importance (drop in val average precision):")
    for name, mean, std in ranked[:12]:
        print(f"    {name:26} {mean:+.4f} ± {std:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["bundle"], args.out, compress=3)
    report = {"recognizer": args.recognizer, "sklearn_version": sklearn.__version__, "label_transfer": dict(stats),
              "chosen_label": chosen, "speechocean_val": val_result,
              "permutation_importance": [{"feature": n, "mean": m, "std": s} for n, m, s in ranked],
              "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "bundle"} for k, v in candidates.items()}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=float))
    size = args.out.stat().st_size / 1e6
    print(f"\nSaved {args.out.relative_to(ROOT)} ({size:.1f} MB) and {args.report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
