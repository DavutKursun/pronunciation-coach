"""Choose the two GOP confirmation thresholds of the v1 rules for a recognizer, honestly.

A difference is reported only when the word's GOP passes the check of feedback.is_confirmed:
below `threshold`, or below `pattern_threshold` for a typical Turkish-speaker error. v1 chose
-2.5 / -1.0 for the original recognizer; a fine-tuned recognizer is sure of itself in a different
way, so the pair is chosen again on labelled speakers (the Speech Accent Archive dev half):

  select_gop_thresholds          the pair with the most Turkish recall while Turkish precision stays
                                 at least 70% and native speakers get at most 2.5% false alarms
  cross_validate_gop_thresholds  the honest estimate of that choice: pick the pair on some
                                 speakers, measure it on the others, until every speaker was held out

The words are evaluated once with no GOP check (every difference plus the word's GOP); any pair is
then applied offline with exactly the rule the app uses, so a sweep takes seconds.
"""

from __future__ import annotations

import math

import pandas as pd

from .feedback import is_confirmed
from .metrics import word_detection_metrics

GRID = [math.inf, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -8.0, -10.0]
V1_THRESHOLDS = (-2.5, -1.0)


def threshold_pairs(grid: list[float] = GRID) -> list[tuple[float, float]]:
    """(threshold, pattern_threshold) pairs in which typical errors never need more evidence than others."""
    return [(t, p) for t in grid for p in grid if p >= t]


def _gop(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def apply_gop_thresholds(rows: pd.DataFrame, threshold: float, pattern_threshold: float) -> pd.DataFrame:
    """Words evaluated with no GOP check -> as if the check had used these thresholds.

    rows: one word per row with "our_errors" (every difference) and "gop" (None/NaN when the word
    has none: then its differences stand, like in confirm_with_gop).
    """
    out = rows.copy()
    out["our_errors"] = [[i for i in issues if is_confirmed(i, _gop(gop), threshold, pattern_threshold)]
                         for issues, gop in zip(rows["our_errors"], rows["gop"])]
    out["flagged"] = out["our_errors"].map(bool)
    return out


def scores(rows: pd.DataFrame) -> dict:
    """Turkish precision, recall, F1 and false alarm; native false alarm (words with an expert label)."""
    labelled = rows[rows["expert_wrong"].notna()]
    turkish = labelled[labelled["group"] == "turkish"]
    english = labelled[labelled["group"] == "english"]
    t = word_detection_metrics(turkish["expert_wrong"].astype(bool), turkish["flagged"])
    e = word_detection_metrics(english["expert_wrong"].astype(bool), english["flagged"])
    return {"precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "turkish_false_alarm": t["false_alarm"], "english_false_alarm": e["false_alarm"]}


def select_gop_thresholds(rows: pd.DataFrame, grid: list[float] = GRID, min_precision: float = 0.70,
                          max_false_alarm: float = 0.025) -> dict | None:
    """The pair with the most Turkish recall within the limits, or None if no pair is within them.

    Ties go to the higher precision, then fewer native false alarms, then the pair that asks for
    more evidence (lower thresholds): pairs that flag the same dev words are not told apart by the
    data, and the stricter one is the safer bet on new speakers.
    """
    best = None
    for t, p in threshold_pairs(grid):
        s = scores(apply_gop_thresholds(rows, t, p))
        if s["precision"] < min_precision or s["english_false_alarm"] > max_false_alarm:
            continue
        key = (s["recall"], s["precision"], -s["english_false_alarm"], -t, -p)
        if best is None or key > best[0]:
            best = (key, (t, p), s)
    return None if best is None else {"thresholds": best[1], "scores": best[2]}


def speaker_folds(rows: pd.DataFrame, n_folds: int) -> dict[str, int]:
    """Speakers of each group dealt round-robin into folds, so every fold has both groups."""
    folds = {}
    for group in sorted(rows["group"].unique()):
        for i, speaker in enumerate(sorted(rows.loc[rows["group"] == group, "speaker"].unique())):
            folds[speaker] = i % n_folds
    return folds


def cross_validate_gop_thresholds(rows: pd.DataFrame, grid: list[float] = GRID, n_folds: int = 6,
                                  fallback: tuple[float, float] = V1_THRESHOLDS, **limits) -> dict:
    """Choose the pair without the held-out speakers, apply it to them; every speaker is held out once.

    Returns the held-out rows (in the original order, flags as they would be on new speakers) and,
    per fold, the speakers and the chosen pair (the fallback when no pair was within the limits).
    """
    fold = rows["speaker"].map(speaker_folds(rows, n_folds))
    held_out, folds = [], []
    for f in range(n_folds):
        pick = select_gop_thresholds(rows[fold != f], grid, **limits)
        pair = pick["thresholds"] if pick else fallback
        folds.append({"fold": f, "thresholds": pair, "fallback": pick is None,
                      "speakers": sorted(rows.loc[fold == f, "speaker"].unique())})
        held_out.append(apply_gop_thresholds(rows[fold == f], *pair))
    return {"rows": pd.concat(held_out).loc[rows.index], "folds": folds}
