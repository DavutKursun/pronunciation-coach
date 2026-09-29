"""Choose the two GOP confirmation thresholds of the v1 rules for a recognizer, honestly.

A difference is reported only when the word's GOP passes the check of feedback.is_confirmed:
below `threshold`, or below `pattern_threshold` for a typical Turkish-speaker error. v1 chose
-2.5 / -1.0 for the original recognizer; a fine-tuned recognizer is sure of itself in a different
way, so the pair is chosen again on labelled speakers (the Speech Accent Archive dev half):

  select_gop_thresholds          the pair with the most Turkish recall while Turkish precision stays
                                 at least 70% and native speakers get at most 2.5% false alarms
  cross_validate_gop_thresholds  the honest estimate of that choice: pick the pair on some
                                 speakers, measure it on the others, until every speaker was held out

Optionally a third threshold is chosen with them: the hidden-w margin (feedback.report_hidden_w),
which reports w -> v when v/β/ʋ came close to w although the recognizer wrote w. Its reports must
also be right: at least 70% of them on a w the expert marked wrong (like the precision limit). Word
precision alone cannot check this, as most such words are wrong for other sounds anyway, and a
wrong "w" tip would still mislead the learner.

The words are evaluated once with no GOP check and no hidden-w rule (every difference, the word's
GOP and its w margin); any thresholds are then applied offline with exactly the rules the app
uses, so a sweep takes seconds.
"""

from __future__ import annotations

import math

import pandas as pd

from .feedback import hidden_w_issue, is_confirmed
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


def apply_w_margin(rows: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """Add the hidden w -> v of every word whose "w_margin" is above `threshold` (None: the rule is off)."""
    out = rows.copy()
    if threshold is not None and "w_margin" in out:
        out["our_errors"] = [issues + [hidden_w_issue(int(index), rival)] if _gop(margin) is not None and margin > threshold
                             else issues
                             for issues, margin, rival, index in zip(out["our_errors"], out["w_margin"], out["w_rival"],
                                                                     out.get("word_idx", pd.Series(0, index=out.index)))]
    out["flagged"] = out["our_errors"].map(bool)
    return out


def apply_thresholds(rows: pd.DataFrame, thresholds: tuple) -> pd.DataFrame:
    """(threshold, pattern_threshold) or (threshold, pattern_threshold, w_margin_threshold), in the app's order:
    the GOP check first, then the hidden-w rule."""
    threshold, pattern_threshold, *w = thresholds
    return apply_w_margin(apply_gop_thresholds(rows, threshold, pattern_threshold), w[0] if w else None)


def hidden_w_precision(rows: pd.DataFrame, threshold: float | None) -> tuple[int, int]:
    """(right, all) hidden-w reports on Turkish speakers: right when the expert marked that word's w wrong."""
    if threshold is None or "w_margin" not in rows:
        return 0, 0
    reported = [margin is not None and not (isinstance(margin, float) and math.isnan(margin)) and margin > threshold
                for margin in rows["w_margin"]]
    turkish = rows[pd.Series(reported, index=rows.index) & (rows["group"] == "turkish")]
    right = sum(any(e.expected == "w" and e.kind != "ins" for e in errors) for errors in turkish["expert_errors"])
    return right, len(turkish)


def scores(rows: pd.DataFrame) -> dict:
    """Turkish precision, recall, F1 and false alarm; native false alarm (words with an expert label)."""
    labelled = rows[rows["expert_wrong"].notna()]
    turkish = labelled[labelled["group"] == "turkish"]
    english = labelled[labelled["group"] == "english"]
    t = word_detection_metrics(turkish["expert_wrong"].astype(bool), turkish["flagged"])
    e = word_detection_metrics(english["expert_wrong"].astype(bool), english["flagged"])
    return {"precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "turkish_false_alarm": t["false_alarm"], "english_false_alarm": e["false_alarm"]}


def select_gop_thresholds(rows: pd.DataFrame, grid: list[float] = GRID, w_grid: list | None = None,
                          min_precision: float = 0.70, max_false_alarm: float = 0.025,
                          min_w_precision: float = 0.70) -> dict | None:
    """The thresholds with the most Turkish recall within the limits, or None if none are within them.

    With `w_grid` (values for the hidden-w margin, None = rule off) the result is a triple.
    Ties go to the higher precision, then fewer native false alarms, then the thresholds that ask
    for more evidence (lower GOP thresholds, a higher w margin): thresholds that flag the same dev
    words are not told apart by the data, and the stricter ones are the safer bet on new speakers.
    """
    best = None
    w_values = w_grid if w_grid is not None else [None]
    w_precision = {w: hidden_w_precision(rows, w) for w in w_values}       # does not depend on the GOP check
    for t, p in threshold_pairs(grid):
        checked = apply_gop_thresholds(rows, t, p)
        for w in w_values:
            right, reports = w_precision[w]
            if reports and right / reports < min_w_precision:
                continue
            s = scores(apply_w_margin(checked, w))
            s["w_report_precision"] = right / reports if reports else None
            if s["precision"] < min_precision or s["english_false_alarm"] > max_false_alarm:
                continue
            key = (s["recall"], s["precision"], -s["english_false_alarm"], -t, -p, math.inf if w is None else w)
            if best is None or key > best[0]:
                best = (key, (t, p) if w_grid is None else (t, p, w), s)
    return None if best is None else {"thresholds": best[1], "scores": best[2]}


def speaker_folds(rows: pd.DataFrame, n_folds: int) -> dict[str, int]:
    """Speakers of each group dealt round-robin into folds, so every fold has both groups."""
    folds = {}
    for group in sorted(rows["group"].unique()):
        for i, speaker in enumerate(sorted(rows.loc[rows["group"] == group, "speaker"].unique())):
            folds[speaker] = i % n_folds
    return folds


def cross_validate_gop_thresholds(rows: pd.DataFrame, grid: list[float] = GRID, n_folds: int = 6,
                                  fallback: tuple = V1_THRESHOLDS, w_grid: list | None = None, **limits) -> dict:
    """Choose the pair without the held-out speakers, apply it to them; every speaker is held out once.

    Returns the held-out rows (in the original order, flags as they would be on new speakers) and,
    per fold, the speakers and the chosen pair (the fallback when no pair was within the limits).
    """
    fold = rows["speaker"].map(speaker_folds(rows, n_folds))
    held_out, folds = [], []
    for f in range(n_folds):
        pick = select_gop_thresholds(rows[fold != f], grid, w_grid, **limits)
        chosen = pick["thresholds"] if pick else fallback
        folds.append({"fold": f, "thresholds": chosen, "fallback": pick is None,
                      "speakers": sorted(rows.loc[fold == f, "speaker"].unique())})
        held_out.append(apply_thresholds(rows[fold == f], chosen))
    return {"rows": pd.concat(held_out).loc[rows.index], "folds": folds}
