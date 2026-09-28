"""Word- and error-level detection metrics: do we flag the words (and sounds) the experts flagged?

Word level, for every word: the expert found it wrong or not, we flagged it or not. That gives a
confusion matrix (tp = both, fp = only we, fn = only the expert, tn = neither) and from it:

  false alarm  fp / (fp + tn)   correct words we flagged
  recall       tp / (tp + fn)   wrong words we flagged (the "catch rate")
  precision    tp / (tp + fp)   flagged words that were really wrong
  F1           harmonic mean of precision and recall

Error level: a word can have several errors ("things" said "tins": θ → t and z → s). Each expert
error counts on its own: did we report an error on the same sound?

Used by scripts/evaluate_words.py, scripts/evaluate_saa.py and scripts/train_scorer.py.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .feedback import Issue

# speechocean762 word accuracy (0-10): 10 = "the pronunciation of the word is perfect",
# 7-9 = "most phones are pronounced correctly but have accents", 4-6 = "less than 30% of the
# phones are wrongly pronounced". Our feedback reports any difference from the expected sounds,
# so a word counts as really wrong when the experts did not find it perfect.
SPEECHOCEAN_WRONG_BELOW = 10


def word_detection_metrics(expert_wrong: Sequence[bool], flagged: Sequence[bool]) -> dict:
    wrong = np.asarray(expert_wrong, dtype=bool)
    ours = np.asarray(flagged, dtype=bool)
    tp, fp = int((wrong & ours).sum()), int((~wrong & ours).sum())
    fn, tn = int((wrong & ~ours).sum()), int((~wrong & ~ours).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "words": tp + fp + fn + tn,
        "expert_wrong": tp + fn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_alarm": fp / (fp + tn) if fp + tn else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when it is undefined (fewer than 2 values or a constant input)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def top_pairs(pair_lists: Iterable[list[str]], n: int = 15) -> list[tuple[str, int]]:
    """Most common "expected → heard" pairs over many words."""
    return Counter(p for pairs in pair_lists for p in pairs).most_common(n)


def match_errors(expert: list[Issue], ours: list[Issue]) -> tuple[int, int]:
    """How many expert errors we also reported: (found, found exactly).

    Found: we reported an error on the same expected sound (a substitution or a deletion), or
    an added sound for an added sound. Exactly: also the same realization (θ → t for θ → t).
    Each reported error accounts for at most one expert error.
    """
    remaining = list(ours)
    found = exact = 0
    for e in expert:
        if e.kind == "ins":
            candidates = [o for o in remaining if o.kind == "ins"]
        else:
            candidates = [o for o in remaining if o.kind != "ins" and o.expected == e.expected]
        same = [o for o in candidates if o.kind == e.kind and o.heard == e.heard]
        pick = (same or candidates or [None])[0]
        if pick is not None:
            remaining.remove(pick)
            found += 1
            exact += bool(same)
    return found, exact


def error_level_catch(words: Iterable[tuple[list[Issue], list[Issue]]]) -> dict:
    """Error-level catch rate over (expert errors, our reported errors) pairs, one pair per word."""
    n_errors = n_found = n_exact = 0
    multi = {"all": 0, "some": 0, "none": 0}
    by_tip: dict[str, dict[str, int]] = {}
    for expert, ours in words:
        found, exact = match_errors(expert, ours)
        n_errors += len(expert)
        n_found += found
        n_exact += exact
        if len(expert) >= 2:
            multi["all" if found == len(expert) else "some" if found else "none"] += 1
        for e in expert:
            if e.tip:
                counts = by_tip.setdefault(e.tip, {"errors": 0, "found": 0})
                counts["errors"] += 1
                counts["found"] += match_errors([e], ours)[0]
    return {
        "expert_errors": n_errors, "found": n_found, "found_exact": n_exact,
        "catch": n_found / n_errors if n_errors else 0.0,
        "exact": n_exact / n_errors if n_errors else 0.0,
        "multi_error_words": sum(multi.values()),
        "multi_all": multi["all"], "multi_some": multi["some"], "multi_none": multi["none"],
        "by_tip": by_tip,
    }


def format_detection(name: str, m: dict) -> str:
    return (f"{name:12} words {m['words']:5}  expert wrong {m['expert_wrong']:5}   false alarm {m['false_alarm']:6.1%}  "
            f"recall {m['recall']:6.1%}  precision {m['precision']:6.1%}  F1 {m['f1']:.3f}")


def format_confusion(m: dict) -> str:
    c = m["confusion"]
    return ("                      we flagged   we did not\n"
            f"  expert: wrong       {c['tp']:>10}   {c['fn']:>10}\n"
            f"  expert: correct     {c['fp']:>10}   {c['tn']:>10}")
