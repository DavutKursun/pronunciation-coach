"""Word- and error-level detection metrics: do we flag the words (and sounds) the experts flagged?

Word level, for every word: the expert found it wrong or not, we flagged it or not. That gives a
confusion matrix (tp = both, fp = only we, fn = only the expert, tn = neither) and from it:

  false alarm  fp / (fp + tn)   correct words we flagged
  recall       tp / (tp + fn)   wrong words we flagged (the "catch rate")
  precision    tp / (tp + fp)   flagged words that were really wrong
  F1           harmonic mean of precision and recall

Error level: a word can have several errors ("things" said "tins": θ → t and z → s). Each expert
error counts on its own: did we report an error on the same sound?

Phone level (mispronunciation detection and diagnosis, MDD, as on L2-ARCTIC): for every expected
sound, compare what the annotator heard with what the recognizer output:

  TA  said right, recognized right     FR  said right, recognized as an error (false rejection)
  FA  said wrong, recognized right     TR  said wrong, recognized as an error (true rejection)

  precision TR / (TR + FR), recall TR / (TR + FA), F1; diagnosis accuracy = share of TR where the
  recognizer heard the same sound as the annotator. PER = edit distance to what the annotator
  heard / its length. "Right" uses our accepted variants (a flap for t is not an error).

Used by scripts/evaluate_words.py, scripts/evaluate_saa.py, scripts/train_scorer.py and
scripts/evaluate_l2arctic.py.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

import numpy as np

from .align import align
from .feedback import Issue
from .phonemes import is_acceptable

# speechocean762 word accuracy (0-10): 10 = "the pronunciation of the word is perfect",
# 7-9 = "most phones are pronounced correctly but have accents", 4-6 = "less than 30% of the
# phones are wrongly pronounced". Our feedback reports any difference from the expected sounds,
# so a word counts as really wrong when the experts did not find it perfect.
SPEECHOCEAN_WRONG_BELOW = 10


def word_detection_metrics(expert_wrong: Sequence[bool], flagged: Sequence[bool]) -> dict:
    wrong = np.asarray(expert_wrong, dtype=bool)
    ours = np.asarray(flagged, dtype=bool)
    return detection_from_counts(int((wrong & ours).sum()), int((~wrong & ours).sum()),
                                 int((wrong & ~ours).sum()), int((~wrong & ~ours).sum()))


def detection_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """The same metrics from the four cells of the confusion matrix."""
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
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


def bootstrap_ci(units: Sequence, statistic: Callable[[list], float], n_resamples: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """95% confidence interval of `statistic` by resampling whole units (speakers) with replacement.

    With few speakers, most of the uncertainty comes from who was recorded, not from how many
    words they said, so the speaker is the unit that gets resampled.
    """
    rng = np.random.default_rng(seed)
    values = [statistic([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(n_resamples)]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_diff(units_a: Sequence, units_b: Sequence, statistic: Callable[[list], float],
                          n_resamples: int = 2000, seed: int = 0) -> dict:
    """Is system B really better than system A? Both are measured on the same speakers.

    units_a[i] and units_b[i] belong to the same speaker. Each resample draws speakers with
    replacement and computes statistic(B) - statistic(A) on the SAME speakers, so differences
    between speakers cancel out and only the difference between the systems remains. The
    improvement counts as real when the 95% interval of the difference does not contain zero.
    """
    if len(units_a) != len(units_b):
        raise ValueError("both systems must be measured on the same speakers")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(units_a), len(units_a))
        diffs.append(statistic([units_b[i] for i in pick]) - statistic([units_a[i] for i in pick]))
    low, high = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    a, b = statistic(list(units_a)), statistic(list(units_b))
    return {"a": a, "b": b, "diff": b - a, "ci95": (low, high), "real": low > 0 or high < 0}


def format_confusion(m: dict) -> str:
    c = m["confusion"]
    return ("                      we flagged   we did not\n"
            f"  expert: wrong       {c['tp']:>10}   {c['fn']:>10}\n"
            f"  expert: correct     {c['fp']:>10}   {c['tn']:>10}")


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Plain Levenshtein distance: every substitution, deletion and insertion costs 1."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def phone_error_rate(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> float:
    """PER over (reference, hypothesis) pairs: total edits / total reference phonemes."""
    edits = length = 0
    for reference, hypothesis in pairs:
        edits += edit_distance(reference, hypothesis)
        length += len(reference)
    return edits / length if length else 0.0


def phone_states(canonical: Sequence[str], other: Sequence[str], flags: Sequence[bool]):
    """Per expected sound: None (said as expected) or the other sound ("" = left out); per gap: added sounds."""
    state: list[str | None] = [None] * len(canonical)
    added: dict[int, list[str]] = {}
    seen = 0                                       # expected sounds before the current position
    for op in align(list(canonical), list(other), list(flags)):
        if op.exp_pos is None:
            added.setdefault(seen, []).append(op.heard)
            continue
        seen = op.exp_pos + 1
        if op.kind == "del":
            state[op.exp_pos] = ""
        elif op.kind == "sub":
            state[op.exp_pos] = op.heard
    return state, added


def mdd_counts(canonical: Sequence[str], perceived: Sequence[str], recognized: Sequence[str],
               function_flags: Sequence[bool] | None = None) -> Counter:
    """TA, FR, FA, TR and correct diagnoses for one sentence (see the module docstring).

    Both the annotator's and the recognizer's version are aligned with the expected sounds. Every
    expected sound is one decision; every gap where either side added a sound is one more (TR if
    both added something, FA if only the annotator did, FR if only the recognizer did).
    """
    flags = function_flags or [False] * len(canonical)
    p_state, p_added = phone_states(canonical, perceived, flags)
    r_state, r_added = phone_states(canonical, recognized, flags)
    counts: Counter = Counter()
    for expected, p, r in zip(canonical, p_state, r_state):
        if p is None:
            counts["fr" if r is not None else "ta"] += 1
        elif r is None:
            counts["fa"] += 1
        else:
            counts["tr"] += 1
            counts["diagnosis_ok"] += p == r or (p != "" and r != "" and is_acceptable(p, r))
    for gap in set(p_added) | set(r_added):
        if gap in p_added and gap in r_added:
            counts["tr"] += 1
            counts["diagnosis_ok"] += p_added[gap] == r_added[gap]
        else:
            counts["fa" if gap in p_added else "fr"] += 1
    return counts


def mdd_metrics(counts: Counter) -> dict:
    ta, fr, fa, tr = (counts.get(k, 0) for k in ("ta", "fr", "fa", "tr"))
    precision = tr / (tr + fr) if tr + fr else 0.0
    recall = tr / (tr + fa) if tr + fa else 0.0
    return {
        "ta": ta, "fr": fr, "fa": fa, "tr": tr,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "diagnosis_accuracy": counts.get("diagnosis_ok", 0) / tr if tr else 0.0,
        "false_rejection_rate": fr / (ta + fr) if ta + fr else 0.0,
    }
