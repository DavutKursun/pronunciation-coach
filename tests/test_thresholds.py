import math

import pandas as pd
import pytest

from pronunciation.feedback import Issue
from pronunciation.thresholds import (apply_gop_thresholds, cross_validate_gop_thresholds, scores,
                                      select_gop_thresholds, threshold_pairs)

TH = Issue(0, "sub", "θ", "t", "th_voiceless", "")      # a typical Turkish error (it has a tip)
OTHER = Issue(0, "sub", "m", "n", None, "")             # any other difference
GRID = [math.inf, 0.0, -1.0, -2.0, -3.0]


def word(speaker, group, wrong, gop, *issues):
    return {"speaker": speaker, "group": group, "expert_wrong": wrong, "gop": gop, "our_errors": list(issues)}


def table():
    """Words evaluated with no GOP check: every difference, and the word's GOP."""
    rows = [
        word("t1", "turkish", True, -2.5, OTHER),     # caught only if other differences need > -2.5
        word("t1", "turkish", True, -0.5, TH),        # caught only if patterns need > -0.5
        word("t2", "turkish", True, -1.5, TH),
        word("t2", "turkish", False, -1.5, OTHER),    # a false alarm for thresholds > -1.5
        word("t2", "turkish", False, 0.0),            # no difference at all
        word("t1", "turkish", True, None),            # missed whatever the thresholds
    ]
    rows += [word("e1", "english", False, -0.5, OTHER)]          # a native false alarm for thresholds > -0.5
    rows += [word("e1", "english", False, 0.0) for _ in range(49)]
    return pd.DataFrame(rows)


def test_threshold_pairs_never_ask_more_evidence_for_typical_errors():
    pairs = threshold_pairs(GRID)
    assert all(p >= t for t, p in pairs) and (-2.0, 0.0) in pairs and (0.0, -2.0) not in pairs


def test_apply_gop_thresholds_matches_the_feedback_rule():
    rows = apply_gop_thresholds(table(), -2.0, -1.0)
    assert rows.flagged.tolist()[:6] == [True, False, True, False, False, False]
    # a word without a GOP (pandas stores None as NaN) keeps its differences, like confirm_with_gop
    rows = apply_gop_thresholds(pd.DataFrame([word("t", "turkish", True, None, OTHER)]), -2.0, -1.0)
    assert rows.flagged.tolist() == [True]


def test_scores_on_the_table():
    s = scores(apply_gop_thresholds(table(), -1.0, -1.0))
    # Turkish: 4 wrong words, flagged 2 of them + 1 correct word; native: 0 of 50
    assert s["recall"] == pytest.approx(2 / 4) and s["precision"] == pytest.approx(2 / 3)
    assert s["english_false_alarm"] == 0.0


def test_select_gop_thresholds_takes_the_most_recall_within_the_limits():
    rows = table()
    chosen = select_gop_thresholds(rows, GRID, min_precision=0.70, max_false_alarm=0.025)
    # (-2, 0) and (-2, inf) flag the same words; the one asking for more evidence is taken
    assert chosen["thresholds"] == (-2.0, 0.0)            # catches the -0.5 θ, leaves out the -1.5 false alarm
    assert chosen["scores"]["recall"] == pytest.approx(3 / 4) and chosen["scores"]["precision"] == 1.0
    # nothing with more recall stays within the limits
    for t, p in threshold_pairs(GRID):
        s = scores(apply_gop_thresholds(rows, t, p))
        if s["precision"] >= 0.70 and s["english_false_alarm"] <= 0.025:
            assert s["recall"] <= chosen["scores"]["recall"]
    assert select_gop_thresholds(rows, GRID, min_precision=1.01) is None      # impossible limits


def test_cross_validation_holds_every_speaker_out_once():
    rows = pd.concat([table().assign(speaker=lambda d, k=k: d.speaker + f"_{k}") for k in range(3)],
                     ignore_index=True)
    result = cross_validate_gop_thresholds(rows, GRID, n_folds=3, min_precision=0.70, max_false_alarm=0.025)
    assert len(result["folds"]) == 3
    held_out = [s for fold in result["folds"] for s in fold["speakers"]]
    assert sorted(held_out) == sorted(rows.speaker.unique())            # each speaker exactly once
    assert result["rows"].index.equals(rows.index)
    assert all(fold["thresholds"] == (-2.0, 0.0) and not fold["fallback"] for fold in result["folds"])


def test_cross_validation_falls_back_when_no_pair_fits():
    result = cross_validate_gop_thresholds(table(), GRID, n_folds=2, fallback=(-2.5, -1.0), min_precision=1.01)
    assert all(fold["fallback"] and fold["thresholds"] == (-2.5, -1.0) for fold in result["folds"])
