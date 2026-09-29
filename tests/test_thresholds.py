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
    return {"speaker": speaker, "group": group, "expert_wrong": wrong, "gop": gop, "our_errors": list(issues),
            "expert_errors": []}


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


def test_apply_w_margin_adds_the_hidden_w_like_the_app():
    from pronunciation.thresholds import apply_w_margin

    rows = pd.DataFrame([word("t1", "turkish", True, 0.0), word("e1", "english", False, 0.0)])
    rows["w_margin"], rows["w_rival"] = [-1.0, -6.0], ["β", "v"]
    out = apply_w_margin(rows, -3.0)
    assert out.flagged.tolist() == [True, False]
    [issue] = out.our_errors[0]
    assert (issue.kind, issue.expected, issue.heard, issue.tip) == ("sub", "w", "β", "w")
    assert apply_w_margin(rows, None).flagged.tolist() == [False, False]     # rule off


def test_selection_can_include_the_w_threshold():
    rows = table()
    rows["w_margin"], rows["w_rival"] = None, None
    rows.loc[5, ["w_margin", "w_rival"]] = [-1.0, "v"]     # the missed Turkish word: a hidden w -> v
    rows.at[5, "expert_errors"] = [Issue(0, "sub", "w", "v", "w", "")]
    rows.loc[6, ["w_margin", "w_rival"]] = [-4.0, "v"]     # a native word with v further away
    chosen = select_gop_thresholds(rows, GRID, w_grid=[None, -2.0, -5.0], min_precision=0.70, max_false_alarm=0.025)
    assert chosen["thresholds"] == (-2.0, 0.0, -2.0)       # -2 catches the hidden w and leaves the native word alone
    assert chosen["scores"]["recall"] == 1.0


W_ERROR = Issue(0, "sub", "w", "v", "w", "")              # what the expert wrote: w said as v


def hidden_w_table():
    rows = table()
    rows["w_margin"], rows["w_rival"] = None, None
    rows.loc[5, ["w_margin", "w_rival"]] = [-1.0, "v"]      # a missed Turkish word: a real hidden w -> v
    rows.at[5, "expert_errors"] = [W_ERROR]
    rows.loc[2, ["w_margin", "w_rival"]] = [-1.8, "v"]      # wrong for its θ only: a w report there would be wrong
    rows.at[2, "expert_errors"] = [TH]
    extra = pd.DataFrame([word("t3", "turkish", True, None)])   # another missed word with a real hidden w
    extra["w_margin"], extra["w_rival"] = -1.9, "v"
    extra.at[0, "expert_errors"] = [W_ERROR]
    return pd.concat([rows, extra], ignore_index=True)


def test_hidden_w_precision_counts_reports_on_a_real_w_error():
    from pronunciation.thresholds import hidden_w_precision

    rows = hidden_w_table()
    assert hidden_w_precision(rows, -1.5) == (1, 1)          # only the -1.0 word: right
    assert hidden_w_precision(rows, -2.0) == (2, 3)          # the θ word gets a wrong w report
    assert hidden_w_precision(rows, None) == (0, 0)


def test_w_threshold_keeps_the_w_reports_precise():
    rows = hidden_w_table()
    lenient = select_gop_thresholds(rows, GRID, w_grid=[None, -1.5, -2.0], min_w_precision=0.6)
    strict = select_gop_thresholds(rows, GRID, w_grid=[None, -1.5, -2.0])     # default: at least 70%, like precision
    assert lenient["thresholds"][2] == -2.0 and strict["thresholds"][2] == -1.5
    assert strict["scores"]["w_report_precision"] == 1.0


# v2-3d: hidden errors of every pattern, each with its own margin threshold
Z_ERROR = Issue(0, "sub", "z", "s", "final_voicing", "")    # what the expert wrote: a final z said as s
PATTERN_GRID = [0.0, -1.0, -2.0, -3.0]


def hidden_word(speaker, group, wrong, margin, expert=(), tip="final_voicing", expected="z", sound="s"):
    """A word in which greedy decoding heard no difference, and `sound` came `margin` close to `expected`."""
    from pronunciation.feedback import Rival

    return word(speaker, group, wrong, 0.0) | {"rivals": [Rival(0, expected, tip, sound, margin)],
                                               "expert_errors": list(expert), "word_idx": 0}


def rows_of(*rows, natives=100):
    """The words, and native words with nothing to report."""
    quiet = [word(f"e{k % 5}", "english", False, 0.0) | {"rivals": [], "word_idx": 0} for k in range(natives)]
    return pd.DataFrame(list(rows) + quiet)


def test_apply_settings_adds_hidden_errors_like_the_app():
    from pronunciation.feedback import hidden_issue
    from pronunciation.thresholds import apply_settings

    rows = rows_of(hidden_word("t1", "turkish", True, -0.5, [Z_ERROR]), hidden_word("t2", "turkish", True, -2.5),
                   natives=1)
    out = apply_settings(rows, {"gop_threshold": None, "hidden_thresholds": {"final_voicing": -1.0}})
    assert out.flagged.tolist() == [True, False, False]
    assert out.our_errors[0] == [hidden_issue(0, rows.rivals[0][0])] and out.our_errors[0][0].hidden
    assert not apply_settings(rows, {"hidden_thresholds": {"final_voicing": None}}).flagged.any()     # off


def test_diagnosis_precision_needs_the_same_sound_and_pattern():
    from pronunciation.thresholds import diagnosis_precision

    hidden_z = Issue(0, "sub", "z", "s", "final_voicing", "", hidden=True)
    hidden_d = Issue(0, "sub", "d", "t", "final_voicing", "", hidden=True)
    z_missing = Issue(0, "del", "z", None, None, "")
    rows = pd.DataFrame([
        word("t1", "turkish", True, 0.0, hidden_z) | {"expert_errors": [Z_ERROR]},         # right
        word("t1", "turkish", True, 0.0, hidden_d) | {"expert_errors": [Z_ERROR]},         # the pattern, another sound
        word("t1", "turkish", True, 0.0, hidden_z) | {"expert_errors": [z_missing]},       # the sound, another error
        word("t1", "turkish", True, 0.0, Issue(0, "sub", "z", "s", "final_voicing", "")) | {"expert_errors": [Z_ERROR]},
        word("e1", "english", False, 0.0, hidden_z),                                         # natives do not count
    ])
    assert diagnosis_precision(rows, "final_voicing") == (1, 3)      # the greedy report is not a hidden one


def test_a_hidden_pattern_needs_at_least_ten_reports():
    from pronunciation.thresholds import select_hidden_thresholds

    nine = rows_of(*[hidden_word(f"t{k % 3}", "turkish", True, -0.5, [Z_ERROR]) for k in range(9)])
    ten = rows_of(*[hidden_word(f"t{k % 3}", "turkish", True, -0.5, [Z_ERROR]) for k in range(10)])
    assert select_hidden_thresholds(nine, PATTERN_GRID)["final_voicing"] is None     # right every time, but too few
    assert select_hidden_thresholds(ten, PATTERN_GRID)["final_voicing"] == -3.0      # the most lenient that fits
    assert select_hidden_thresholds(nine, PATTERN_GRID, min_reports=9)["final_voicing"] == -3.0
    assert set(select_hidden_thresholds(ten, PATTERN_GRID).values()) == {None, -3.0}  # patterns never seen stay off


def test_hidden_reports_must_name_the_error_the_expert_heard():
    from pronunciation.thresholds import select_hidden_thresholds

    right = [hidden_word(f"t{k % 3}", "turkish", True, -0.5, [Z_ERROR]) for k in range(10)]
    # wrong words, but for their θ: "z said as s" names the wrong error (word precision cannot see this)
    misnamed = [hidden_word(f"t{k % 3}", "turkish", True, -2.5, [TH]) for k in range(5)]
    rows = rows_of(*right, *misnamed)
    assert select_hidden_thresholds(rows, PATTERN_GRID)["final_voicing"] == -2.0          # -3: 10 of 15 right
    assert select_hidden_thresholds(rows, PATTERN_GRID, min_diagnosis_precision=0.6)["final_voicing"] == -3.0


def test_patterns_are_chosen_one_after_another_in_phonemes_order():
    from pronunciation.thresholds import select_hidden_thresholds

    th_error = Issue(0, "sub", "θ", "t", "th_voiceless", "")
    th = [hidden_word(f"t{k % 3}", "turkish", True, -0.5, [th_error], "th_voiceless", "θ", "t") for k in range(10)]
    z = [hidden_word(f"t{k % 3}", "turkish", True, -0.5, [Z_ERROR]) for k in range(10)]
    natives = [hidden_word("e9", "english", False, -1.5, [], "th_voiceless", "θ", "t") for _ in range(2)]
    natives.append(hidden_word("e9", "english", False, -2.5))
    rows = rows_of(*th, *z, *natives)          # 103 native words: 2 flagged = 1.9%, 3 flagged = 2.9%
    chosen = select_hidden_thresholds(rows, PATTERN_GRID)
    # θ comes before final devoicing in phonemes.py and takes the two native words; -3 for z would add a third
    assert chosen["th_voiceless"] == -3.0 and chosen["final_voicing"] == -2.0


def test_generalized_selection_is_cross_validated_by_speaker():
    from pronunciation.thresholds import cross_validate_generalized

    rows = rows_of(*[hidden_word(f"t{k % 6}", "turkish", True, -0.5, [Z_ERROR]) for k in range(36)], natives=60)
    result = cross_validate_generalized(rows, GRID, PATTERN_GRID, n_folds=3)
    held_out = [s for fold in result["folds"] for s in fold["speakers"]]
    assert sorted(held_out) == sorted(rows.speaker.unique())           # each speaker exactly once
    assert result["rows"].index.equals(rows.index)
    # chosen on the 24 training words of each fold, applied to the held-out speakers
    assert all(fold["settings"]["hidden_thresholds"]["final_voicing"] == -3.0 for fold in result["folds"])
    assert result["rows"].flagged[rows.group == "turkish"].all()
