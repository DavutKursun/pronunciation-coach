import pytest

from pronunciation.feedback import Issue
from pronunciation.metrics import (bootstrap_ci, error_level_catch, match_errors, pearson, top_pairs,
                                   word_detection_metrics)


def test_word_detection_metrics_on_a_small_table():
    # 10 words: the expert found 4 wrong; we flagged 3 of those 4 and 2 of the 6 correct ones
    expert_wrong = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    flagged = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    m = word_detection_metrics(expert_wrong, flagged)
    assert m["confusion"] == {"tp": 3, "fp": 2, "fn": 1, "tn": 4}
    assert m["words"] == 10 and m["expert_wrong"] == 4
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["false_alarm"] == pytest.approx(2 / 6)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


def test_word_detection_metrics_without_flags_or_errors():
    m = word_detection_metrics([0, 0, 1], [0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert word_detection_metrics([], [])["words"] == 0


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0      # undefined for a constant input


def test_top_pairs():
    assert top_pairs([["ð → d", "z → s"], ["z → s"], []], 1) == [("z → s", 2)]


def issue(kind, expected, heard, tip=None):
    return Issue(0, kind, expected, heard, tip, "")


def test_match_errors_counts_each_expert_error_once():
    # "things" said "tins": the expert wrote θ → t and z → s; we found θ → t and z missing
    expert = [issue("sub", "θ", "t", "th_voiceless"), issue("sub", "z", "s", "final_voicing")]
    ours = [issue("sub", "θ", "t", "th_voiceless"), issue("del", "z", None)]
    assert match_errors(expert, ours) == (2, 1)          # both found, one exactly
    assert match_errors(expert, ours[:1]) == (1, 1)
    assert match_errors(expert, []) == (0, 0)
    # one reported error cannot account for two expert errors on the same sound
    assert match_errors([issue("sub", "ɪ", "i"), issue("sub", "ɪ", "i")], [issue("sub", "ɪ", "i")]) == (1, 1)


def test_match_errors_pairs_insertions_with_insertions():
    expert = [issue("ins", None, "ɪ", "epenthesis")]
    assert match_errors(expert, [issue("ins", None, "ə")]) == (1, 0)
    assert match_errors(expert, [issue("sub", "s", "z")]) == (0, 0)


def test_error_level_catch_on_a_small_table():
    words = [
        # expert: 2 errors, we found both
        ([issue("sub", "θ", "t", "th_voiceless"), issue("sub", "ɪ", "i", "short_i")],
         [issue("sub", "θ", "t", "th_voiceless"), issue("sub", "ɪ", "iː", "short_i")]),
        # expert: 2 errors, we found one
        ([issue("sub", "θ", "t", "th_voiceless"), issue("sub", "z", "s", "final_voicing")],
         [issue("sub", "θ", "t", "th_voiceless")]),
        # expert: 1 error, we found none (and reported something else)
        ([issue("sub", "w", "v", "w")], [issue("sub", "ɛ", "æ")]),
        # expert: no error
        ([], [issue("sub", "t", "d")]),
    ]
    m = error_level_catch(words)
    assert m["expert_errors"] == 5
    assert m["found"] == 3 and m["found_exact"] == 2
    assert m["catch"] == pytest.approx(3 / 5)
    assert m["multi_error_words"] == 2
    assert (m["multi_all"], m["multi_some"], m["multi_none"]) == (1, 1, 0)
    assert m["by_tip"]["th_voiceless"] == {"errors": 2, "found": 2}
    assert m["by_tip"]["final_voicing"] == {"errors": 1, "found": 0}
    assert m["by_tip"]["w"] == {"errors": 1, "found": 0}


def test_bootstrap_ci_over_speakers():
    def share(units):
        return sum(u[0] for u in units) / sum(u[1] for u in units)

    same = [(1, 2)] * 5                      # every speaker: 1 of 2 -> no uncertainty
    assert bootstrap_ci(same, share) == pytest.approx((0.5, 0.5))
    mixed = [(0, 10), (5, 10), (10, 10)]
    low, high = bootstrap_ci(mixed, share, n_resamples=500, seed=1)
    assert 0.0 <= low < share(mixed) < high <= 1.0
    assert bootstrap_ci(mixed, share, n_resamples=500, seed=1) == (low, high)   # reproducible


def test_edit_distance_and_per():
    from pronunciation.metrics import edit_distance, phone_error_rate

    assert edit_distance(["θ", "ɪ", "ŋ", "k"], ["t", "ɪ", "ŋ"]) == 2          # one substitution, one deletion
    assert edit_distance([], ["a"]) == 1 and edit_distance(["a"], ["a"]) == 0
    # PER over a set of sentences: total edits / total reference phonemes
    assert phone_error_rate([(["a", "b"], ["a", "c"]), (["d", "e"], ["d", "e"])]) == pytest.approx(1 / 4)


def test_mdd_counts_on_hand_made_examples():
    from pronunciation.metrics import mdd_counts

    canonical = ["θ", "ɪ", "ŋ", "k"]
    # the speaker said "tink"; the model heard "tink": a true rejection with the right diagnosis
    c = mdd_counts(canonical, ["t", "ɪ", "ŋ", "k"], ["t", "ɪ", "ŋ", "k"])
    assert (c["ta"], c["fr"], c["fa"], c["tr"], c["diagnosis_ok"]) == (3, 0, 0, 1, 1)
    # the model heard "sink": detected, but diagnosed wrong
    c = mdd_counts(canonical, ["t", "ɪ", "ŋ", "k"], ["s", "ɪ", "ŋ", "k"])
    assert (c["tr"], c["diagnosis_ok"]) == (1, 0)
    # the model heard "think": the error is missed (false acceptance)
    c = mdd_counts(canonical, ["t", "ɪ", "ŋ", "k"], canonical)
    assert (c["ta"], c["fa"], c["tr"]) == (3, 1, 0)
    # a correct "think" heard as "thin": a false rejection (deletion)
    c = mdd_counts(canonical, canonical, ["θ", "ɪ", "n"])
    assert (c["ta"], c["fr"]) == (2, 2)


def test_mdd_counts_insertions_and_deletions():
    from pronunciation.metrics import mdd_counts

    canonical = ["s", "k", "uː", "l"]
    # "is-chool": an added vowel; the model also heard an added vowel (a different one): detected
    c = mdd_counts(canonical, ["ɪ", "s", "k", "uː", "l"], ["ə", "s", "k", "uː", "l"])
    assert (c["ta"], c["tr"], c["diagnosis_ok"], c["fa"], c["fr"]) == (4, 1, 0, 0, 0)
    # an added vowel the model did not hear: false acceptance
    c = mdd_counts(canonical, ["ɪ", "s", "k", "uː", "l"], canonical)
    assert (c["ta"], c["fa"]) == (4, 1)
    # a sound the model added although the speaker did not: false rejection
    c = mdd_counts(canonical, canonical, ["ɪ", "s", "k", "uː", "l"])
    assert (c["ta"], c["fr"]) == (4, 1)
    # the final l dropped, and the model heard it dropped: true rejection, right diagnosis
    c = mdd_counts(canonical, ["s", "k", "uː"], ["s", "k", "uː"])
    assert (c["ta"], c["tr"], c["diagnosis_ok"]) == (3, 1, 1)


def test_mdd_counts_accepted_variants_are_not_errors():
    from pronunciation.metrics import mdd_counts

    # "better" with a flap on both sides: our feedback accepts ɾ for t, so nothing is wrong
    c = mdd_counts(["b", "ɛ", "t", "ɚ"], ["b", "ɛ", "ɾ", "ɚ"], ["b", "ɛ", "ɾ", "ɚ"])
    assert (c["ta"], c["fr"], c["fa"], c["tr"]) == (4, 0, 0, 0)


def test_mdd_metrics():
    from collections import Counter

    from pronunciation.metrics import mdd_metrics

    m = mdd_metrics(Counter(ta=80, fr=5, fa=10, tr=15, diagnosis_ok=9))
    assert m["precision"] == pytest.approx(15 / 20)
    assert m["recall"] == pytest.approx(15 / 25)
    assert m["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert m["diagnosis_accuracy"] == pytest.approx(9 / 15)
    assert m["false_rejection_rate"] == pytest.approx(5 / 85)


def test_canonical_bias_compares_per_against_both_references():
    from pronunciation.metrics import canonical_bias

    expected = ["θ", "ɪ", "ŋ", "z"]
    heard = ["t", "ɪ", "ŋ", "s"]                    # the expert heard "tings"
    # a recognizer that writes what should have been said: far from what was heard
    b = canonical_bias([(expected, heard, expected)])
    assert b["per_vs_heard"] == pytest.approx(2 / 4) and b["per_vs_expected"] == 0.0
    assert b["gap"] == pytest.approx(0.5)             # positive: pulled towards the expected sounds
    # a recognizer that writes what was said
    b = canonical_bias([(heard, heard, expected)])
    assert b["per_vs_heard"] == 0.0 and b["gap"] == pytest.approx(-0.5)
    # over several recordings: total edits / total reference length
    b = canonical_bias([(expected, heard, expected), (["a"], ["a"], ["a"])])
    assert b["per_vs_heard"] == pytest.approx(2 / 5) and b["recordings"] == 2
