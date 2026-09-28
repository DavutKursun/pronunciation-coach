import joblib
import numpy as np
import pytest
from conftest import BLANK, TOKEN_TO_ID, fake_recognition

from pronunciation.assess import PronunciationCoach, analyze
from pronunciation.detector import (FEATURES, Detector, choose_thresholds, competitors, cross_validate_thresholds,
                                    detector_for, feature_matrix, gop_stats, phone_rows)

THESE = [["ð", "iː", "z"]]


def rival(recognition, phone, other, p_phone=0.5, p_other=0.4):
    """In the frames the model gave to `phone`, make `other` almost as likely (a hidden z -> s)."""
    frames = recognition.log_probs.argmax(axis=1) == TOKEN_TO_ID[phone]
    recognition.log_probs[frames, TOKEN_TO_ID[phone]] = np.log(p_phone)
    recognition.log_probs[frames, TOKEN_TO_ID[other]] = np.log(p_other)
    return recognition


def rows_for(text, raw, recognition):
    result = analyze(text, raw, recognition, TOKEN_TO_ID, BLANK)
    return phone_rows([text], raw, result, recognition, TOKEN_TO_ID, BLANK)


class StubModel:
    """Error probability 0.9 when a typical replacement is nearly as likely as the expected sound."""

    def predict_proba(self, X):
        margin = X[:, FEATURES.index("competitor_margin_max")]
        p = np.where(np.nan_to_num(margin, nan=-99) > -1.0, 0.9, 0.05)
        return np.column_stack([1 - p, p])


def bundle(recognizer="test-model", thresholds=None):
    return {"model": StubModel(), "features": FEATURES, "gop_stats": {"__all__": [0.0, 1.0, 0.0, 1.0]},
            "recognizer": recognizer, "thresholds": thresholds or {"red": 0.8, "yellow": 0.5}}


def test_competitors_come_from_the_pattern_table_and_final_devoicing():
    assert {"t", "s"} <= competitors("θ", word_final=False)
    assert competitors("z", word_final=True) == {"s"}
    assert competitors("z", word_final=False) == set()      # devoicing only counts at the end of a word
    assert competitors("m", word_final=True) == set()


def test_phone_rows_describe_every_expected_sound():
    rows = rows_for("these", THESE, rival(fake_recognition(["ð", "iː", "z"]), "z", "s"))
    assert [r["phone"] for r in rows] == ["ð", "iː", "z"]
    z = rows[2]
    assert z["op"] == "match"                                  # the greedy transcription heard z ...
    assert z["competitor_margin_max"] == pytest.approx(np.log(0.4 / 0.5))   # ... but s was close
    assert z["top_other"] == "s" and z["has_competitor"] == 1.0
    assert z["in_final_cluster"] == 1.0 and z["pos_final"] == 1.0 and z["is_fricative"] == 1.0
    assert z["is_voiced"] == 1.0
    # CTC puts a sound in one or two frames and fills the rest with blanks, so a sound lasts from its
    # first frame to the next sound's first frame: 2 frames + 1 blank in fake_recognition = 60 ms
    assert rows[1]["duration"] == pytest.approx(0.06)
    assert z["prev_vowel_duration"] == pytest.approx(0.06)     # iː comes right before z
    assert "speaking_rate" not in FEATURES                     # a speaker-level shortcut, see detector.py
    assert np.isnan(rows[0]["prev_lpr"]) and rows[0]["next_lpr"] == rows[1]["lpr"]
    assert rows[1]["function_word"] == 0.0


def test_a_sound_missing_from_the_transcription_is_a_deletion():
    rows = rows_for("these", THESE, fake_recognition(["ð", "iː"]))
    assert rows[2]["op"] == "del" and rows[2]["op_del"] == 1.0


def test_gop_stats_and_normalized_features():
    rows = [{"phone": "z", "lpp": -1.0 - i % 2, "lpr": -0.5 * (i % 2)} for i in range(40)]
    stats = gop_stats(rows, [True] * 40)
    assert stats["z"][0] == pytest.approx(-1.5) and stats["z"][1] == pytest.approx(0.5)
    full = [{**{f: 0.0 for f in FEATURES}, "phone": "z", "lpp": -2.5, "lpr": 0.0}]
    matrix = feature_matrix(full, stats)
    assert matrix.shape == (1, len(FEATURES))
    assert matrix[0, FEATURES.index("lpp_z")] == pytest.approx((-2.5 + 1.5) / 0.5)


def test_detector_finds_a_devoiced_z_the_transcription_missed():
    detector = Detector(bundle())
    recognition = rival(fake_recognition(["ð", "iː", "z"]), "z", "s")
    [word] = detector.assess("these", THESE, recognition, TOKEN_TO_ID, BLANK).words
    assert word.level == "red" and word.error_prob == pytest.approx(0.9)
    assert [(i.expected, i.heard, i.tip) for i in word.issues] == [("z", "s", "final_voicing")]


def test_detector_ignores_a_difference_it_does_not_believe():
    # m has no typical replacement, so the stub model gives it a low error probability
    recognition = fake_recognition(["n", "æ", "n"])
    raw = [["m", "æ", "n"]]
    assert analyze("man", raw, recognition, TOKEN_TO_ID, BLANK).words[0].issues      # v1 reports m -> n
    [word] = Detector(bundle()).assess("man", raw, recognition, TOKEN_TO_ID, BLANK).words
    assert word.level is None and word.issues == [] and word.error_prob == pytest.approx(0.05)


def test_detector_keeps_the_v1_rule_for_added_sounds():
    detector = Detector(bundle())
    [word] = detector.assess("school", [["s", "k", "uː", "l"]], fake_recognition(["ɪ", "s", "k", "uː", "l"]),
                             TOKEN_TO_ID, BLANK).words
    assert [i.tip for i in word.issues] == ["epenthesis"] and word.level == "yellow"


def test_coach_falls_back_to_v1_without_a_matching_detector(tmp_path):
    class FakeRecognizer:
        token_to_id, blank_id, model_id = TOKEN_TO_ID, BLANK, "test-model"

        def recognize(self, audio):
            return rival(fake_recognition(["ð", "iː", "z"]), "z", "s")

    path = tmp_path / "detector.joblib"
    assert PronunciationCoach(FakeRecognizer(), scorer_path=None, detector_path=path).detector is None  # no file
    joblib.dump(bundle(recognizer="another-model"), path)
    assert detector_for(path, "test-model") is None                  # trained on another recognizer
    assert PronunciationCoach(FakeRecognizer(), scorer_path=None, detector_path=path).detector is None

    joblib.dump(bundle(recognizer="test-model"), tmp_path / "ok.joblib")
    coach = PronunciationCoach(FakeRecognizer(), scorer_path=None, detector_path=tmp_path / "ok.joblib")
    [these] = coach.assess(np.zeros(16000, np.float32), "these").words
    assert coach.detector is not None and these.level == "red"


def test_choose_thresholds_on_a_small_table():
    # Turkish words: probabilities and expert labels; the best words are wrong, then it gets mixed
    turkish = {"p": [0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05],
               "wrong": [1, 1, 1, 1, 0, 1, 0, 1, 0, 0], "extra": [0] * 10}
    english = {"p": [0.55] + [0.1] * 99, "wrong": [0] * 100, "extra": [0] * 100}
    t = choose_thresholds(turkish, english, red_precision=0.85, min_precision=0.70, max_english_false_alarm=0.02)
    assert t["red"] == 0.7        # 0.95..0.7: 4 of 4 wrong; 0.6 would drop precision to 4/5 = 80%
    assert t["yellow"] == 0.2     # 6 of 8 wrong (75%), English false alarm 1/100 = 1%; 0.1 -> 6 of 9
    strict_english = {"p": [0.55, 0.52] + [0.1] * 98, "wrong": [0] * 100, "extra": [0] * 100}
    t = choose_thresholds(turkish, strict_english, max_english_false_alarm=0.01)
    assert t["yellow"] == 0.55    # anything lower flags both English words (2%)


def test_choose_thresholds_counts_words_flagged_by_the_added_sound_rule():
    turkish = {"p": [0.9, 0.1], "wrong": [1, 0], "extra": [0, 1]}
    english = {"p": [0.1] * 10, "wrong": [0] * 10, "extra": [0] * 10}
    t = choose_thresholds(turkish, english, min_precision=0.7)
    assert t == {"red": 0.9, "yellow": 0.9}     # the extra flag alone already costs precision (1 of 2)


def test_cross_validated_thresholds_are_chosen_without_the_held_out_speakers():
    speakers = [f"t{i}" for i in range(6) for _ in range(5)] + [f"e{i}" for i in range(6) for _ in range(5)]
    groups = ["turkish"] * 30 + ["english"] * 30
    rng = np.random.default_rng(0)
    p = rng.random(60)
    wrong = [pi > 0.5 for pi in p[:30]] + [False] * 30
    result = cross_validate_thresholds(speakers, groups, p, wrong, [False] * 60, n_folds=3)
    assert len(result["thresholds"]) == 3 and result["flagged"].shape == (60,)
    assert all(t["yellow"] <= t["red"] for t in result["thresholds"])


def test_red_threshold_also_keeps_native_false_alarms_low():
    turkish = {"p": [0.9, 0.8], "wrong": [1, 1], "extra": [0, 0]}
    english = {"p": [0.85] + [0.1] * 19, "wrong": [0] * 20, "extra": [0] * 20}
    t = choose_thresholds(turkish, english, max_english_false_alarm=0.02)
    assert t["red"] == 0.9        # 0.8 is precise enough but flags 1 of 20 native words (5%)
