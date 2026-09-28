import numpy as np
from conftest import BLANK, TOKEN_TO_ID, fake_recognition, needs_espeak

from pronunciation.assess import FEATURE_NAMES, analyze
from pronunciation.g2p import phonemize_words, tokenize

TEXT = "I think this is the third thing."


def expected_raw():
    return phonemize_words(tokenize(TEXT))


@needs_espeak
def test_perfect_reading():
    raw = expected_raw()
    heard = [p for word in raw for p in word]
    result = analyze(TEXT, raw, fake_recognition(heard), TOKEN_TO_ID, BLANK)
    assert result.phone_accuracy == 1.0
    assert all(w.issues == [] for w in result.words)
    assert result.features["align_ok"] == 1.0
    assert result.features["gop_bad_frac"] == 0.0
    assert list(result.features) == FEATURE_NAMES


@needs_espeak
def test_typical_turkish_accent_th_sounds():
    raw = expected_raw()
    accented = [["t" if p == "θ" else "d" if p == "ð" else p for p in word] for word in raw]
    heard = [p for word in accented for p in word]
    result = analyze(TEXT, raw, fake_recognition(heard), TOKEN_TO_ID, BLANK)
    assert result.tips[:2] == ["th_voiceless", "th_voiced"]
    assert result.phone_accuracy < 1.0
    assert result.features["gop_bad_frac"] > 0.0
    think = result.words[1]
    assert think.text == "think" and think.score < 1.0


@needs_espeak
def test_silence_gives_zero_accuracy_and_no_crash():
    raw = expected_raw()
    result = analyze(TEXT, raw, fake_recognition([], lead_blanks=2), TOKEN_TO_ID, BLANK)
    assert result.phone_accuracy == 0.0
    assert result.features["align_ok"] == 0.0


THINK = [["θ", "ɪ", "ŋ", "k"]]


def near_tie(heard, said, expected, p_said=0.5, p_expected=0.45):
    """The model picked `said`, but `expected` was almost as likely: a probable mishearing."""
    recognition = fake_recognition(heard)
    frames = recognition.log_probs.argmax(axis=1) == TOKEN_TO_ID[said]
    recognition.log_probs[frames, TOKEN_TO_ID[said]] = np.log(p_said)
    recognition.log_probs[frames, TOKEN_TO_ID[expected]] = np.log(p_expected)
    return recognition


def test_gop_dismisses_an_unsure_difference():
    recognition = near_tie(["t", "ɪ", "ŋ", "k"], said="t", expected="θ")
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=-1.0).words
    assert word.issues == []
    assert [i.heard for i in word.dismissed] == ["t"]
    assert -1.0 < word.gop < 0.0


def test_gop_confirms_a_clear_error():
    recognition = fake_recognition(["t", "ɪ", "ŋ", "k"], confidence=0.999)
    result = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=-1.0)
    assert [i.tip for i in result.words[0].issues] == ["th_voiceless"]
    assert result.tips == ["th_voiceless"]
    assert result.words[0].gop < -5


def test_without_a_threshold_every_difference_is_reported():
    recognition = near_tie(["t", "ɪ", "ŋ", "k"], said="t", expected="θ")
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=None).words
    assert [i.heard for i in word.issues] == ["t"] and word.dismissed == []


def test_extra_vowel_is_reported_even_when_gop_is_good():
    # GOP only scores the expected sounds, so it cannot judge an added one
    recognition = fake_recognition(["ɪ", "s", "k", "uː", "l"], confidence=0.999)
    [word] = analyze("school", [["s", "k", "uː", "l"]], recognition, TOKEN_TO_ID, BLANK, gop_threshold=-1.0).words
    assert word.gop == 0.0
    assert [i.tip for i in word.issues] == ["epenthesis"]


def test_extra_sound_without_a_known_pattern_is_dismissed_when_gop_is_good():
    recognition = fake_recognition(["θ", "ɪ", "ŋ", "k", "ə"], confidence=0.999)
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=-1.0).words
    assert word.issues == []
    assert [i.heard for i in word.dismissed] == ["ə"]


def test_gop_confirmation_is_on_by_default():
    recognition = near_tie(["t", "ɪ", "ŋ", "k"], said="t", expected="θ")
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK).words
    assert word.issues == [] and len(word.dismissed) == 1


def test_known_pattern_needs_less_gop_evidence():
    # GOP of θ is log(0.15 / 0.8) = -1.7: not enough for an unknown difference (-2.5),
    # enough for a typical Turkish-speaker error (-1.0)
    recognition = near_tie(["t", "ɪ", "ŋ", "k"], said="t", expected="θ", p_said=0.8, p_expected=0.15)
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=-2.5, pattern_threshold=-1.0).words
    assert [i.tip for i in word.issues] == ["th_voiceless"]
    assert -2.5 < word.gop < -1.0


def test_unknown_difference_needs_full_gop_evidence():
    recognition = near_tie(["k", "ɪ", "ŋ", "k"], said="k", expected="θ", p_said=0.8, p_expected=0.15)
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK, gop_threshold=-2.5, pattern_threshold=-1.0).words
    assert word.issues == [] and [i.heard for i in word.dismissed] == ["k"]


def test_pattern_threshold_is_looser_by_default():
    recognition = near_tie(["t", "ɪ", "ŋ", "k"], said="t", expected="θ", p_said=0.8, p_expected=0.15)
    [word] = analyze("think", THINK, recognition, TOKEN_TO_ID, BLANK).words
    assert [i.tip for i in word.issues] == ["th_voiceless"]
