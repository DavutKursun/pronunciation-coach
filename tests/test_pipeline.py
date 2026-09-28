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
