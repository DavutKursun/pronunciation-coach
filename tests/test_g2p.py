from conftest import needs_espeak

from pronunciation.g2p import phonemize_words, tokenize


def test_tokenize_keeps_contractions_and_drops_punctuation():
    assert tokenize("Don't stop, it's 2 o’clock!") == ["Don't", "stop", "it's", "2", "o'clock"]


@needs_espeak
def test_phonemize_words():
    assert phonemize_words(["think", "Ship", "car"]) == [["θ", "ɪ", "ŋ", "k"], ["ʃ", "ɪ", "p"], ["k", "ɑːɹ"]]
