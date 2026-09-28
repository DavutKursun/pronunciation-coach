import pytest

from pronunciation.speechocean import (ARPABET, arpabet_to_ipa, is_error, transfer_labels)


@pytest.mark.parametrize("arpabet, ipa", [
    ("AA1", "ɑː"), ("AE", "æ"), ("AH0", "ə"), ("AH1", "ʌ"), ("AH", "ʌ"), ("AO2", "ɔː"),
    ("ER0", "ɚ"), ("ER1", "ɜː"), ("IH0", "ɪ"), ("IY1", "iː"), ("UH", "ʊ"), ("UW0", "uː"),
    ("TH", "θ"), ("DH", "ð"), ("NG", "ŋ"), ("R", "ɹ"), ("Y", "j"), ("JH", "dʒ"), ("G", "ɡ"),
])
def test_arpabet_to_ipa(arpabet, ipa):
    assert arpabet_to_ipa(arpabet) == ipa


def test_every_arpabet_phone_maps_to_a_phoneme_the_recognizer_knows():
    from pronunciation.phonemes import ALLOWED_PHONES

    assert len(ARPABET) == 39
    assert all(arpabet_to_ipa(p) in ALLOWED_PHONES for p in list(ARPABET) + ["AH0", "ER0"])


def test_is_error_rounds_the_average_expert_score():
    # five experts score 0/1/2; the dataset gives their average
    assert is_error(0.4, "strict") and not is_error(0.6, "strict")
    assert is_error(1.4, "lenient") and not is_error(1.6, "lenient")
    assert not is_error(2.0, "lenient")
    with pytest.raises(ValueError):
        is_error(1.0, "other")


def test_transfer_labels_same_phonemes():
    labels, stats = transfer_labels(["θ", "iː", "m"], ["TH", "IY0", "M"], [1.2, 0.0, 1.6], function_word=False)
    assert labels == [1.2, 0.0, 1.6]
    assert stats["same"] == 3


def test_transfer_labels_lexicon_differences():
    # eSpeak: "brother" b ɹ ʌ ð ɚ; CMU: B R AH1 DH ER0 -> the same, ER0 = ɚ
    labels, _ = transfer_labels(["b", "ɹ", "ʌ", "ð", "ɚ"], ["B", "R", "AH1", "DH", "ER0"], [2, 2, 1.8, 0.4, 2], False)
    assert labels == [2, 2, 1.8, 0.4, 2]
    # eSpeak "happy" ends in i, CMU in IY0 (iː): an accepted variant, the label moves over
    labels, stats = transfer_labels(["h", "æ", "p", "i"], ["HH", "AE1", "P", "IY0"], [2, 2, 2, 0.8], False)
    assert labels == [2, 2, 2, 0.8] and stats["variant"] == 1


def test_transfer_labels_leaves_unmatched_phonemes_without_a_label():
    # eSpeak has an extra sound the CMU word does not have: no label for it
    labels, stats = transfer_labels(["æ", "n", "d"], ["AE1", "N"], [2.0, 1.0], function_word=True)
    assert labels == [2.0, 1.0, None]
    assert stats["unlabeled"] == 1
    labels, stats = transfer_labels(["ə", "l"], ["L"], [0.0], function_word=False)
    assert labels == [None, 0.0] and stats["unlabeled"] == 1
    # a vowel aligned with a consonant is a lexicon mismatch, not a label we trust
    labels, stats = transfer_labels(["ə"], ["L"], [0.0], function_word=False)
    assert labels == [None] and stats["uncertain"] == 1
