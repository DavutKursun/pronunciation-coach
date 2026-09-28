import numpy as np
from conftest import needs_espeak

from pronunciation import phonemes
from pronunciation.recognizer import Decoder

# a tiny multilingual vocabulary: "ə1" (Mandarin tone) and "tʰ" (aspirated t) are not English
VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "ə": 4, "ə1": 5, "r": 6, "tʰ": 7, "t": 8}


def frames(*rows):
    """Log-probabilities from rows of {token: probability}; the rest shares what is left."""
    out = np.full((len(rows), len(VOCAB)), 0.0)
    for f, row in enumerate(rows):
        rest = (1 - sum(row.values())) / (len(VOCAB) - len(row))
        out[f] = rest
        for token, p in row.items():
            out[f, VOCAB[token]] = p
    return np.log(out)


def decoder():
    return Decoder(VOCAB, blank_id=0, special_ids={1, 2, 3})


def test_non_english_tokens_fall_back_to_the_best_english_one():
    log_probs = frames({"ə1": 0.6, "ə": 0.3}, {"<pad>": 0.9}, {"tʰ": 0.5, "t": 0.4})
    assert decoder()(log_probs, seconds=0.06).phones == ["ə", "t"]


def test_typical_turkish_sounds_are_still_recognized():
    log_probs = frames({"r": 0.8}, {"<pad>": 0.9})
    assert decoder()(log_probs, seconds=0.04).phones == ["r"]


def test_restricted_log_probs_are_normalized_for_gop():
    raw = frames({"ə1": 0.6, "ə": 0.3})
    probs = np.exp(decoder()(raw, seconds=0.02).log_probs[0])
    assert probs[VOCAB["ə1"]] == 0.0 and probs[VOCAB["tʰ"]] == 0.0
    assert np.isclose(probs.sum(), 1.0)
    # the allowed tokens keep their relative sizes
    assert np.isclose(probs[VOCAB["ə"]] / probs[VOCAB["t"]], 0.3 / np.exp(raw[0, VOCAB["t"]]))


def test_allowed_phones_cover_tip_patterns_and_variants():
    heard_in_tips = {heard for _, heard in phonemes.SUBSTITUTION_TIPS}
    accepted = {p for variants in phonemes.ACCEPT.values() for p in variants}
    assert heard_in_tips | accepted <= phonemes.ALLOWED_PHONES
    assert {"r", "ɾ", "e", "a", "ø", "œ", "ɯ"} <= phonemes.ALLOWED_PHONES
    assert not {"ə1", "tʰ", "uɨ"} & phonemes.ALLOWED_PHONES


@needs_espeak
def test_allowed_phones_cover_everything_espeak_says_in_the_practice_sentences():
    import json
    from pathlib import Path

    from pronunciation.g2p import phonemize_words, tokenize

    sentences = json.loads((Path(__file__).parents[1] / "data" / "sentences.json").read_text())
    said = {p for s in sentences for word in phonemize_words(tokenize(s["text"])) for p in word}
    assert said <= phonemes.ALLOWED_PHONES
