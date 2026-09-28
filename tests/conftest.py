import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pronunciation.recognizer import Recognition  # noqa: E402

PHONES = ("n s t ə l i k d m ɛ ɾ e ɪ p o ɐ z ð f j v b ɹ ʊ iː r w ʌ u ɡ æ aɪ ʃ h ɔ ɑː ŋ ɚ eɪ uː "
          "oʊ ᵻ θ aʊ dʒ əl ɜː ʒ tʃ ɔː ɑːɹ ɔːɹ oːɹ ɛɹ ɪɹ ʊɹ aɪɚ n̩ ʔ").split()
VOCAB = ["<pad>", "<s>", "</s>", "<unk>"] + PHONES
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
BLANK = 0


def espeak_available() -> bool:
    try:
        from pronunciation.g2p import phonemize_words

        return phonemize_words(["think"]) == [["θ", "ɪ", "ŋ", "k"]]
    except Exception:
        return False


needs_espeak = pytest.mark.skipif(not espeak_available(), reason="eSpeak NG / phonemizer not installed")


def fake_recognition(phones, frames_per_phone=2, confidence=0.95, lead_blanks=3):
    """Log-probabilities a CTC model would produce if it heard exactly `phones`."""
    frames = [BLANK] * lead_blanks
    for p in phones:
        frames += [TOKEN_TO_ID[p]] * frames_per_phone + [BLANK]
    frames += [BLANK] * lead_blanks
    vocab_size = len(VOCAB)
    log_probs = np.full((len(frames), vocab_size), np.log((1 - confidence) / (vocab_size - 1)))
    for t, token in enumerate(frames):
        log_probs[t, token] = np.log(confidence)
    return Recognition(log_probs=log_probs, phones=list(phones), seconds=len(frames) * 0.02)


@pytest.fixture
def vocab():
    return TOKEN_TO_ID
