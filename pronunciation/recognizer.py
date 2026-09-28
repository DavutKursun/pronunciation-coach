"""Phoneme recognition with a pretrained wav2vec2 CTC model."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .ctc import NEG_INF, greedy_decode
from .phonemes import ALLOWED_PHONES

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


@dataclass
class Recognition:
    log_probs: np.ndarray   # [frames, vocab] log-probabilities, one frame = 20 ms
    phones: list[str]       # what the model heard (greedy CTC decoding)
    seconds: float          # audio duration


def pick_device() -> str:
    import torch

    if os.environ.get("PC_DEVICE"):
        return os.environ["PC_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


class Decoder:
    """Turns the model's log-probabilities into a Recognition.

    Kept apart from the model so that cached log-probabilities (scripts/evaluate_words.py)
    and tests go through exactly the same decoding as the app.
    """

    def __init__(self, vocab: dict[str, int], blank_id: int, special_ids: set[int],
                 allowed: set[str] | None = ALLOWED_PHONES):
        self.token_to_id: dict[str, int] = dict(vocab)
        self.id_to_token: dict[int, str] = {i: t for t, i in vocab.items()}
        self.blank_id = blank_id
        self.special_ids = set(special_ids)
        # tokens that may never be output: everything except the blank and the allowed phonemes
        self.blocked = np.array(sorted(i for t, i in vocab.items()
                                       if allowed is not None and i != blank_id and t not in allowed), dtype=int)

    def restrict(self, log_probs: np.ndarray) -> np.ndarray:
        """Give blocked tokens zero probability and renormalize, so GOP uses the same choices."""
        if self.blocked.size == 0:
            return log_probs
        out = log_probs.copy()
        out[:, self.blocked] = NEG_INF
        return out - np.logaddexp.reduce(out, axis=1, keepdims=True)

    def __call__(self, log_probs: np.ndarray, seconds: float) -> Recognition:
        log_probs = self.restrict(log_probs)
        decoded = greedy_decode(log_probs, self.blank_id, self.special_ids)
        phones = [self.id_to_token[token] for token, _ in decoded]
        return Recognition(log_probs=log_probs, phones=phones, seconds=seconds)


class PhonemeRecognizer:
    """Wraps facebook/wav2vec2-lv-60-espeak-cv-ft (IPA phonemes in eSpeak style)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

        self.model_id = model_id
        self.device = device or pick_device()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        # The model's tokenizer config has do_phonemize=True, which starts an eSpeak backend
        # on load and fails when phonemizer cannot find the library (common on macOS).
        # We only use the tokenizer to decode model output; g2p.py does the phonemizing.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, do_phonemize=False)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device).eval()

        blank_id = self.tokenizer.pad_token_id
        special_ids = {i for i in self.tokenizer.all_special_ids if i != blank_id}
        self.decoder = Decoder(self.tokenizer.get_vocab(), blank_id, special_ids)
        self.token_to_id = self.decoder.token_to_id
        self.blank_id = blank_id

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        """Raw [frames, vocab] log-probabilities for mono float32 16 kHz audio."""
        import torch

        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def recognize(self, audio: np.ndarray) -> Recognition:
        """audio: mono float32 at 16 kHz (see audio.to_model_input)."""
        return self.decoder(self.log_probs(audio), seconds=len(audio) / SAMPLE_RATE)
