"""The full pipeline: audio + target sentence -> per-word feedback, features and a score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import align
from .ctc import forced_align, gop_scores
from .feedback import WordResult, build_word_results, confirm_with_gop, top_tips
from .g2p import phonemize_words, tokenize
from .phonemes import FUNCTION_WORDS, SPLITS, WEAK_FORMS, merge_repeats, normalize

# Order matters: the trained scorer expects exactly these columns.
FEATURE_NAMES = [
    "phone_accuracy", "sub_rate", "del_rate", "ins_rate",
    "word_perfect_frac", "word_score_mean", "word_score_min",
    "gop_lpp_mean", "gop_lpp_min", "gop_lpr_mean", "gop_lpr_min", "gop_lpr_p10", "gop_bad_frac",
    "align_ok", "n_expected", "duration_s", "heard_per_second",
]
GOP_FLOOR = -20.0      # used when forced alignment is impossible (e.g. almost silent audio)
GOP_BAD = -2.0         # lpr below this counts as a badly pronounced phoneme
# A word's differences are reported only if its GOP is below this. Chosen on speechocean762
# train (scripts/evaluate_words.py --sweep): fewer false alarms without losing detected errors.
GOP_CONFIRM: float | None = -2.5
# Typical Turkish-speaker errors need less evidence: learners are likely to make them. Chosen on
# the Speech Accent Archive dev half, checked on speechocean762 train and the synthetic set.
GOP_CONFIRM_PATTERN: float | None = -1.0


@dataclass
class Assessment:
    text: str
    words: list[WordResult]
    heard_phones: list[str]
    phone_accuracy: float
    features: dict[str, float]
    tips: list[str] = field(default_factory=list)
    score: float | None = None   # 0-10, predicted by the trained scorer (None without one)


def prepare_expected(words: list[str], raw_word_phones: list[list[str]]):
    """Flatten per-word phonemes, remembering which word each phoneme came from."""
    word_phones = [merge_repeats(normalize(p)) for p in raw_word_phones]
    flat, exp_word, function_flags = [], [], []
    for w, (text, phones) in enumerate(zip(words, word_phones)):
        is_function = text.lower() in FUNCTION_WORDS
        for p in phones:
            flat.append(p)
            exp_word.append(w)
            function_flags.append(is_function)
    return word_phones, flat, exp_word, function_flags


def compare(words: list[str], raw_word_phones: list[list[str]], heard: list[str]) -> list[WordResult]:
    """Align what was heard with the expected phonemes and turn the differences into per-word results."""
    word_phones, flat, exp_word, function_flags = prepare_expected(words, raw_word_phones)
    variants = [WEAK_FORMS.get(w.lower(), {}).get(p, set()) for w, phones in zip(words, word_phones) for p in phones]
    ops = align(flat, heard, function_flags, variants)
    return build_word_results(words, word_phones, ops, exp_word)


def phones_to_ids(phones: list[str], token_to_id: dict[str, int]) -> list[int]:
    """Map expected phonemes to model token ids (splitting or skipping unknown ones)."""
    ids = []
    for p in phones:
        if p in token_to_id:
            ids.append(token_to_id[p])
        else:
            ids.extend(token_to_id[q] for q in SPLITS.get(p, []) if q in token_to_id)
    return ids


def compute_features(
    word_results: list[WordResult], n_expected: int, heard: list[str],
    lpp: np.ndarray, lpr: np.ndarray, align_ok: bool, seconds: float,
) -> dict[str, float]:
    ops = [op for w in word_results for op in w.ops]
    matches = sum(op.kind == "match" for op in ops)
    subs = sum(op.kind == "sub" for op in ops)
    dels = sum(op.kind == "del" for op in ops)
    ins = sum(op.kind == "ins" for op in ops)
    n = max(n_expected, 1)
    word_scores = [w.score for w in word_results] or [0.0]

    return {
        "phone_accuracy": matches / (n_expected + ins) if n_expected + ins else 0.0,
        "sub_rate": subs / n,
        "del_rate": dels / n,
        "ins_rate": ins / n,
        "word_perfect_frac": float(np.mean([s == 1.0 for s in word_scores])),
        "word_score_mean": float(np.mean(word_scores)),
        "word_score_min": float(np.min(word_scores)),
        "gop_lpp_mean": float(lpp.mean()),
        "gop_lpp_min": float(lpp.min()),
        "gop_lpr_mean": float(lpr.mean()),
        "gop_lpr_min": float(lpr.min()),
        "gop_lpr_p10": float(np.percentile(lpr, 10)),
        "gop_bad_frac": float(np.mean(lpr < GOP_BAD)),
        "align_ok": float(align_ok),
        "n_expected": float(n_expected),
        "duration_s": float(seconds),
        "heard_per_second": len(heard) / seconds if seconds > 0 else 0.0,
    }


def analyze(text: str, raw_word_phones: list[list[str]], recognition, token_to_id: dict[str, int], blank_id: int,
            gop_threshold: float | None = GOP_CONFIRM,
            pattern_threshold: float | None = GOP_CONFIRM_PATTERN) -> Assessment:
    """Everything after recognition. Kept separate from the model so it can be unit-tested."""
    words = tokenize(text)
    heard = normalize(recognition.phones)
    word_results = compare(words, raw_word_phones, heard)
    n_expected = sum(len(w.expected) for w in word_results)

    # GOP: force-align the expected sounds (as model tokens) to the audio, remembering their word
    target_ids, target_word = [], []
    for w, phones in enumerate(raw_word_phones):
        ids = phones_to_ids(phones, token_to_id)
        target_ids += ids
        target_word += [w] * len(ids)
    spans = forced_align(recognition.log_probs, target_ids, blank_id) if target_ids else None
    if spans is not None:
        lpp, lpr = gop_scores(recognition.log_probs, target_ids, spans, blank_id)
        target_word = np.array(target_word)
        for w, word in enumerate(word_results):
            word_lpr = lpr[target_word == w]
            word.gop = float(word_lpr.min()) if word_lpr.size else None
            confirm_with_gop(word, gop_threshold, pattern_threshold)
    else:
        lpp = lpr = np.array([GOP_FLOOR])

    features = compute_features(word_results, n_expected, heard, lpp, lpr, spans is not None, recognition.seconds)
    return Assessment(
        text=text,
        words=word_results,
        heard_phones=heard,
        phone_accuracy=features["phone_accuracy"],
        features=features,
        tips=top_tips(word_results),
    )


class PronunciationCoach:
    """Load once, then call assess() for every recording."""

    def __init__(self, recognizer=None, scorer_path: str | Path | None = "models/scorer.joblib"):
        if recognizer is None:
            from .recognizer import PhonemeRecognizer

            recognizer = PhonemeRecognizer()
        self.recognizer = recognizer
        self.scorer = None
        if scorer_path and Path(scorer_path).exists():
            import joblib

            self.scorer = joblib.load(scorer_path)

    def assess(self, audio: np.ndarray, text: str) -> Assessment:
        """audio: mono float32 at 16 kHz."""
        words = tokenize(text)
        if not words:
            raise ValueError("The sentence has no words.")
        raw_word_phones = phonemize_words(words)
        recognition = self.recognizer.recognize(audio)
        result = analyze(text, raw_word_phones, recognition,
                         self.recognizer.token_to_id, self.recognizer.blank_id)
        if self.scorer is not None:
            row = [[result.features[name] for name in self.scorer["features"]]]
            result.score = float(np.clip(self.scorer["model"].predict(row)[0], 0, 10))
        return result
