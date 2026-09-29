"""The full pipeline: audio + target sentence -> per-word feedback, features and a score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import align
from .ctc import forced_align, gop_scores
from .feedback import (Rival, WordResult, build_word_results, confirm_with_gop, final_cluster_start, pattern_rivals,
                       report_hidden, report_hidden_w, top_tips)
from .g2p import phonemize_words, tokenize
from .phonemes import FUNCTION_WORDS, SPLITS, WORD_VARIANTS, merge_repeats, normalize

# Order matters: the trained scorer expects exactly these columns.
FEATURE_NAMES = [
    "phone_accuracy", "sub_rate", "del_rate", "ins_rate",
    "word_perfect_frac", "word_score_mean", "word_score_min",
    "gop_lpp_mean", "gop_lpp_min", "gop_lpr_mean", "gop_lpr_min", "gop_lpr_p10", "gop_bad_frac",
    "align_ok", "n_expected", "duration_s", "heard_per_second",
]
GOP_FLOOR = -20.0      # used when forced alignment is impossible (e.g. almost silent audio)
GOP_BAD = -2.0         # lpr below this counts as a badly pronounced phoneme
W_RIVALS = ("v", "β", "ʋ")   # how a Turkish speaker's w can sound: v, a bilabial β, a labiodental ʋ
# A word's differences are reported only if its GOP is below this. Chosen on speechocean762
# train (scripts/evaluate_words.py --sweep): fewer false alarms without losing detected errors.
GOP_CONFIRM: float | None = -2.5
# Typical Turkish-speaker errors need less evidence: learners are likely to make them. Chosen on
# the Speech Accent Archive dev half, checked on speechocean762 train and the synthetic set.
GOP_CONFIRM_PATTERN: float | None = -1.0


@dataclass
class GopAlignment:
    """Forced alignment of the expected sounds (see expected_targets) and their GOP scores."""
    target_ids: list[int]
    target_word: list[int]
    phone_target: list[int | None]
    spans: list[tuple[int, int]] | None      # None when the audio is too short for the expected sounds
    lpp: np.ndarray | None
    lpr: np.ndarray | None


@dataclass
class Assessment:
    text: str
    words: list[WordResult]
    heard_phones: list[str]
    phone_accuracy: float
    features: dict[str, float]
    tips: list[str] = field(default_factory=list)
    score: float | None = None   # 0-10, predicted by the trained scorer (None without one)
    alignment: GopAlignment | None = None


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
    variants = [WORD_VARIANTS.get(w.lower(), {}).get(p, set()) for w, phones in zip(words, word_phones) for p in phones]
    ops = align(flat, heard, function_flags, variants)
    return build_word_results(words, word_phones, ops, exp_word)


def expected_targets(raw_word_phones: list[list[str]], token_to_id: dict[str, int]):
    """Model tokens to force-align, the word of each token, and the token behind every expected phoneme.

    Returns (target_ids, target_word, phone_target): target_ids are phones_to_ids() word by word;
    phone_target[k] is the index in target_ids of the token that covers the k-th expected phoneme
    of prepare_expected() (None if no token does). One token can cover two phonemes: "store" is
    s t ɔːɹ for the model but s t oː ɹ for the comparison, and both oː and ɹ get the ɔːɹ token.
    """
    target_ids, target_word, phone_target = [], [], []
    for w, phones in enumerate(raw_word_phones):
        parts: list[tuple[str, int | None]] = []
        for p in phones:
            if p in token_to_id:
                target_ids.append(token_to_id[p])
                target_word.append(w)
                parts += [(q, len(target_ids) - 1) for q in normalize([p])]
                continue
            for q in normalize([p]):              # the same splitting as phones_to_ids
                if q in SPLITS.get(p, []) and q in token_to_id:
                    target_ids.append(token_to_id[q])
                    target_word.append(w)
                    parts.append((q, len(target_ids) - 1))
                else:
                    parts.append((q, None))
        # like merge_repeats() in prepare_expected: a repeated phoneme keeps the first one's token
        phone_target += [t for k, (q, t) in enumerate(parts) if k == 0 or parts[k - 1][0] != q]
    return target_ids, target_word, phone_target


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


def rival_margin(log_probs: np.ndarray, span: tuple[int, int], target_id: int, rival_ids: list[int]) -> tuple[float, int]:
    """The largest log p(rival) - log p(target) over the span's frames and the rivals, and which rival it was."""
    start, end = span
    frames = log_probs[start:end + 1]
    margins = frames[:, rival_ids] - frames[:, [target_id]]
    frame, rival = np.unravel_index(margins.argmax(), margins.shape)
    return float(margins[frame, rival]), int(rival)


def measure_hidden_w(word_results: list[WordResult], alignment: "GopAlignment", log_probs: np.ndarray,
                     token_to_id: dict[str, int]) -> None:
    """For every expected w the recognizer heard as w: the largest log p(v/β/ʋ) - log p(w) in its frames."""
    rivals = [c for c in W_RIVALS if c in token_to_id]
    if not rivals or "w" not in token_to_id or alignment.spans is None:
        return
    rival_ids, w_id = [token_to_id[c] for c in rivals], token_to_id["w"]
    for word in word_results:
        for op in word.ops:
            target = alignment.phone_target[op.exp_pos] if op.exp_pos is not None else None
            if op.expected != "w" or op.kind != "match" or target is None:
                continue
            margin, rival = rival_margin(log_probs, alignment.spans[target], w_id, rival_ids)
            if word.w_margin is None or margin > word.w_margin:
                word.w_margin, word.w_rival, word.w_pos = margin, rivals[rival], op.exp_pos


def measure_hidden(word_results: list[WordResult], alignment: "GopAlignment", log_probs: np.ndarray,
                   token_to_id: dict[str, int]) -> None:
    """Fill word.rivals: for every sound the recognizer wrote exactly as expected and every Turkish-speaker
    pattern of that sound (feedback.pattern_rivals), how close the pattern's sounds came to it."""
    if alignment.spans is None:
        return
    first = 0
    for word in word_results:
        final_start = final_cluster_start(word.expected)
        function_word = word.text.lower() in FUNCTION_WORDS
        variants = WORD_VARIANTS.get(word.text.lower(), {})
        for op in word.ops:
            target = alignment.phone_target[op.exp_pos] if op.exp_pos is not None else None
            if op.kind != "match" or op.heard != op.expected or target is None:
                continue
            patterns = pattern_rivals(op.expected, op.exp_pos - first >= final_start, function_word,
                                      variants.get(op.expected, set()))
            for tip, sounds in patterns.items():
                sounds = [s for s in sounds if s in token_to_id]
                if sounds:
                    margin, k = rival_margin(log_probs, alignment.spans[target], alignment.target_ids[target],
                                             [token_to_id[s] for s in sounds])
                    word.rivals.append(Rival(op.exp_pos, op.expected, tip, sounds[k], margin))
        first += len(word.expected)


def analyze(text: str, raw_word_phones: list[list[str]], recognition, token_to_id: dict[str, int], blank_id: int,
            gop_threshold: float | None = GOP_CONFIRM,
            pattern_threshold: float | None = GOP_CONFIRM_PATTERN,
            w_margin_threshold: float | None = None,
            hidden_thresholds: dict[str, float | None] | None = None) -> Assessment:
    """Everything after recognition. Kept separate from the model so it can be unit-tested.

    Hidden errors: `w_margin_threshold` is the hidden-w rule of v2-3c, `hidden_thresholds` its
    generalization to every pattern (v2-3d, {tip: margin threshold or None = off}); one at a time.
    """
    if w_margin_threshold is not None and hidden_thresholds:
        raise ValueError("use one hidden-error rule: w_margin_threshold (v2-3c) or hidden_thresholds (v2-3d)")
    words = tokenize(text)
    heard = normalize(recognition.phones)
    word_results = compare(words, raw_word_phones, heard)
    n_expected = sum(len(w.expected) for w in word_results)

    # GOP: force-align the expected sounds (as model tokens) to the audio, remembering their word
    target_ids, target_word, phone_target = expected_targets(raw_word_phones, token_to_id)
    spans = forced_align(recognition.log_probs, target_ids, blank_id) if target_ids else None
    alignment = GopAlignment(target_ids, target_word, phone_target, spans, None, None)
    if spans is not None:
        alignment.lpp, alignment.lpr = lpp, lpr = gop_scores(recognition.log_probs, target_ids, spans, blank_id)
        target_word = np.array(target_word)
        for w, word in enumerate(word_results):
            word_lpr = lpr[target_word == w]
            word.gop = float(word_lpr.min()) if word_lpr.size else None
            confirm_with_gop(word, gop_threshold, pattern_threshold)
    else:
        lpp = lpr = np.array([GOP_FLOOR])

    features = compute_features(word_results, n_expected, heard, lpp, lpr, spans is not None, recognition.seconds)
    # after the scoring features, like the GOP check: feedback rules do not change what the scorer sees
    measure_hidden_w(word_results, alignment, recognition.log_probs, token_to_id)
    measure_hidden(word_results, alignment, recognition.log_probs, token_to_id)
    for w, word in enumerate(word_results):
        report_hidden_w(word, w, w_margin_threshold)
        report_hidden(word, w, hidden_thresholds)
    return Assessment(
        text=text,
        words=word_results,
        heard_phones=heard,
        phone_accuracy=features["phone_accuracy"],
        features=features,
        tips=top_tips(word_results),
        alignment=alignment,
    )


class PronunciationCoach:
    """Load once, then call assess() for every recording.

    Uses the learned error detector (v2) when models/detector.joblib exists and was trained on
    this recognizer's output; otherwise the v1 rules.
    """

    def __init__(self, recognizer=None, scorer_path: str | Path | None = "models/scorer.joblib",
                 detector_path: str | Path | None = "models/detector.joblib"):
        if recognizer is None:
            from .recognizer import PhonemeRecognizer

            recognizer = PhonemeRecognizer()
        self.recognizer = recognizer
        self.scorer = None
        if scorer_path and Path(scorer_path).exists():
            import joblib

            self.scorer = joblib.load(scorer_path)
        self.detector = None
        if detector_path:
            from .detector import detector_for

            self.detector = detector_for(detector_path, getattr(recognizer, "model_id", None))

    def assess(self, audio: np.ndarray, text: str) -> Assessment:
        """audio: mono float32 at 16 kHz."""
        words = tokenize(text)
        if not words:
            raise ValueError("The sentence has no words.")
        raw_word_phones = phonemize_words(words)
        recognition = self.recognizer.recognize(audio)
        decide = self.detector.assess if self.detector else analyze
        result = decide(text, raw_word_phones, recognition, self.recognizer.token_to_id, self.recognizer.blank_id)
        if self.scorer is not None:
            row = [[result.features[name] for name in self.scorer["features"]]]
            result.score = float(np.clip(self.scorer["model"].predict(row)[0], 0, 10))
        return result
