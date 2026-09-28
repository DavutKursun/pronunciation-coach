"""v2: a learned error detector. For every expected sound: how likely is it that it was said wrong?

v1 decides with hand-picked rules: a difference in the greedy transcription, confirmed by a GOP
threshold. The detector instead looks at many signals per expected sound and learns from the
speechocean762 expert scores how they combine (scripts/train_detector.py):

  GOP            lpp, lpr, and both normalized per sound (the recognizer is less sure of some
                 sounds even when they are said well)
  competitor     in the frames of the sound: log p(typical Turkish replacement) - log p(expected),
                 e.g. s for a word-final z, whatever the greedy transcription says
  alignment      was the sound matched, replaced (by a known pattern?) or missing in the transcription
  sound class    vowel, stop, fricative, affricate, nasal, liquid, glide; voiced or not
                 (the exact sound is not a feature: which sounds speechocean's Mandarin speakers get
                 wrong need not carry over to Turkish speakers)
  position       first/last in the word, in the final consonant group, function word
  timing         duration, duration of the vowel before it (a vowel is longer before a voiced final
                 consonant). CTC puts each sound in one or two frames and fills the rest with blanks,
                 so a sound's duration is the time from its first frame to the next sound's first frame
  context        GOP of the neighbouring sounds and the worst GOP of the word

The speaking rate is left out on purpose: in speechocean762 it mostly tells slow (often child)
speakers apart and measures pauses (median 4.8 sounds per second there, 9.8 in the Speech Accent
Archive paragraph), so it is a speaker-level shortcut that does not carry over to Turkish adults.

A word gets the highest error probability of its sounds, and two thresholds turn it into
"red" (sure error) or "yellow" (possible error). Added sounds (the extra vowel in "is-chool",
a g after ŋ) belong to no expected sound, so they keep the v1 rules.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .align import Op
from .assess import GOP_CONFIRM, Assessment, analyze, prepare_expected
from .feedback import WordResult, find_issues, top_tips
from .g2p import tokenize
from .phonemes import DEVOICED, SUBSTITUTION_TIPS, is_vowel

FRAME_SECONDS = 0.02
STOPS = {"p", "b", "t", "d", "k", "ɡ", "ʔ", "ɾ"}
FRICATIVES = {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "x", "ç", "ɣ", "β"}
AFFRICATES = {"tʃ", "dʒ"}
NASALS = {"m", "n", "ŋ", "n̩"}
LIQUIDS = {"l", "ɹ", "r", "ɫ", "l̩"}
GLIDES = {"w", "j"}
VOICED_CONSONANTS = {"b", "d", "ɡ", "v", "ð", "z", "ʒ", "dʒ", "m", "n", "ŋ", "n̩", "l", "l̩", "ɹ", "r", "ɾ", "w",
                     "j", "β", "ɣ"}

FEATURES = [
    "lpp", "lpr", "lpp_z", "lpr_z",
    "competitor_margin_max", "competitor_margin_mean", "has_competitor",
    "op_sub", "op_del", "op_pattern_sub",
    "is_vowel", "is_stop", "is_fricative", "is_affricate", "is_nasal", "is_liquid", "is_glide", "is_voiced",
    "pos_initial", "pos_final", "in_final_cluster", "relative_position", "word_length", "function_word",
    "duration", "duration_rel", "prev_vowel_duration", "prev_vowel_duration_rel",
    "prev_lpr", "next_lpr", "word_min_lpr",
]
NAN = float("nan")


def competitors(phone: str, word_final: bool) -> set[str]:
    """Typical Turkish-speaker replacements of a sound: the pattern table, plus devoicing at the end of a word."""
    out = {heard for expected, heard in SUBSTITUTION_TIPS if expected == phone}
    if word_final and phone in DEVOICED:
        out.add(DEVOICED[phone])
    return out


def is_pattern_substitution(expected: str, heard: str | None, word_final: bool) -> bool:
    return heard is not None and heard in competitors(expected, word_final)


def phone_rows(words: list[str], raw_word_phones: list[list[str]], result: Assessment, recognition,
               token_to_id: dict[str, int], blank_id: int) -> list[dict]:
    """One row per expected sound (in prepare_expected order): features plus what the tips need."""
    word_phones, flat, exp_word, function_flags = prepare_expected(words, raw_word_phones)
    al = result.alignment
    log_probs = recognition.log_probs
    id_to_token = {i: t for t, i in token_to_id.items()}
    ops = {op.exp_pos: op for w in result.words for op in w.ops if op.exp_pos is not None}
    starts = np.cumsum([0] + [len(p) for p in word_phones])

    def span(k):
        t = al.phone_target[k] if al is not None and al.spans is not None else None
        return (t, *al.spans[t]) if t is not None else (None, None, None)

    def lpr_of(k):
        t = span(k)[0]
        return float(al.lpr[t]) if t is not None else NAN

    # duration of each model token: from its first frame to the next token's first frame
    token_seconds = []
    if al is not None and al.spans is not None:
        starts_t = [start for start, _ in al.spans]
        token_seconds = [((starts_t[t + 1] if t + 1 < len(starts_t) else end + 1) - start) * FRAME_SECONDS
                         for t, (start, end) in enumerate(al.spans)]
    mean_duration = float(np.mean(token_seconds)) if token_seconds else NAN

    rows = []
    for k, phone in enumerate(flat):
        w = exp_word[k]
        local, n = k - starts[w], len(word_phones[w])
        vowels = [i for i, p in enumerate(word_phones[w]) if is_vowel(p)]
        final_cluster = local > vowels[-1] if vowels else True
        op = ops.get(k)
        t, start, end = span(k)
        row = {
            "word": w, "phone": phone, "op": op.kind if op else "del", "heard": op.heard if op else None,
            "word_final": final_cluster, "top_other": None,
            "lpp": NAN, "lpr": NAN, "competitor_margin_max": NAN, "competitor_margin_mean": NAN,
            "duration": NAN, "prev_vowel_duration": NAN,
        }
        if t is not None:
            frames = log_probs[start:end + 1]
            expected_id = al.target_ids[t]
            row["lpp"], row["lpr"] = float(al.lpp[t]), float(al.lpr[t])
            row["duration"] = token_seconds[t]
            margins = {c: frames[:, token_to_id[c]] - frames[:, expected_id]
                       for c in competitors(phone, final_cluster) if c in token_to_id}
            margins = {c: m for c, m in margins.items() if np.isfinite(m).all()}
            if margins:
                best = max(margins, key=lambda c: margins[c].max())
                row["competitor_margin_max"] = float(margins[best].max())
                row["competitor_margin_mean"] = float(margins[best].mean())
            others = frames.mean(axis=0)
            others[[blank_id, expected_id]] = -np.inf
            if np.isfinite(others.max()):
                row["top_other"] = id_to_token[int(others.argmax())]
            if local > 0 and is_vowel(flat[k - 1]) and span(k - 1)[0] is not None:
                row["prev_vowel_duration"] = token_seconds[span(k - 1)[0]]
        word_lprs = [lpr_of(i) for i in range(starts[w], starts[w + 1])]
        row.update({
            "has_competitor": float(bool(competitors(phone, final_cluster))),
            "op_sub": float(row["op"] == "sub"), "op_del": float(row["op"] == "del"),
            "op_pattern_sub": float(row["op"] == "sub" and is_pattern_substitution(phone, row["heard"], final_cluster)),
            "is_vowel": float(is_vowel(phone)), "is_stop": float(phone in STOPS),
            "is_fricative": float(phone in FRICATIVES), "is_affricate": float(phone in AFFRICATES),
            "is_nasal": float(phone in NASALS), "is_liquid": float(phone in LIQUIDS), "is_glide": float(phone in GLIDES),
            "is_voiced": float(is_vowel(phone) or phone in VOICED_CONSONANTS),
            "pos_initial": float(local == 0), "pos_final": float(local == n - 1),
            "in_final_cluster": float(final_cluster), "relative_position": local / max(n - 1, 1),
            "word_length": float(n), "function_word": float(function_flags[k]),
            "duration_rel": row["duration"] / mean_duration,
            "prev_vowel_duration_rel": row["prev_vowel_duration"] / mean_duration,
            "prev_lpr": lpr_of(k - 1) if k > 0 else NAN,
            "next_lpr": lpr_of(k + 1) if k + 1 < len(flat) else NAN,
            "word_min_lpr": float(np.nanmin(word_lprs)) if not np.isnan(word_lprs).all() else NAN,
        })
        rows.append(row)
    return rows


def gop_stats(rows: list[dict], correct: list[bool], min_std: float = 0.25) -> dict[str, list[float]]:
    """Mean and std of lpp and lpr per sound over correctly said examples (for lpp_z, lpr_z)."""
    def stats(values):
        values = np.array(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return [NAN, NAN]
        return [float(values.mean()), max(float(values.std()), min_std)]

    by_phone: dict[str, list[dict]] = {}
    for row, ok in zip(rows, correct):
        if ok:
            by_phone.setdefault(row["phone"], []).append(row)
    everyone = [r for r, ok in zip(rows, correct) if ok]
    table = {"__all__": stats([r["lpp"] for r in everyone]) + stats([r["lpr"] for r in everyone])}
    for phone, group in by_phone.items():
        if len(group) >= 20:
            table[phone] = stats([r["lpp"] for r in group]) + stats([r["lpr"] for r in group])
    return table


def feature_matrix(rows: list[dict], stats: dict[str, list[float]], features: list[str] = FEATURES) -> np.ndarray:
    matrix = np.full((len(rows), len(features)), NAN)
    for i, row in enumerate(rows):
        mean_lpp, std_lpp, mean_lpr, std_lpr = stats.get(row["phone"], stats["__all__"])
        values = {**row, "lpp_z": (row["lpp"] - mean_lpp) / std_lpp, "lpr_z": (row["lpr"] - mean_lpr) / std_lpr}
        matrix[i] = [values[f] for f in features]
    return matrix


class Detector:
    """A trained detector: the model, its feature list, GOP statistics, recognizer and thresholds."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.stats = bundle["gop_stats"]
        self.recognizer = bundle["recognizer"]
        self.thresholds = bundle.get("thresholds") or {"red": 1.01, "yellow": 1.01}

    def probabilities(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        return self.model.predict_proba(feature_matrix(rows, self.stats, self.features))[:, 1]

    def assess(self, text: str, raw_word_phones: list[list[str]], recognition, token_to_id: dict[str, int],
               blank_id: int, thresholds: dict | None = None) -> Assessment:
        red, yellow = (thresholds or self.thresholds)["red"], (thresholds or self.thresholds)["yellow"]
        result = analyze(text, raw_word_phones, recognition, token_to_id, blank_id)   # alignment, GOP, v1 features
        rows = phone_rows(tokenize(text), raw_word_phones, result, recognition, token_to_id, blank_id)
        probs = self.probabilities(rows)
        first = 0
        for w, word in enumerate(result.words):
            k_range = range(first, first + len(word.expected))
            word.error_prob = float(probs[list(k_range)].max()) if len(k_range) else 0.0
            ops = []
            for op in word.ops:
                if op.exp_pos is None:                                  # added sound: v1 rules below
                    ops.append(op)
                elif probs[op.exp_pos] >= yellow:
                    heard = op.heard
                    if op.kind == "match":   # the transcription heard it right: take the strongest rival in its frames
                        heard = rows[op.exp_pos]["top_other"]
                    ops.append(Op("sub", op.expected, heard, op.exp_pos, op.heard_pos) if heard else
                               Op("del", op.expected, None, op.exp_pos, None))
                else:
                    ops.append(replace(op, kind="match"))
            issues = find_issues(w, WordResult(word.text, word.expected, ops), first)
            word.issues = [i for i in issues if i.kind != "ins" or i.tip or
                           (word.gop is not None and word.gop < GOP_CONFIRM)]
            word.dismissed = []
            if word.error_prob >= red:
                word.level = "red"
            elif word.issues:
                word.level = "yellow"
            first += len(word.expected)
        result.tips = top_tips(result.words)
        return result


@lru_cache(maxsize=4)
def load_detector(path: str) -> Detector:
    import joblib

    return Detector(joblib.load(path))


def detector_for(path: str | Path, recognizer_id: str | None) -> Detector | None:
    """The detector at `path` if it exists and was trained on this recognizer's output, else None (use v1)."""
    if not Path(path).exists() or recognizer_id is None:
        return None
    detector = load_detector(str(path))
    return detector if detector.recognizer == recognizer_id else None


def _precision(wrong: np.ndarray, flagged: np.ndarray) -> float:
    return (wrong & flagged).sum() / flagged.sum() if flagged.sum() else 0.0


def _false_alarm(wrong: np.ndarray, flagged: np.ndarray) -> float:
    return (~wrong & flagged).sum() / (~wrong).sum() if (~wrong).sum() else 0.0


def choose_thresholds(turkish: dict, english: dict, red_precision: float = 0.85, min_precision: float = 0.70,
                      max_english_false_alarm: float = 0.02) -> dict[str, float]:
    """Word-level thresholds on the error probability.

    turkish / english: arrays "p" (word error probability), "wrong" (expert label) and "extra"
    (flagged anyway by the v1 rules for added sounds). Red: the lowest threshold whose flagged
    Turkish words have at least `red_precision` precision, while native speakers get at most
    `max_english_false_alarm` false alarms. Yellow: the lowest threshold (the most recall) at or
    below red where red + yellow keep `min_precision` on Turkish speakers and the same native
    limit; if there is none, there is no yellow band (yellow = red).
    """
    tp, tw, tx = (np.asarray(turkish[k]) for k in ("p", "wrong", "extra"))
    ep, ew, ex = (np.asarray(english[k]) for k in ("p", "wrong", "extra"))
    tw, tx, ew, ex = tw.astype(bool), tx.astype(bool), ew.astype(bool), ex.astype(bool)
    candidates = sorted(set(np.concatenate([tp, ep]).tolist())) + [1.01]

    red = next((t for t in candidates if (tp >= t).any() and _precision(tw, tp >= t) >= red_precision
                and _false_alarm(ew, (ep >= t) | ex) <= max_english_false_alarm), 1.01)
    yellow = next((t for t in candidates if t <= red
                   and _precision(tw, (tp >= t) | tx) >= min_precision
                   and _false_alarm(ew, (ep >= t) | ex) <= max_english_false_alarm), red)
    return {"red": float(red), "yellow": float(yellow)}


def cross_validate_thresholds(speaker: list[str], group: list[str], p: list[float], wrong: list[bool],
                              extra: list[bool], n_folds: int = 6, **constraints) -> dict:
    """Honest estimate: choose the thresholds on some speakers, measure on the others, fold by fold."""
    speaker, group = np.asarray(speaker), np.asarray(group)
    p, wrong, extra = np.asarray(p, dtype=float), np.asarray(wrong, dtype=bool), np.asarray(extra, dtype=bool)
    folds = {}
    for g in ("turkish", "english"):
        for i, s in enumerate(sorted(set(speaker[group == g]))):
            folds[s] = i % n_folds
    fold = np.array([folds[s] for s in speaker])
    flagged, red_flagged, chosen = np.zeros(len(p), bool), np.zeros(len(p), bool), []
    for f in range(n_folds):
        train, held = fold != f, fold == f
        pick = {g: {"p": p[train & (group == g)], "wrong": wrong[train & (group == g)],
                    "extra": extra[train & (group == g)]} for g in ("turkish", "english")}
        t = choose_thresholds(pick["turkish"], pick["english"], **constraints)
        chosen.append(t)
        flagged[held] = (p[held] >= t["yellow"]) | extra[held]
        red_flagged[held] = p[held] >= t["red"]
    return {"flagged": flagged, "red": red_flagged, "thresholds": chosen}
