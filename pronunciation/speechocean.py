"""speechocean762 phone labels, moved onto our (eSpeak) expected phonemes.

Five experts score every phone 0-2 (2 = correct, 1 = right but with a heavy accent, 0 = wrong or
missing); the dataset gives their average, so the values are 0, 0.2, ..., 2. The phones are
ARPAbet (the CMU dictionary), while our expected phonemes come from eSpeak in IPA, so each word is
converted and aligned with our own edit distance before the scores are moved over.
"""

from __future__ import annotations

import re
from collections import Counter

from .align import align
from .phonemes import is_acceptable, is_vowel, normalize

# ARPAbet -> IPA in the eSpeak style our recognizer uses. AH and ER depend on stress (digit 0 = unstressed).
ARPABET = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɜː", "EY": "eɪ",
    "IH": "ɪ", "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
UNSTRESSED = {"AH": "ə", "ER": "ɚ"}

LABELS = {"strict": 0.5, "lenient": 1.5}   # an error when the average score is below this


def arpabet_to_ipa(phone: str) -> str:
    base = re.sub(r"\d", "", phone)
    if phone.endswith("0") and base in UNSTRESSED:
        return UNSTRESSED[base]
    return ARPABET[base]


def is_error(score: float, label: str) -> bool:
    """strict: the average rounds to 0 ("wrong or missing"); lenient: it rounds to 0 or 1 (also "heavy accent")."""
    if label not in LABELS:
        raise ValueError(f"unknown label definition: {label}")
    return score < LABELS[label]


def transfer_labels(expected: list[str], arpabet: list[str], scores: list[float],
                    function_word: bool) -> tuple[list[float | None], Counter]:
    """The expert score for each of our expected phonemes of one word (None when there is none).

    Our phonemes and the CMU ones are aligned with the same weighted edit distance as the feedback.
    A pair that matches (or is an accepted variant) or is a same-class substitution (a lexicon
    difference such as ɪ vs ə) passes its score on. A vowel paired with a consonant is too
    uncertain to trust, and phonemes without a partner get no label.
    """
    theirs = normalize([arpabet_to_ipa(p) for p in arpabet])
    # normalize() may split a CMU phone in two (it never does for ARPAbet today); keep scores aligned
    their_scores = [s for p, s in zip(arpabet, scores) for _ in normalize([arpabet_to_ipa(p)])]
    ops = align(expected, theirs, [function_word] * len(expected))
    labels: list[float | None] = [None] * len(expected)
    stats: Counter = Counter()
    for op in ops:
        if op.exp_pos is None:
            stats["dropped"] += 1
        elif op.heard_pos is None:
            stats["unlabeled"] += 1
        elif op.expected == op.heard:
            stats["same"] += 1
            labels[op.exp_pos] = their_scores[op.heard_pos]
        elif is_acceptable(op.expected, op.heard, function_word):
            stats["variant"] += 1
            labels[op.exp_pos] = their_scores[op.heard_pos]
        elif is_vowel(op.expected) == is_vowel(op.heard):
            stats["substituted"] += 1
            labels[op.exp_pos] = their_scores[op.heard_pos]
        else:
            stats["uncertain"] += 1
    return labels, stats
