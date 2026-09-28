"""Speech Accent Archive helpers: turn expert transcriptions into labels we can compare with.

Every speaker reads the same paragraph. Trained phoneticians wrote down what each speaker said in
narrow IPA: it marks details (aspiration, dental t, nasalized vowels...) that are not different
phonemes and that our recognizer and eSpeak never write. narrow_to_broad() removes those details,
so an expert word can be aligned with the expected phonemes like the recognizer output is.

A word is "really wrong" when the expert wrote something outside our accepted variants.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .align import align, edit_cost
from .assess import compare
from .phonemes import VOWEL_CHARS, normalize

PARAGRAPH = (
    "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, "
    "five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake "
    "and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her "
    "Wednesday at the train station."
)

SYLLABIC = "̩"
# Marks that do not change the phoneme (NFD: combining marks are separate characters).
REMOVE = set(
    "ˈˌːˑ"            # stress, length (SAA uses length for phonetic detail, not for ship/sheep)
    "ʰʲʷˠˀ"           # aspirated, palatalized, labialized, velarized, glottalized
    "˺̚"          # unreleased stop (two ways of writing it)
    "̃"          # nasalized
    "̆"          # extra short
    "̥̬"    # voiceless / voiced: partial devoicing (bæɡ̥) is normal English; full devoicing is written p/t/k/s
    "̪̺̻̼"  # dental, apical, laminal, linguolabial: t̪ is still a t
    "̝̞̟̠̘̙̹̜̈̽"  # raised, lowered, advanced, retracted, ATR, rounding, centralized
    "̯͜͡‿"        # non-syllabic, tie bars
)
LETTERS = {"ʧ": "tʃ", "ʤ": "dʒ", "g": "ɡ", "ɝ": "ɜ", "ʉ": "u", "ɨ": "ɪ"}
LONG_ONLY = {"ɑ": "ɑː", "ɜ": "ɜː"}   # eSpeak only has the long forms of these vowels
MULTI = ["aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "tʃ", "dʒ", "n" + SYLLABIC, "l" + SYLLABIC]
VOWELS = "".join(sorted(VOWEL_CHARS))
WORD_GAP = 0.5   # cost of skipping a paragraph word or an extra transcribed word in align_words


def narrow_to_broad(word: str) -> list[str]:
    """One transcribed word in narrow IPA -> our broad phonemes, e.g. "pʰliz̥" -> ["p", "l", "i", "z"]."""
    s = unicodedata.normalize("NFD", word)
    for narrow, broad in LETTERS.items():
        s = s.replace(narrow, broad)
    # r-coloured vowels: ə˞ = ɚ, ɜ˞ = ɜ, any other vowel with a hook = vowel + ɹ
    s = s.replace("ə˞", "ɚ").replace("ɜ˞", "ɜ")
    s = re.sub(f"([{VOWELS}])˞", r"\1ɹ", s)
    s = re.sub(f"(?<![nl]){SYLLABIC}", "", s)
    s = "".join(c for c in s if c not in REMOVE)
    s = re.sub(f"([{VOWELS}])əɹ", r"\1ɹ", s)          # schwa glide before r: stoəɹ = stoɹ
    s = s.replace("ɜɹ", "ɜ")
    s = re.sub(f"əɹ(?![{VOWELS}])", "ɚ", s)             # bɹʌðəɹ = bɹʌðɚ
    s = re.sub(f"([{VOWELS}])\\1+", r"\1", s)           # ii = i

    phones, k = [], 0
    while k < len(s):
        multi = next((m for m in MULTI if s.startswith(m, k)), None)
        phone = multi or s[k]
        phones.append(LONG_ONLY.get(phone, phone))
        k += len(phone)
    return normalize(phones)


def parse_transcription(text: str) -> list[str]:
    """The transcribed words inside [...]; some files start with a "Turkish speaker 1" header."""
    match = re.search(r"\[(.*?)\]", text, re.S)
    return (match.group(1) if match else text).split()


def _word_cost(expected: list[str], heard: list[str], function_word: bool) -> float:
    flags = [function_word] * len(expected)
    return edit_cost(align(expected, heard, flags), flags) / max(len(expected), len(heard), 1)


def align_words(expected: list[list[str]], heard: list[list[str]], function_flags: list[bool]) -> list[list[str] | None]:
    """Match transcribed words to paragraph words (speakers repeat or skip words).

    The same dynamic programming as the phoneme alignment, one level up: matching two words costs
    their phoneme edit distance per sound, skipping a word costs WORD_GAP. Returns the transcribed
    phonemes for every paragraph word, or None when the speaker skipped it.
    """
    n, m = len(expected), len(heard)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], move[i][0] = i * WORD_GAP, "skip"
    for j in range(1, m + 1):
        cost[0][j], move[0][j] = j * WORD_GAP, "extra"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j], move[i][j] = min(
                (cost[i - 1][j - 1] + _word_cost(expected[i - 1], heard[j - 1], function_flags[i - 1]), "match"),
                (cost[i - 1][j] + WORD_GAP, "skip"),
                (cost[i][j - 1] + WORD_GAP, "extra"),
                key=lambda c: c[0])
    matched: list[list[str] | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "match":
            matched[i - 1] = heard[j - 1]
            i, j = i - 1, j - 1
        elif step == "skip":
            i -= 1
        else:
            j -= 1
    return matched


@dataclass
class Label:
    """What the expert heard in one paragraph word."""
    wrong: bool | None                               # None: the speaker skipped the word
    tips: set[str] = field(default_factory=set)      # Turkish-speaker patterns in the expert's version
    pairs: list[str] = field(default_factory=list)   # "expected → heard" for every difference


def expert_labels(words: list[str], expected: list[list[str]], expert: list[list[str] | None]) -> list[Label]:
    """Align each expert word with its expected phonemes, using the same rules as our feedback."""
    labels = []
    for text, exp, heard in zip(words, expected, expert):
        if heard is None:
            labels.append(Label(wrong=None))
            continue
        [result] = compare([text], [exp], heard)
        labels.append(Label(
            wrong=bool(result.issues),
            tips={i.tip for i in result.issues if i.tip},
            pairs=[f"{i.expected or '-'} → {i.heard or '-'}" for i in result.issues],
        ))
    return labels
