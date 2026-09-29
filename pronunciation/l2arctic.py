"""L2-ARCTIC: non-native English with hand-annotated pronunciation errors, as fine-tuning targets.

Annotators marked every phone of ~150 sentences per speaker (README of the corpus):
  plain "IY1"      said correctly (the forced-alignment label, ARPAbet with a stress digit)
  "CPL,PPL,s"      substitution: CPL should have been said, PPL was heard ("Z,S,s")
  "CPL,sil,d"      deletion
  "sil,PPL,a"      addition
  "err"            the annotator could not tell what was said;  "*" = an accented version of PPL

The recognizer is fine-tuned to output what the annotator HEARD, in its own phoneme style: the
target of a sentence is our eSpeak expected phonemes, changed only where the annotator marked an
error. So the model keeps its output format and only learns to hear the errors (training on the
expected phonemes would teach it not to hear them, as v2-2 showed with speechocean762 labels).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .align import Op, align
from .feedback import Issue, WordResult, find_issues
from .phonemes import FUNCTION_WORDS, is_vowel, merge_repeats, normalize
from .speechocean import ARPABET, UNSTRESSED, arpabet_to_ipa

SPLIT_FILE = Path(__file__).resolve().parents[1] / "data" / "l2arctic_split.json"
SILENCE = {"sil", "sp", "spn", ""}

# speaker: (native language, gender), from the corpus README
SPEAKERS = {
    "ABA": ("Arabic", "M"), "SKA": ("Arabic", "F"), "YBAA": ("Arabic", "M"), "ZHAA": ("Arabic", "F"),
    "BWC": ("Mandarin", "M"), "LXC": ("Mandarin", "F"), "NCC": ("Mandarin", "F"), "TXHC": ("Mandarin", "M"),
    "ASI": ("Hindi", "M"), "RRBI": ("Hindi", "M"), "SVBI": ("Hindi", "F"), "TNI": ("Hindi", "F"),
    "HJK": ("Korean", "F"), "HKK": ("Korean", "M"), "YDCK": ("Korean", "F"), "YKWK": ("Korean", "M"),
    "EBVS": ("Spanish", "M"), "ERMS": ("Spanish", "M"), "MBMPS": ("Spanish", "F"), "NJS": ("Spanish", "F"),
    "HQTV": ("Vietnamese", "M"), "PNV": ("Vietnamese", "F"), "THV": ("Vietnamese", "F"), "TLV": ("Vietnamese", "M"),
}


def load_speakers(part: str, split: dict | None = None, final: bool = False) -> list[str]:
    """Speakers of one part of the split. The test speakers are locked until the final evaluation (v2-4)."""
    if part == "test" and not final:
        raise PermissionError("The L2-ARCTIC test speakers are locked until v2-4 (use --final).")
    split = split or json.loads(SPLIT_FILE.read_text())
    return list(split[part])


@dataclass(frozen=True)
class PhoneLabel:
    canonical: str | None           # what should have been said (ARPAbet with stress); None for an addition
    perceived: str | None           # what was heard; None for a deletion or when unsure ("err")
    kind: str                       # "correct" | "sub" | "del" | "ins"
    accented: bool = False          # "*": heard as an accented version of `perceived`
    unsure: bool = False            # "err": the annotator could not tell


def parse_phone_label(label: str) -> PhoneLabel | None:
    """One interval of the "phones" tier; None for silence. Labels may contain stray spaces."""
    text = label.strip()
    if "," not in text:
        return None if text.lower() in SILENCE else PhoneLabel(text, text, "correct")
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3 or parts[2].lower() not in ("s", "d", "a"):
        raise ValueError(f"unexpected label: {label!r}")
    cpl, ppl, tag = parts[0], parts[1], parts[2].lower()
    accented = ppl.endswith("*")
    ppl = ppl.rstrip("*").strip()
    unsure = ppl.lower() == "err"
    if tag == "d":
        return PhoneLabel(cpl, None, "del")
    if tag == "a":
        return PhoneLabel(None, None if unsure else ppl, "ins", accented, unsure)
    if unsure:
        return PhoneLabel(cpl, None, "sub", unsure=True)
    if re.sub(r"\d", "", ppl) == re.sub(r"\d", "", cpl):   # "R,R*,s": an accented R is still an R
        return PhoneLabel(cpl, cpl, "correct", accented=True)
    return PhoneLabel(cpl, ppl, "sub", accented)


def perceived_to_ipa(perceived: str, canonical: str | None) -> str:
    """Perceived labels have no stress digit. AH and ER then follow the expected sound: after an
    unstressed (or no) expected sound they are the reduced vowels ə and ɚ, otherwise ʌ and ɜː. AX is ə."""
    base = re.sub(r"\d", "", perceived).upper()
    if base == "AX":
        return "ə"
    if base in UNSTRESSED and not perceived[-1:].isdigit():
        stressed = canonical is not None and canonical[-1:] in ("1", "2")
        return ARPABET[base] if stressed else UNSTRESSED[base]
    return arpabet_to_ipa(perceived.upper())


def parse_textgrid(text: str) -> dict[str, list[tuple[float, float, str]]]:
    """Interval tiers of a Praat TextGrid (long text format): tier name -> [(xmin, xmax, text)]."""
    tiers = {}
    for block in re.split(r"item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "([^"]*)"', block).group(1)
        intervals = re.findall(r'intervals \[\d+\]:\s*xmin = (\S+)\s*xmax = (\S+)\s*text = "((?:[^"]|"")*)"', block)
        tiers[name] = [(float(a), float(b), t.replace('""', '"')) for a, b, t in intervals]
    return tiers


@dataclass
class WordAnnotation:
    text: str
    phones: list[PhoneLabel] = field(default_factory=list)


def word_annotations(words: list[tuple[float, float, str]], phones: list[tuple[float, float, str]]) -> list[WordAnnotation]:
    """Group the phone labels by word (by the middle of each phone). An addition in a pause goes to the nearest word."""
    out = [(start, end, WordAnnotation(text)) for start, end, text in words if text.strip()]
    for start, end, label in phones:
        parsed = parse_phone_label(label)
        if parsed is None or not out:
            continue
        middle = (start + end) / 2
        distance = [0.0 if s <= middle <= e else min(abs(middle - s), abs(middle - e)) for s, e, _ in out]
        out[distance.index(min(distance))][2].phones.append(parsed)
    return [w for _, _, w in out]


@dataclass
class Target:
    ok: bool
    reason: str = ""
    tokens: list[str] = field(default_factory=list)     # what the recognizer should output (model tokens)
    canonical: list[str] = field(default_factory=list)  # eSpeak expected tokens (what it outputs for correct speech)
    expected: list[str] = field(default_factory=list)   # expected phonemes as compared by the feedback (normalized)
    perceived: list[str] = field(default_factory=list)  # what the annotator heard, normalized the same way
    errors: list[Issue] = field(default_factory=list)   # every annotated error, with its Turkish-speaker tip if any
    accented: int = 0                                    # "*" labels on the expected sound: not errors


def _word_parts(raw: list[str]) -> list[tuple[str, int]]:
    """(normalized phoneme, index of the eSpeak token it comes from), repeats merged like prepare_expected."""
    parts = [(p, i) for i, token in enumerate(raw) for p in normalize([token])]
    return [part for k, part in enumerate(parts) if k == 0 or parts[k - 1][0] != part[0]]


def build_target(words: list[WordAnnotation], espeak: list[list[str]]) -> Target:
    """Our expected (eSpeak) tokens, with the annotated errors applied. See the module docstring."""
    if len(words) != len(espeak):
        return Target(False, "word mismatch")
    target = Target(True)
    first = 0
    for w, (word, raw) in enumerate(zip(words, espeak)):
        if any(label.unsure for label in word.phones):
            return Target(False, "err label")
        parts = _word_parts(raw)
        ours = [p for p, _ in parts]
        canonical = [label for label in word.phones if label.kind != "ins"]
        if not canonical and ours:
            return Target(False, "unknown word")
        try:
            theirs = [arpabet_to_ipa(label.canonical) for label in canonical]
            heard = {id(label): perceived_to_ipa(label.perceived, label.canonical)
                     for label in word.phones if label.kind in ("sub", "ins")}
        except KeyError:
            return Target(False, "unknown phone")
        ops = align(ours, theirs, [word.text.lower() in FUNCTION_WORDS] * len(ours))
        partner = {op.heard_pos: op.exp_pos for op in ops if op.heard_pos is not None and op.exp_pos is not None}

        change: dict[int, str | None] = {}                 # our position -> heard phoneme (None = deleted)
        added: dict[int, list[str]] = {}                   # our position -> added phonemes after it (-1 = word start)
        j = -1
        for label in word.phones:
            if label.kind == "ins":
                anchor = next((partner[i] for i in range(j, -1, -1) if i in partner), -1)
                added.setdefault(anchor, []).append(heard[id(label)])
                continue
            j += 1
            target.accented += label.accented and label.kind == "correct"
            if label.kind == "correct":
                continue
            k = partner.get(j)
            if k is None or is_vowel(ours[k]) != is_vowel(theirs[j]):
                return Target(False, "unreliable alignment")
            new = None if label.kind == "del" else heard[id(label)]
            if new != ours[k]:          # a lexicon difference can make the "error" our expected sound
                change[k] = new

        tokens, perceived, word_ops = [], [], []
        by_token: dict[int, list[int]] = {}
        for k, (_, i) in enumerate(parts):
            by_token.setdefault(i, []).append(k)

        def emit(k: int, whole_token: bool) -> None:
            if k in change:
                if change[k] is None:
                    word_ops.append(Op("del", ours[k], None, first + k, None))
                else:
                    word_ops.append(Op("sub", ours[k], change[k], first + k, None))
                    tokens.append(change[k])
                    perceived.append(change[k])
            else:
                word_ops.append(Op("match", ours[k], ours[k], first + k, None))
                if not whole_token:
                    tokens.append(ours[k])
                perceived.append(ours[k])
            for p in added.get(k, []):
                word_ops.append(Op("ins", None, p, None, None))
                tokens.append(p)
                perceived.append(p)

        for p in added.get(-1, []):
            word_ops.append(Op("ins", None, p, None, None))
            tokens.append(p)
            perceived.append(p)
        previous_edited = False
        for i, token in enumerate(raw):
            ks = by_token.get(i, [])
            if not ks:                 # a repeat merged away ("during": ʊɹ ɹ); drop it if the r was edited
                if not previous_edited:
                    tokens.append(token)
                continue
            edited = any(k in change for k in ks) or any(added.get(k) for k in ks[:-1])
            if not edited:
                tokens.append(token)
            for k in ks:
                emit(k, whole_token=not edited)
            previous_edited = edited

        target.canonical += raw
        target.expected += ours
        target.perceived += merge_repeats(perceived)
        target.errors += find_issues(w, WordResult(word.text, ours, word_ops), first)
        target.tokens += tokens
        first += len(ours)
    return target
