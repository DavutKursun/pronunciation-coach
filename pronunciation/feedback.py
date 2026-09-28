"""Turn a phoneme alignment into per-word results and learner-friendly feedback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .align import Op
from .phonemes import DEVOICED, SUBSTITUTION_TIPS, describe, is_vowel


@dataclass
class Issue:
    word_index: int
    kind: str              # "sub" | "del" | "ins"
    expected: str | None
    heard: str | None
    tip: str | None        # key into phonemes.TIPS, when the error matches a known pattern
    message: str


@dataclass
class WordResult:
    text: str
    expected: list[str]                                  # normalized expected phones
    ops: list[Op] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)      # differences we report
    dismissed: list[Issue] = field(default_factory=list)   # differences GOP did not confirm
    gop: float | None = None                               # lowest GOP (lpr) of the word's sounds

    @property
    def heard(self) -> list[str]:
        return [op.heard for op in self.ops if op.heard is not None]

    @property
    def n_matches(self) -> int:
        return sum(op.kind == "match" for op in self.ops)

    @property
    def n_insertions(self) -> int:
        return sum(op.kind == "ins" for op in self.ops)

    @property
    def score(self) -> float:
        """Share of the word's sounds said correctly; extra sounds count against it."""
        denominator = len(self.expected) + self.n_insertions
        return self.n_matches / denominator if denominator else 1.0


def starts_with_cluster(phones: list[str]) -> bool:
    return len(phones) >= 2 and not is_vowel(phones[0]) and not is_vowel(phones[1])


def group_by_word(ops: list[Op], exp_word: list[int], word_phones: list[list[str]]) -> list[list[Op]]:
    """Assign every operation to a word.

    Matches, substitutions and deletions belong to the word of their expected phone.
    An insertion inside a word belongs to that word. An insertion between two words
    goes to the next word when it looks like an extra vowel before a consonant group
    (Turkish speakers often say "is-chool" for "school"), otherwise to the previous word.
    """
    groups: list[list[Op]] = [[] for _ in word_phones]
    exp_positions = [k for k, op in enumerate(ops) if op.exp_pos is not None]

    for k, op in enumerate(ops):
        if op.exp_pos is not None:
            groups[exp_word[op.exp_pos]].append(op)
            continue
        prev_k = max((p for p in exp_positions if p < k), default=None)
        next_k = min((p for p in exp_positions if p > k), default=None)
        prev_w = exp_word[ops[prev_k].exp_pos] if prev_k is not None else None
        next_w = exp_word[ops[next_k].exp_pos] if next_k is not None else None
        if prev_w is None and next_w is None:
            continue  # nothing expected at all
        if prev_w is None:
            target = next_w
        elif next_w is None or prev_w == next_w:
            target = prev_w
        elif is_vowel(op.heard) and starts_with_cluster(word_phones[next_w]):
            target = next_w
        else:
            target = prev_w
        groups[target].append(op)
    return groups


def find_issues(word_index: int, word: WordResult, first_exp_pos: int) -> list[Issue]:
    phones = word.expected
    vowel_positions = [i for i, p in enumerate(phones) if is_vowel(p)]
    final_cluster_start = vowel_positions[-1] + 1 if vowel_positions else 0

    issues: list[Issue] = []
    ops = word.ops
    for k, op in enumerate(ops):
        if op.kind == "sub":
            local = op.exp_pos - first_exp_pos
            if local >= final_cluster_start and DEVOICED.get(op.expected) == op.heard:
                tip = "final_voicing"
            else:
                tip = SUBSTITUTION_TIPS.get((op.expected, op.heard))
            message = f"{describe(op.expected)} sounded like {describe(op.heard)}"
            issues.append(Issue(word_index, "sub", op.expected, op.heard, tip, message))
        elif op.kind == "del":
            issues.append(Issue(word_index, "del", op.expected, None, None,
                                f"{describe(op.expected)} is missing"))
        elif op.kind == "ins":
            prev_exp = next((o.expected for o in reversed(ops[:k]) if o.expected), None)
            next_exp = next((o.expected for o in ops[k + 1:] if o.expected), None)
            tip = None
            if op.heard == "ɡ" and prev_exp == "ŋ":
                tip = "ng"
            elif is_vowel(op.heard) and next_exp and not is_vowel(next_exp):
                at_start = prev_exp is None and starts_with_cluster(phones)
                inside_cluster = prev_exp is not None and not is_vowel(prev_exp)
                if at_start or inside_cluster:
                    tip = "epenthesis"
            issues.append(Issue(word_index, "ins", None, op.heard, tip,
                                f"extra sound {describe(op.heard)}"))
    return issues


def build_word_results(
    words: list[str], word_phones: list[list[str]], ops: list[Op], exp_word: list[int]
) -> list[WordResult]:
    groups = group_by_word(ops, exp_word, word_phones)
    results = []
    first_pos = 0
    for w, (text, phones, word_ops) in enumerate(zip(words, word_phones, groups)):
        result = WordResult(text=text, expected=phones, ops=word_ops)
        result.issues = find_issues(w, result, first_pos)
        results.append(result)
        first_pos += len(phones)
    return results


def confirm_with_gop(word: WordResult, threshold: float | None) -> None:
    """Report a word's differences only when GOP agrees that it was not said well.

    Greedy decoding picks the single most likely sound per frame, so a near tie can turn a
    good "think" into a heard "t". If every expected sound of the word still has a GOP of at
    least `threshold`, its differences are most likely mishearings and are moved to `dismissed`.
    Added sounds that match a known pattern (the extra vowel in "is-chool", "sing-ging") stay:
    GOP only scores the expected sounds, so it cannot judge an added one.
    """
    if threshold is None or word.gop is None or word.gop < threshold:
        return
    keep = [i for i in word.issues if i.kind == "ins" and i.tip]
    word.dismissed = [i for i in word.issues if not (i.kind == "ins" and i.tip)]
    word.issues = keep


def top_tips(results: list[WordResult], limit: int | None = None) -> list[str]:
    """Known error patterns in this recording, most frequent first."""
    counts = Counter(issue.tip for r in results for issue in r.issues if issue.tip)
    return [tip for tip, _ in counts.most_common(limit)]
