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
    level: str | None = None          # learned detector only: "red" (sure error), "yellow" (possible error) or None
    error_prob: float | None = None   # learned detector only: highest error probability of the word's sounds
    w_margin: float | None = None     # an expected w heard as w: how close v/β/ʋ came to it (log-probability)
    w_rival: str | None = None        # which of v/β/ʋ came closest
    w_pos: int | None = None          # position of that w among the expected sounds

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


def is_confirmed(issue: Issue, gop: float | None, threshold: float | None,
                 pattern_threshold: float | None = None) -> bool:
    """The GOP check for one difference of a word whose GOP (its worst expected sound) is `gop`.

    Confirmed when the GOP is below `threshold`, or below `pattern_threshold` for a typical
    Turkish-speaker error (it has a tip; None = the same threshold). Added sounds of a known
    pattern are always confirmed, since GOP only scores the expected sounds. Without a threshold
    or without a GOP for the word, every difference stands.
    """
    if threshold is None or gop is None or (issue.kind == "ins" and issue.tip):
        return True
    limit = pattern_threshold if issue.tip and pattern_threshold is not None else threshold
    return gop < limit


def confirm_with_gop(word: WordResult, threshold: float | None, pattern_threshold: float | None = None) -> None:
    """Report a word's differences only when GOP agrees that it was not said well.

    Greedy decoding picks the single most likely sound per frame, so a near tie can turn a
    good "think" into a heard "t". A difference is reported only if is_confirmed(); otherwise it
    is most likely a mishearing and is moved to `dismissed`. Typical Turkish-speaker errors need
    less evidence (`pattern_threshold`), since these speakers are likely to make them.
    """
    if threshold is None or word.gop is None:
        return
    word.dismissed = [i for i in word.issues if not is_confirmed(i, word.gop, threshold, pattern_threshold)]
    word.issues = [i for i in word.issues if is_confirmed(i, word.gop, threshold, pattern_threshold)]


def hidden_w_issue(word_index: int, rival: str) -> Issue:
    return Issue(word_index, "sub", "w", rival, SUBSTITUTION_TIPS.get(("w", rival)),
                 f"{describe('w')} sounded like {describe(rival)}")


def report_hidden_w(word: WordResult, word_index: int, threshold: float | None) -> None:
    """Report w -> v although the recognizer wrote w, when v/β/ʋ came within `threshold` of w.

    Turkish speakers often say English w as [v] or [β]. The fine-tuned recognizer still ranks such
    sounds a little above or below w, and greedy decoding keeps the w, but the log-probability
    margin of the closest rival separates real w -> v errors well (Speech Accent Archive dev).
    """
    if threshold is None or word.w_margin is None or word.w_margin <= threshold:
        return
    word.ops = [Op("sub", op.expected, word.w_rival, op.exp_pos, op.heard_pos) if op.exp_pos == word.w_pos else op
                for op in word.ops]
    word.issues.append(hidden_w_issue(word_index, word.w_rival))


def top_tips(results: list[WordResult], limit: int | None = None) -> list[str]:
    """Known error patterns in this recording, most frequent first."""
    counts = Counter(issue.tip for r in results for issue in r.issues if issue.tip)
    return [tip for tip, _ in counts.most_common(limit)]
