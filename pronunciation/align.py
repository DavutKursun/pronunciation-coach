"""Align the phonemes the learner should have said with the phonemes they actually said.

This is a weighted edit distance (Levenshtein / Needleman-Wunsch) solved with dynamic
programming, followed by a backtrace that recovers WHICH operations were used:
a match, a substitution (said a different sound), a deletion (skipped a sound)
or an insertion (added a sound).

Costs are phonetically motivated: accent variants cost nothing, swapping a vowel for a
vowel (or consonant for consonant) is cheaper than mixing the two, so the alignment
pairs up sounds the way a human listener would.
"""

from __future__ import annotations

from dataclasses import dataclass

from .phonemes import is_acceptable, is_vowel

INDEL_COST = 1.0         # skipping or adding a sound
VARIANT_COST = 0.01      # an accepted accent variant: free in practice, but an exact match wins ties
SAME_CLASS_COST = 0.6    # vowel <-> vowel, consonant <-> consonant
CROSS_CLASS_COST = 1.2   # vowel <-> consonant (rare in real speech)


@dataclass
class Op:
    kind: str                # "match" | "sub" | "del" | "ins"
    expected: str | None     # expected phone (None for insertions)
    heard: str | None        # heard phone (None for deletions, and for an optional sound left out)
    exp_pos: int | None      # index in the expected sequence
    heard_pos: int | None    # index in the heard sequence


def substitution_cost(expected: str, heard: str, function_word: bool = False, extra: set = frozenset()) -> float:
    if expected == heard:
        return 0.0
    if is_acceptable(expected, heard, function_word, extra):
        return VARIANT_COST
    return SAME_CLASS_COST if is_vowel(expected) == is_vowel(heard) else CROSS_CLASS_COST


def align(expected: list[str], heard: list[str], function_flags: list[bool] | None = None,
          variants: list[set] | None = None) -> list[Op]:
    """Return the cheapest sequence of operations that turns `expected` into `heard`.

    function_flags[i] is True when expected[i] belongs to a function word ("the", "to"...),
    where reduced vowels are accepted. variants[i] holds extra accepted realizations of
    expected[i] in this word (weak forms, e.g. θ in "with"); None in it means the sound may be
    left out (the h of "her"). Leaving it out is then a match, not a deletion.
    """
    n, m = len(expected), len(heard)
    flags = function_flags or [False] * n
    extra = variants or [frozenset()] * n

    # cost[i][j] = cheapest way to align expected[:i] with heard[:j]
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]  # which step reached this cell
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + (VARIANT_COST if None in extra[i - 1] else INDEL_COST)
        move[i][0] = "del"
    for j in range(1, m + 1):
        cost[0][j], move[0][j] = j * INDEL_COST, "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = substitution_cost(expected[i - 1], heard[j - 1], flags[i - 1], extra[i - 1])
            skip = VARIANT_COST if None in extra[i - 1] else INDEL_COST
            candidates = (
                (cost[i - 1][j - 1] + sub, "diag"),     # match or substitution
                (cost[i - 1][j] + skip, "del"),         # expected sound was skipped
                (cost[i][j - 1] + INDEL_COST, "ins"),   # an extra sound was said
            )
            # min() keeps the first of equal candidates: prefer diag, then del, then ins
            cost[i][j], move[i][j] = min(candidates, key=lambda c: c[0])

    # backtrace from the bottom-right corner
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "diag":
            e, h = expected[i - 1], heard[j - 1]
            kind = "match" if is_acceptable(e, h, flags[i - 1], extra[i - 1]) else "sub"
            ops.append(Op(kind, e, h, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == "del":
            kind = "match" if None in extra[i - 1] else "del"
            ops.append(Op(kind, expected[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(Op("ins", None, heard[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def edit_cost(ops: list[Op], function_flags: list[bool] | None = None, variants: list[set] | None = None) -> float:
    """Total cost of an alignment (useful for tests and debugging)."""
    total = 0.0
    for op in ops:
        if op.kind in ("del", "ins"):
            total += INDEL_COST
        elif op.heard is None:
            total += VARIANT_COST   # an optional sound left out
        else:
            flag = function_flags[op.exp_pos] if function_flags else False
            extra = variants[op.exp_pos] if variants else frozenset()
            total += substitution_cost(op.expected, op.heard, flag, extra)
    return total
