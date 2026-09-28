"""CTC decoding and forced alignment, written from scratch with NumPy.

A CTC model outputs, for every 20 ms audio frame, a probability for each phoneme plus a
special "blank" token. Two questions are answered here:

1. greedy_decode: what did the speaker most likely say? (best token per frame, merge
   repeats, drop blanks)
2. forced_align: given what the speaker SHOULD have said, which frames belong to each
   expected phoneme? This is the Viterbi algorithm over the CTC state graph. From the
   aligned frames we compute Goodness of Pronunciation (GOP) scores: how confident the
   model is that the expected phoneme was actually produced.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1e30


def greedy_decode(log_probs: np.ndarray, blank_id: int, ignore_ids: set[int] | None = None) -> list[tuple[int, int]]:
    """Return (token_id, first_frame) pairs of the most likely token sequence."""
    ignore = ignore_ids or set()
    best = log_probs.argmax(axis=1)
    out: list[tuple[int, int]] = []
    previous = -1
    for frame, token in enumerate(best.tolist()):
        if token != previous and token != blank_id and token not in ignore:
            out.append((token, frame))
        previous = token
    return out


def forced_align(log_probs: np.ndarray, targets: list[int], blank_id: int) -> list[tuple[int, int]] | None:
    """Viterbi alignment of `targets` to the frames.

    Returns one (start_frame, end_frame) span per target, or None when the audio is
    too short to contain all targets.
    """
    num_frames = log_probs.shape[0]
    if not targets or num_frames == 0:
        return None

    # CTC state graph: blank, t1, blank, t2, ..., tL, blank
    states = [blank_id]
    for token in targets:
        states += [token, blank_id]
    states_arr = np.array(states)
    num_states = len(states)

    # a state may skip the blank before it when it is a label different from the label two states back
    can_skip = np.zeros(num_states, dtype=bool)
    can_skip[2:] = (states_arr[2:] != blank_id) & (states_arr[2:] != states_arr[:-2])

    emissions = log_probs[:, states_arr]                  # [frames, states]
    score = np.full(num_states, NEG_INF)
    score[0] = emissions[0, 0]
    score[1] = emissions[0, 1]
    backpointer = np.zeros((num_frames, num_states), dtype=np.int64)

    state_idx = np.arange(num_states)
    for t in range(1, num_frames):
        stay = score
        advance = np.concatenate(([NEG_INF], score[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG_INF, NEG_INF], score[:-2])), NEG_INF)
        candidates = np.stack([stay, advance, skip])     # [3, states]
        choice = candidates.argmax(axis=0)
        score = candidates[choice, state_idx] + emissions[t]
        backpointer[t] = state_idx - choice

    # a valid path ends in the last label or the final blank
    last = num_states - 1 if score[-1] >= score[-2] else num_states - 2
    if score[last] <= NEG_INF / 2:
        return None

    path = np.empty(num_frames, dtype=np.int64)
    path[-1] = last
    for t in range(num_frames - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]

    spans = []
    for k in range(len(targets)):
        frames = np.flatnonzero(path == 2 * k + 1)
        if frames.size == 0:
            return None
        spans.append((int(frames[0]), int(frames[-1])))
    return spans


def gop_scores(
    log_probs: np.ndarray, targets: list[int], spans: list[tuple[int, int]], blank_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Goodness of Pronunciation for every target phoneme.

    lpp: best log posterior of the expected phoneme inside its frames (near 0 = confident)
    lpr: log posterior ratio against the best competing phoneme (0 = the expected phoneme
         was the model's top choice; very negative = another sound was heard)
    """
    competitors = log_probs.copy()
    competitors[:, blank_id] = NEG_INF
    best_other = competitors.max(axis=1)

    lpp = np.empty(len(targets))
    lpr = np.empty(len(targets))
    for k, (token, (start, end)) in enumerate(zip(targets, spans)):
        frames = slice(start, end + 1)
        lpp[k] = log_probs[frames, token].max()
        lpr[k] = (log_probs[frames, token] - best_other[frames]).max()
    return lpp, lpr
