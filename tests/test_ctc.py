import numpy as np

from pronunciation.ctc import forced_align, gop_scores, greedy_decode


def one_hot_log_probs(frames, vocab_size=5, p=0.9):
    lp = np.full((len(frames), vocab_size), np.log((1 - p) / (vocab_size - 1)))
    for t, token in enumerate(frames):
        lp[t, token] = np.log(p)
    return lp


def test_greedy_decode_merges_repeats_and_drops_blanks():
    lp = one_hot_log_probs([0, 1, 1, 0, 1, 2, 2, 0, 3])
    assert [tok for tok, _ in greedy_decode(lp, blank_id=0)] == [1, 1, 2, 3]


def test_greedy_decode_ignores_special_tokens():
    lp = one_hot_log_probs([4, 1, 0])
    assert [tok for tok, _ in greedy_decode(lp, blank_id=0, ignore_ids={4})] == [1]


def test_forced_align_finds_the_frames_of_each_target():
    lp = one_hot_log_probs([0, 1, 1, 0, 0, 2, 0, 3, 3])
    assert forced_align(lp, [1, 2, 3], blank_id=0) == [(1, 2), (5, 5), (7, 8)]


def test_forced_align_handles_repeated_labels():
    # the same phoneme twice needs a blank in between
    lp = one_hot_log_probs([1, 0, 1])
    assert forced_align(lp, [1, 1], blank_id=0) == [(0, 0), (2, 2)]


def test_forced_align_returns_none_when_audio_is_too_short():
    lp = one_hot_log_probs([1, 2])
    assert forced_align(lp, [1, 2, 3], blank_id=0) is None
    assert forced_align(one_hot_log_probs([1, 1]), [1, 1], blank_id=0) is None


def test_gop_is_low_when_a_different_sound_was_heard():
    # expected 1 2 3, but the model heard 4 instead of 2
    lp = one_hot_log_probs([0, 1, 0, 4, 0, 3, 0])
    targets = [1, 2, 3]
    spans = forced_align(lp, targets, blank_id=0)
    lpp, lpr = gop_scores(lp, targets, spans, blank_id=0)
    assert lpr[0] == 0 and lpr[2] == 0          # correct phonemes: top choice
    assert lpr[1] < -3                           # wrong phoneme: far from top choice
    assert lpp[1] < lpp[0]
