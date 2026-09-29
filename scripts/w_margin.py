"""Can the recognizer's probabilities tell a Turkish w -> v from a correct w? (SAA dev)

At every expected w, in the frames forced-aligned to it: margin = the largest
log p(v/β/ʋ) - log p(w). Positives: w the expert heard as v, β or ʋ (Turkish speakers);
negatives: w the expert found correct (US English and Turkish speakers). Reports the AUC with a
95% interval from resampling speakers, and, for the hidden-w rule (a w the recognizer wrote as w
reported as w -> v above a margin), how many of its reports on Turkish speakers fall on a real w
error. Only the dev half is used.

Usage:
    python scripts/w_margin.py experiments/v1.json experiments/v2.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_saa  # noqa: E402
from pronunciation.assess import W_RIVALS, analyze, prepare_expected  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.phonemes import FUNCTION_WORDS  # noqa: E402
from pronunciation.saa import PARAGRAPH, align_words, expert_labels, narrow_to_broad, parse_transcription  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402
from pronunciation.thresholds import hidden_w_precision  # noqa: E402

TAUS = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]


def margins(recognizer: str) -> list[tuple[str, int, float]]:
    """(speaker, 1 = expert heard v/β/ʋ, 0 = correct w, margin) for every expected w."""
    words = tokenize(PARAGRAPH)
    raw = phonemize_words(words)
    word_phones, flat, exp_word, _ = prepare_expected(words, raw)
    decoder, data = evaluate_saa.load("dev", recognizer)
    rivals, w_id = [decoder.token_to_id[c] for c in W_RIVALS], decoder.token_to_id["w"]
    items = []
    for speaker, log_probs, seconds, transcription in data:
        expert = align_words(word_phones, [narrow_to_broad(t) for t in parse_transcription(transcription)],
                             [w.lower() in FUNCTION_WORDS for w in words])
        labels = expert_labels(words, word_phones, expert)
        recognition = decoder(log_probs, seconds)
        alignment = analyze(PARAGRAPH, raw, recognition, decoder.token_to_id, decoder.blank_id).alignment
        for k, phone in enumerate(flat):
            label = labels[exp_word[k]]
            if phone != "w" or label.wrong is None or alignment.spans is None or alignment.phone_target[k] is None:
                continue
            w_errors = [e for e in label.errors if e.expected == "w"]
            if w_errors and not any(e.heard in W_RIVALS for e in w_errors):
                continue                      # w said as something else: in neither group
            start, end = alignment.spans[alignment.phone_target[k]]
            frames = recognition.log_probs[start:end + 1]
            items.append((speaker, int(bool(w_errors)), float((frames[:, rivals] - frames[:, [w_id]]).max())))
    return items


def auc_with_interval(items, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    y, m = np.array([i[1] for i in items]), np.array([i[2] for i in items])
    by_speaker: dict[str, list[int]] = {}
    for j, item in enumerate(items):
        by_speaker.setdefault(item[0], []).append(j)
    speakers, rng, values = sorted(by_speaker), np.random.default_rng(seed), []
    for _ in range(n_resamples):
        idx = [j for s in rng.choice(speakers, len(speakers)) for j in by_speaker[s]]
        if len(set(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], m[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return roc_auc_score(y, m), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.systems:
        system = load_system(path)
        items = margins(system.recognizer)
        y = np.array([i[1] for i in items])
        m = np.array([i[2] for i in items])
        native = np.array([i[0].startswith("english") for i in items])
        auc, low, high = auc_with_interval(items)
        print(f"\n{system.name} ({system.recognizer}): {int(y.sum())} w heard as v/β/ʋ by the expert, "
              f"{int((y == 0).sum())} correct w ({int(((y == 0) & native).sum())} native)")
        print(f"  AUC {auc:.3f} [95% speaker bootstrap {low:.3f}-{high:.3f}]; median margin: errors {np.median(m[y == 1]):.2f}, "
              f"correct Turkish {np.median(m[(y == 0) & ~native]):.2f}, native {np.median(m[(y == 0) & native]):.2f}")
        unchecked = System(system.name, system.recognizer, "rules", {"gop_threshold": None})
        decoder, data = evaluate_saa.load("dev", system.recognizer)
        rows = evaluate_saa.evaluate(decoder, data, unchecked)
        print("  hidden-w rule: reports on Turkish speakers that fall on a real w error, per margin threshold")
        for tau in TAUS:
            right, total = hidden_w_precision(rows, tau)
            native_reports = int(((rows.group == "english") & (rows.w_margin > tau)).sum())
            print(f"    {tau:5.1f}: {right:3} of {total:3}" + (f" ({right / total:.0%})" if total else "")
                  + f", native words reported {native_reports}")


if __name__ == "__main__":
    main()
