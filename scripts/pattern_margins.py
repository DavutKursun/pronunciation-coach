"""Can the recognizer's probabilities tell a Turkish-speaker error from a correct sound? (SAA dev)

The w -> v margin of v2-3c, for every substitution pattern of phonemes.SUBSTITUTION_TIPS and for
final devoicing (only in a word's final consonant group); added sounds (the ɡ after ŋ, an extra
vowel) are left out. At every expected sound with a pattern, in the frames forced-aligned to it:
margin = the largest log p(pattern's sound) - log p(expected sound) (feedback.pattern_rivals,
assess.rival_margin). Positives: sounds the expert heard as that pattern's error (Turkish
speakers); negatives: sounds the expert found correct (US English and Turkish speakers); a sound
the expert heard as another error is in neither group.

Per pattern and recognizer: the expert errors, how many of them greedy decoding wrote as the
expected sound (only the hidden-error rule can report those), and the AUC with a 95% interval from
resampling speakers. This table is only a report: the rule's thresholds are chosen by
scripts/tune_hidden_patterns.py, never by looking at the AUC. Only the dev half is used.

--w-rule prints the v2-3c evidence for the hidden-w rule instead (w with v/β/ʋ only, and how many
of its reports on Turkish speakers fall on a real w error, per margin threshold).

Usage:
    python scripts/pattern_margins.py experiments/v1.json experiments/v2.json
    python scripts/pattern_margins.py --w-rule experiments/v1.json experiments/v2.json
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
from pronunciation.assess import W_RIVALS, analyze, compare, prepare_expected, rival_margin  # noqa: E402
from pronunciation.feedback import final_cluster_start, pattern_rivals, substitution_tip  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.phonemes import FUNCTION_WORDS, TIPS, WORD_VARIANTS  # noqa: E402
from pronunciation.saa import PARAGRAPH, align_words, expert_labels, narrow_to_broad, parse_transcription  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402
from pronunciation.thresholds import HIDDEN_PATTERNS, hidden_w_precision  # noqa: E402

TAUS = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]


def sound_items(recognizer: str) -> list[dict]:
    """One item per expected sound and pattern of that sound: speaker, pattern, label (1 = the expert heard
    the pattern's error, 0 = correct), margin, and whether greedy decoding wrote the expected sound."""
    words = tokenize(PARAGRAPH)
    raw = phonemize_words(words)
    word_phones = prepare_expected(words, raw)[0]
    first = np.cumsum([0] + [len(p) for p in word_phones])
    decoder, data = evaluate_saa.load("dev", recognizer)
    items = []
    for speaker, log_probs, seconds, transcription in data:
        expert = align_words(word_phones, [narrow_to_broad(t) for t in parse_transcription(transcription)],
                             [w.lower() in FUNCTION_WORDS for w in words])
        recognition = decoder(log_probs, seconds)
        result = analyze(PARAGRAPH, raw, recognition, decoder.token_to_id, decoder.blank_id)
        alignment = result.alignment
        if alignment.spans is None:
            continue
        greedy = {op.exp_pos: op for word in result.words for op in word.ops if op.exp_pos is not None}
        for w, (text, phones, heard) in enumerate(zip(words, word_phones, expert)):
            if heard is None:                                   # the speaker skipped the word
                continue
            [said] = compare([text], [phones], heard)           # the expert's word, aligned as in expert_labels
            final_start, variants = final_cluster_start(phones), WORD_VARIANTS.get(text.lower(), {})
            for op in said.ops:
                k = first[w] + op.exp_pos if op.exp_pos is not None else None
                if k is None or alignment.phone_target[k] is None:
                    continue
                final = op.exp_pos >= final_start
                patterns = pattern_rivals(op.expected, final, text.lower() in FUNCTION_WORDS,
                                          variants.get(op.expected, set()))
                for tip, sounds in patterns.items():
                    if op.kind == "match":
                        label = 0
                    elif op.kind == "sub" and substitution_tip(op.expected, op.heard, final) == tip \
                            and speaker.startswith("turkish"):
                        label = 1
                    else:
                        continue                                # another error, or a native one: in neither group
                    ids = [decoder.token_to_id[s] for s in sounds if s in decoder.token_to_id]
                    if not ids:
                        continue
                    target = alignment.phone_target[k]
                    margin = rival_margin(recognition.log_probs, alignment.spans[target],
                                          alignment.target_ids[target], ids)[0]
                    g = greedy.get(k)
                    items.append({"speaker": speaker, "pattern": tip, "label": label, "margin": margin,
                                  "greedy_expected": g is not None and g.kind == "match" and g.heard == g.expected})
    return items


def auc_with_interval(items, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float] | None:
    """AUC of the margin (errors above correct sounds) with a 95% speaker-bootstrap interval; None without both."""
    y, m = np.array([i["label"] for i in items]), np.array([i["margin"] for i in items])
    if len(set(y)) < 2:
        return None
    by_speaker: dict[str, list[int]] = {}
    for j, item in enumerate(items):
        by_speaker.setdefault(item["speaker"], []).append(j)
    speakers, rng, values = sorted(by_speaker), np.random.default_rng(seed), []
    for _ in range(n_resamples):
        idx = [j for s in rng.choice(speakers, len(speakers)) for j in by_speaker[s]]
        if len(set(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], m[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return roc_auc_score(y, m), float(low), float(high)


def pattern_table(systems: list[System]) -> None:
    results = {s.name: sound_items(s.recognizer) for s in systems}
    print("| Pattern | Expert errors (Turkish) | Correct sounds (Turkish / native) | "
          + " | ".join(f"{s.name}: errors greedy wrote as expected | {s.name}: AUC [95% CI]" for s in systems) + " |")
    print("| --- | --- | --- |" + " --- | --- |" * len(systems))
    for tip in HIDDEN_PATTERNS:
        first = [i for i in results[systems[0].name] if i["pattern"] == tip]
        errors = [i for i in first if i["label"] == 1]
        correct = [i for i in first if i["label"] == 0]
        native = sum(i["speaker"].startswith("english") for i in correct)
        cells = [TIPS[tip]["title"], str(len(errors)), f"{len(correct) - native} / {native}"]
        for s in systems:
            items = [i for i in results[s.name] if i["pattern"] == tip]
            missed = sum(i["greedy_expected"] for i in items if i["label"] == 1)
            auc = auc_with_interval(items)
            cells += [str(missed), "–" if auc is None else f"{auc[0]:.3f} [{auc[1]:.3f}–{auc[2]:.3f}]"]
        print("| " + " | ".join(cells) + " |")
    print("\n(errors of the w pattern include w heard as ɹ, which SUBSTITUTION_TIPS lists; the v2-3c rule used v/β/ʋ)")


def w_rule_margins(recognizer: str) -> list[tuple[str, int, float]]:
    """v2-3c: (speaker, 1 = expert heard v/β/ʋ, 0 = correct w, margin) for every expected w."""
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
            margin = rival_margin(recognition.log_probs, alignment.spans[alignment.phone_target[k]], w_id, rivals)[0]
            items.append((speaker, int(bool(w_errors)), margin))
    return items


def w_rule(system: System) -> None:
    items = w_rule_margins(system.recognizer)
    y = np.array([i[1] for i in items])
    m = np.array([i[2] for i in items])
    native = np.array([i[0].startswith("english") for i in items])
    auc, low, high = auc_with_interval([{"speaker": s, "label": label, "margin": margin} for s, label, margin in items])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", type=Path)
    parser.add_argument("--w-rule", action="store_true", help="the v2-3c hidden-w evidence (w with v/β/ʋ only)")
    args = parser.parse_args()
    systems = [load_system(path) for path in args.systems]
    if args.w_rule:
        for system in systems:
            w_rule(system)
    else:
        pattern_table(systems)


if __name__ == "__main__":
    main()
