"""Choose the GOP confirmation thresholds of the v1 rules again for each recognizer (SAA dev).

v1's thresholds (-2.5 / -1.0) were chosen on the original recognizer's probabilities; a fine-tuned
recognizer is sure of itself in a different way. Chosen with them: the hidden-w margin (a w the
recognizer wrote as w is reported as w -> v when v/β/ʋ came this close; "off" = no such rule).
For every system, SAA dev is evaluated once without a GOP check or the hidden-w rule, and then
(pronunciation/thresholds.py):

  as is        the system's own thresholds
  selected     the thresholds chosen on all dev speakers: most Turkish recall while Turkish
               precision stays >= 70%, native speakers get <= 2.5% false alarms and >= 70% of the
               hidden-w reports are right (optimistic: chosen and measured on the same speakers)
  honest (CV)  the same choice by speaker cross-validation: the pair is picked without the speakers
               it is measured on. This is the estimate to decide with.
  no check     every difference the recognizer produced, before any GOP check. Its error-level
               numbers measure the recognizer itself at the sounds the expert marked wrong: how often
               it wrote something else than the expected sound (catch), and the expert's own sound
               (exact); 1 - catch is how often it wrote the expected sound (the canonical bias).

Per-speaker counts of the cross-validated flags go to results/thresholds/<name>.json, and every
pair of systems is compared on them (paired bootstrap over speakers, as in compare_experiments.py).

Usage:
    python scripts/tune_gop_thresholds.py experiments/v1.json experiments/v2-3-best.json experiments/v2-3-epoch3.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_experiments  # noqa: E402
import evaluate_saa  # noqa: E402
from pronunciation.metrics import error_level_catch  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402
from pronunciation.thresholds import (V1_THRESHOLDS, apply_thresholds, cross_validate_gop_thresholds,  # noqa: E402
                                      scores, select_gop_thresholds)

OUT_DIR = ROOT / "results" / "thresholds"
W_GRID = [None, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]      # hidden-w margins to try (None = rule off)
PATTERN_LABELS = {"final_voicing": "final devoicing", "z_to_s": "z → s", "th_voiced": "ð", "th_voiceless": "θ",
                  "w": "w → v", "short_i": "ɪ → i", "r": "r"}


def catch(units: dict[str, list[int]]) -> float:
    found, total = (sum(u[i] for u in units.values()) for i in (0, 1))
    return found / total if total else 0.0


def summary(rows) -> tuple[dict, dict]:
    units = evaluate_saa.saa_units(rows)
    result = scores(rows)
    result["error_catch"] = catch(units["saa_dev_turkish_errors"])
    turkish = rows[(rows.group == "turkish") & rows.expert_wrong.notna()]
    result["error_exact"] = error_level_catch(zip(turkish.expert_errors, turkish.our_errors))["exact"]
    # every reported w error (hidden or heard): is the expert's w wrong there too?
    w_reports = turkish[turkish.our_errors.map(lambda issues: any(i.expected == "w" for i in issues))]
    right = sum(any(e.expected == "w" and e.kind != "ins" for e in errors) for errors in w_reports.expert_errors)
    result["w_report_precision"] = right / len(w_reports) if len(w_reports) else None
    result["w_reports"] = len(w_reports)
    result.update({f"{name}_catch": catch(units[f"saa_dev_{name}"]) for name in evaluate_saa.PATTERNS})
    return result, units


def show(value: float) -> str:
    return "none" if math.isinf(value) else f"{value:g}"


def show_all(thresholds) -> str:
    """ "-2.5 / -1" or, with the hidden-w margin, "-2.5 / -1 / w off" """
    text = f"{show(thresholds[0])} / {show(thresholds[1])}"
    if len(thresholds) > 2:
        text += " / w " + ("off" if thresholds[2] is None else f"{thresholds[2]:g}")
    return text


def tune(system: System) -> dict:
    decoder, data = evaluate_saa.load("dev", system.recognizer)
    unchecked = System(system.name, system.recognizer, "rules", {"gop_threshold": None})
    raw = evaluate_saa.evaluate(decoder, data, unchecked)

    own = (system.settings.get("gop_threshold", V1_THRESHOLDS[0]),
           system.settings.get("pattern_threshold", V1_THRESHOLDS[1]),
           system.settings.get("w_margin_threshold"))
    as_is = apply_thresholds(raw, own)
    direct = evaluate_saa.evaluate(decoder, data, system)       # the offline rules must match the app's
    assert as_is.flagged.tolist() == direct.flagged.tolist(), "offline thresholds disagree with analyze()"

    selected = select_gop_thresholds(raw, w_grid=W_GRID)
    chosen = selected["thresholds"] if selected else own
    cv = cross_validate_gop_thresholds(raw, fallback=own, w_grid=W_GRID)
    parts = {"as_is": (own, as_is), "selected": (chosen, apply_thresholds(raw, chosen)),
             "cv": (None, cv["rows"]), "unchecked": ((math.inf, math.inf, None), raw)}
    result = {"name": system.name, "recognizer": system.recognizer, "folds": cv["folds"], "units": {}}
    for part, (pair, rows) in parts.items():
        result[part], result["units"][part] = summary(rows)
        result[part]["thresholds"] = pair
    return result


def print_table(results: list[dict]) -> None:
    print("| System | Thresholds (GOP / typical errors / hidden w) | Turkish precision | recall | F1 | error-level catch | native false alarm |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    labels = {"as_is": "as is", "selected": "re-selected (dev, optimistic)", "cv": "re-selected, honest (CV)",
              "unchecked": "no GOP check (recognizer alone)"}
    for r in results:
        for part, label in labels.items():
            s = r[part]
            pair = show_all(s["thresholds"]) if s["thresholds"] else "per fold"
            print(f"| {r['name']} | {label}: {pair} | {s['precision']:.1%} | {s['recall']:.1%} | {s['f1']:.3f} | "
                  f"{s['error_catch']:.1%} | {s['english_false_alarm']:.1%} |")
    print("\nRecognizer alone at the sounds the expert marked wrong (Turkish speakers, no GOP check):")
    for r in results:
        u = r["unchecked"]
        print(f"  {r['name']}: wrote something else than the expected sound {u['error_catch']:.1%} "
              f"(the expert's own sound {u['error_exact']:.1%}); wrote the expected sound {1 - u['error_catch']:.1%}")
    print("\nPer pattern (error-level catch, Turkish speakers):")
    names = list(evaluate_saa.PATTERNS)
    print("| System | " + " | ".join(PATTERN_LABELS.get(n, n) for n in names) + " |")
    print("| --- |" + " --- |" * len(names))
    for r in results:
        for part in ("as_is", "cv"):
            print(f"| {r['name']} ({labels[part]}) | " + " | ".join(f"{r[part][f'{n}_catch']:.0%}" for n in names) + " |")
    print("\nReported w errors on Turkish speakers that the expert also marked (tip precision for w):")
    for r in results:
        print("  " + r["name"] + ": " + ", ".join(
            f"{labels[part]} {r[part]['w_report_precision']:.0%} of {r[part]['w_reports']}"
            for part in ("as_is", "cv") if r[part]["w_report_precision"] is not None))
    print("\nThresholds chosen per cross-validation fold (without its speakers):")
    for r in results:
        pairs = [f"{show_all(f['thresholds'])}{'*' if f['fallback'] else ''}" for f in r["folds"]]
        print(f"  {r['name']}: {', '.join(pairs)}   (* = no pair within the limits: the system's own thresholds)")


def print_comparisons(results: list[dict]) -> None:
    for a, b in itertools.combinations(results, 2):
        print(f"\nHonest (CV) comparison: {a['name']} -> {b['name']} (paired bootstrap over speakers)")
        print(f"| Metric | {a['name']} (CV) | {b['name']} (CV) | Difference | 95% CI | Real? |")
        print("| --- | --- | --- | --- | --- | --- |")
        rows = compare_experiments.compare({"name": a["name"], "units": a["units"]["cv"]},
                                           {"name": b["name"], "units": b["units"]["cv"]})
        for row in rows:
            low, high = row["ci95"]
            print(f"| {row['metric']} | {row['a']:.1%} | {row['b']:.1%} | {row['diff']:+.1%} | "
                  f"[{low:+.1%}, {high:+.1%}] | {'yes' if row['real'] else 'no'} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", type=Path, help="system definitions (experiments/*.json)")
    args = parser.parse_args()

    results = [tune(load_system(path)) for path in args.systems]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        (OUT_DIR / f"{r['name']}.json").write_text(json.dumps(r, indent=2, default=float))
    print_table(results)
    print_comparisons(results)
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
