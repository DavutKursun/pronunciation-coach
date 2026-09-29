"""v2-3d: the hidden-error rule for every Turkish-speaker pattern, chosen and checked honestly on SAA dev.

v2-3c reports w -> v when v/β/ʋ came close to a w the recognizer wrote as w. Here the same rule covers
every substitution pattern of phonemes.SUBSTITUTION_TIPS and final devoicing (feedback.report_hidden),
each pattern with its own margin threshold or "off" (pronunciation/thresholds.py):

  1. the GOP thresholds as in v2 (select_gop_thresholds: same grid, same limits), without hidden errors
  2. the pattern thresholds in one pass in phonemes.py order (select_hidden_thresholds): each pattern
     gets the most lenient threshold with Turkish precision >= 70%, native false alarms <= 2.5%, and
     for its own hidden reports on Turkish speakers: at least 10 of them, at least 70% on the error the
     expert heard (diagnosis precision); otherwise it stays off

Both steps run inside every fold of the speaker cross-validation, on that fold's training speakers only
(the honest estimate); the final thresholds are chosen the same way on all dev speakers. The original
recognizer (v1) gets the same treatment, for a fair comparison. The systems as frozen in v2-3c are
measured with their own v2-3c cross-validation (scripts/tune_gop_thresholds.py), which must reproduce
results/thresholds/<name>.json.

Decision rule: the generalized system replaces frozen v2 only if
  - its recall gain over frozen v2 is real (paired bootstrap over speakers: the 95% interval excludes 0),
  - its honest (CV) Turkish precision is >= 70% and native false alarms <= 2.5%,
  - false alarms on the words Turkish speakers said correctly rise by at most 2 points.

Results go to results/hidden/<name>-gen.json and results/hidden/decision.json; with --write-experiments
the chosen settings also go to experiments/<name>-gen.json.

Usage:
    python scripts/tune_hidden_patterns.py experiments/v1.json experiments/v2.json
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
from tune_gop_thresholds import W_GRID  # noqa: E402
from pronunciation.metrics import error_level_catch  # noqa: E402
from pronunciation.phonemes import TIPS  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402
from pronunciation.thresholds import (HIDDEN_PATTERNS, V1_THRESHOLDS, apply_gop_thresholds, apply_settings,  # noqa: E402
                                      apply_thresholds, cross_validate_generalized, cross_validate_gop_thresholds,
                                      names_the_expert_error, scores, select_generalized)

OUT_DIR = ROOT / "results" / "hidden"
V2_3C_DIR = ROOT / "results" / "thresholds"
MIN_PRECISION, MAX_NATIVE_FALSE_ALARM, MAX_TURKISH_FALSE_ALARM_RISE = 0.70, 0.025, 0.02
REPORT_TYPES = {
    "Turkish pattern greedy decoding heard": lambda issue: bool(issue.tip) and not issue.hidden,
    "other difference": lambda issue: not issue.tip,
    "hidden pattern": lambda issue: issue.hidden,
}


def show(value) -> str:
    return "off" if value is None else "none" if math.isinf(value) else f"{value:g}"


def issue_keys(issues) -> list[tuple]:
    return [(i.kind, i.expected, i.heard, i.tip, i.hidden) for i in issues]


def run(system: System) -> dict:
    """The system as frozen in v2-3c and its generalization, both measured by speaker cross-validation."""
    decoder, data = evaluate_saa.load("dev", system.recognizer)
    raw = evaluate_saa.evaluate(decoder, data, System(system.name, system.recognizer, "rules", {"gop_threshold": None}))
    own = (system.settings.get("gop_threshold", V1_THRESHOLDS[0]), system.settings.get("pattern_threshold", V1_THRESHOLDS[1]))

    # v2-3c: GOP pair and the hidden-w margin, as in tune_gop_thresholds.py
    own_w = own + (system.settings.get("w_margin_threshold"),)
    assert apply_thresholds(raw, own_w).flagged.tolist() == evaluate_saa.evaluate(decoder, data, system).flagged.tolist()
    old_cv = cross_validate_gop_thresholds(raw, fallback=own_w, w_grid=W_GRID)

    # v2-3d: GOP pair as in v2, then one threshold per pattern
    pick = select_generalized(raw, fallback=own)
    gen = System(f"{system.name}-gen", system.recognizer, "rules", pick["settings"])
    selected = apply_settings(raw, gen.settings)
    direct = evaluate_saa.evaluate(decoder, data, gen)              # the offline rules must match the app's
    assert selected.flagged.tolist() == direct.flagged.tolist(), "offline settings disagree with analyze()"
    assert list(map(issue_keys, selected.our_errors)) == list(map(issue_keys, direct.our_errors))
    gen_cv = cross_validate_generalized(raw, fallback=own)
    return {"system": system, "gen": gen, "raw": raw, "own": own, "old_cv": old_cv, "pick": pick,
            "selected": selected, "gen_cv": gen_cv}


def reproduced(name: str, rows) -> str:
    """Does the v2-3c cross-validation give the per-speaker counts saved in v2-3c?"""
    path = V2_3C_DIR / f"{name}.json"
    if not path.exists():
        return "no v2-3c file"
    saved = json.loads(path.read_text())["units"]["cv"]
    return "yes" if evaluate_saa.saa_units(rows) == saved else "NO: differs from " + str(path.relative_to(ROOT))


def pattern_stats(rows) -> dict[str, dict]:
    """Per pattern, Turkish speakers: expert errors and how many we found; our reports with that tip and how many
    name the error the expert heard (all reports and the hidden ones)."""
    turkish = rows[(rows.group == "turkish") & rows.expert_wrong.notna()]
    by_tip = error_level_catch(zip(turkish.expert_errors, turkish.our_errors))["by_tip"]
    stats = {}
    for tip in TIPS:
        reports = [(i, e) for issues, e in zip(turkish.our_errors, turkish.expert_errors) for i in issues if i.tip == tip]
        hidden = [(i, e) for i, e in reports if i.hidden]
        stats[tip] = {"errors": by_tip.get(tip, {}).get("errors", 0), "found": by_tip.get(tip, {}).get("found", 0),
                      "reports": len(reports), "right": sum(names_the_expert_error(i, e) for i, e in reports),
                      "hidden_reports": len(hidden), "hidden_right": sum(names_the_expert_error(i, e) for i, e in hidden)}
    return stats


def on_expert_sound(issue, expert) -> bool:
    """The report is on a sound the expert also marked wrong (an added sound: the expert heard one too)."""
    if issue.kind == "ins":
        return any(e.kind == "ins" for e in expert)
    return any(e.kind != "ins" and e.expected == issue.expected for e in expert)


def report_types(rows) -> dict[str, dict]:
    """Precision per kind of report: word level (the expert found the word wrong), sound level (the expert
    marked the same sound) and, for patterns, diagnosis (the expert heard that pattern's error there)."""
    labelled = rows[rows.expert_wrong.notna()]
    turkish = labelled[labelled.group == "turkish"]
    native_ok = labelled[(labelled.group == "english") & ~labelled.expert_wrong.astype(bool)]
    out = {}
    for name, kind in REPORT_TYPES.items():
        has = turkish.our_errors.map(lambda issues, kind=kind: any(kind(i) for i in issues))
        reports = [(i, e) for issues, e in zip(turkish.our_errors, turkish.expert_errors) for i in issues if kind(i)]
        out[name] = {"words": int(has.sum()), "wrong_words": int(turkish.expert_wrong[has].astype(bool).sum()),
                     "reports": len(reports), "on_expert_sound": sum(on_expert_sound(i, e) for i, e in reports),
                     "right_pattern": None if name == "other difference"
                     else sum(names_the_expert_error(i, e) for i, e in reports),
                     "native_words": int(native_ok.our_errors.map(lambda issues, kind=kind: any(kind(i) for i in issues)).sum()),
                     "native_correct_words": len(native_ok)}
    return out


def summary(rows) -> dict:
    result = scores(rows)
    result["ci95"] = {group: evaluate_saa.with_ci(rows[rows.group == group], "expert_wrong")["ci95"]
                      for group in ("turkish", "english")}
    result["patterns"] = pattern_stats(rows)
    result["report_types"] = report_types(rows)
    return result


def cell(s: dict, key: str) -> str:
    group, metric = {"precision": ("turkish", "precision"), "recall": ("turkish", "recall"), "f1": ("turkish", "f1"),
                     "english_false_alarm": ("english", "false_alarm"),
                     "turkish_false_alarm": ("turkish", "false_alarm")}[key]
    low, high = s["ci95"][group][metric]
    if key == "f1":
        return f"{s[key]:.3f} [{low:.3f}–{high:.3f}]"
    return f"{s[key]:.1%} [{low:.1%}–{high:.1%}]"


def print_gop_step(results: list[dict]) -> None:
    print("\n## 1. GOP thresholds (as in v2, without hidden errors)\n")
    for r in results:
        s = r["pick"]["settings"]
        t, p = s["gop_threshold"], s["pattern_threshold"]
        folds = ", ".join(f"{show(f['settings']['gop_threshold'])} / {show(f['settings']['pattern_threshold'])}"
                          + ("*" if f["fallback"] else "") for f in r["gen_cv"]["folds"])
        print(f"{r['gen'].name}: other differences / typical errors = {show(t)} / {show(p)} on all dev speakers"
              + (" (no pair within the limits: the system's own)" if r["pick"]["fallback"] else "")
              + f"; per fold: {folds}")
        if t == 0.0:
            # GOP (lpr) is never above 0: the only looser setting is no check at all, which the grid also has
            none = apply_gop_thresholds(r["raw"], math.inf, p)
            zero = apply_gop_thresholds(r["raw"], 0.0, p)
            differ = zero.flagged != none.flagged
            print(f"  0.0 is the loosest finite value of the grid; the only looser one, no check, was tried too "
                  f"(native false alarm {scores(none)['english_false_alarm']:.1%}, limit 2.5%). Between them "
                  f"{int(differ.sum())} words differ ({int((differ & (r['raw'].group == 'english')).sum())} native): "
                  f"words with GOP exactly 0, in which every expected sound was the recognizer's first choice "
                  f"(blank aside) in at least one of its frames.")


def print_pattern_thresholds(results: list[dict]) -> None:
    print("\n## 2. Pattern thresholds (margin; off = pattern not reported when greedy decoding wrote the expected sound)\n")
    for r in results:
        folds = r["gen_cv"]["folds"]
        print(f"{r['gen'].name}:\n")
        print("| Pattern | all dev speakers | " + " | ".join(f"fold {f['fold'] + 1}" for f in folds) + " |")
        print("| --- | --- |" + " --- |" * len(folds))
        for tip in HIDDEN_PATTERNS:
            values = [r["pick"]["settings"]["hidden_thresholds"][tip]] + [f["settings"]["hidden_thresholds"][tip] for f in folds]
            print(f"| {TIPS[tip]['title']} | " + " | ".join(show(v) for v in values) + " |")
        print()


def print_dev_table(named: list[tuple[str, dict]]) -> None:
    print("\n## 3. SAA dev, honest (speaker cross-validation; 95% speaker-bootstrap intervals)\n")
    print("| System | Turkish precision | recall | F1 | native false alarm | false alarm on Turkish speakers' correct words |")
    print("| --- | --- | --- | --- | --- | --- |")
    for label, s in named:
        print(f"| {label} | " + " | ".join(cell(s, k) for k in ("precision", "recall", "f1", "english_false_alarm",
                                                              "turkish_false_alarm")) + " |")


def print_differences(pairs: list[tuple[str, dict, str, dict]]) -> dict:
    compared = {}
    for name_a, units_a, name_b, units_b in pairs:
        rows = compare_experiments.compare({"name": name_a, "units": units_a}, {"name": name_b, "units": units_b})
        compared[(name_a, name_b)] = {row["metric"]: row for row in rows}
        print(f"\n{name_a} -> {name_b} (paired bootstrap over speakers)\n")
        print(f"| Metric | {name_a} | {name_b} | Difference | 95% CI | Real? |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            low, high = row["ci95"]
            print(f"| {row['metric']} | {row['a']:.1%} | {row['b']:.1%} | {row['diff']:+.1%} | "
                  f"[{low:+.1%}, {high:+.1%}] | {'yes' if row['real'] else 'no'} |")
    return compared


def decide(frozen: dict, gen: dict, diff: dict) -> dict:
    recall = diff["SAA dev, Turkish: recall"]
    turkish_fa = diff["SAA dev, Turkish: false alarm"]
    checks = [
        ("recall gain over frozen v2 is real (95% interval above 0)",
         f"{recall['diff']:+.1%} [{recall['ci95'][0]:+.1%}, {recall['ci95'][1]:+.1%}]", recall["ci95"][0] > 0),
        ("Turkish precision >= 70% (CV)", f"{gen['precision']:.1%}", gen["precision"] >= MIN_PRECISION),
        ("native false alarm <= 2.5% (CV)", f"{gen['english_false_alarm']:.1%}",
         gen["english_false_alarm"] <= MAX_NATIVE_FALSE_ALARM),
        ("false alarm on Turkish speakers' correct words rises by <= 2 points",
         f"{frozen['turkish_false_alarm']:.1%} -> {gen['turkish_false_alarm']:.1%} ({turkish_fa['diff'] * 100:+.1f} points)",
         turkish_fa["diff"] <= MAX_TURKISH_FALSE_ALARM_RISE + 1e-12),
    ]
    return {"checks": [{"check": c, "value": v, "met": bool(m)} for c, v, m in checks],
            "replace": all(m for _, _, m in checks)}


def print_patterns(named: list[tuple[str, dict]]) -> None:
    print("\n## 5. Per pattern, Turkish speakers (CV): catch = expert errors we found on the same sound; "
          "diagnosis = our reports with the tip that name the error the expert heard there\n")
    print("| Pattern | expert errors | " + " | ".join(f"catch: {n}" for n, _ in named) + " | "
          + " | ".join(f"diagnosis: {n}" for n, _ in named) + " |")
    print("| --- | --- |" + " --- |" * (2 * len(named)))
    first = named[0][1]["patterns"]
    for tip in TIPS:
        stats = [s["patterns"][tip] for _, s in named]
        if not first[tip]["errors"] and not any(st["reports"] for st in stats):
            continue
        catch = [f"{st['found'] / st['errors']:.0%}" if st["errors"] else "–" for st in stats]
        diag = [f"{st['right'] / st['reports']:.0%} of {st['reports']}" if st["reports"] else "–" for st in stats]
        print(f"| {TIPS[tip]['title']} | {first[tip]['errors']} | " + " | ".join(catch) + " | " + " | ".join(diag) + " |")
    print("\nHidden reports only (right / all, Turkish speakers):")
    for name, s in named:
        hidden = [f"{tip} {st['hidden_right']}/{st['hidden_reports']}" for tip, st in s["patterns"].items() if st["hidden_reports"]]
        print(f"  {name}: " + (", ".join(hidden) if hidden else "none"))


def print_report_types(named: list[tuple[str, dict]]) -> None:
    print("\n## 6. Precision by kind of report (CV). Word: Turkish words with such a report that the expert found wrong. "
          "Sound: reports on a sound the expert marked. Pattern: reports naming the expert's error. "
          "Native: correct native words with such a report.\n")
    print("| System | Kind of report | word precision | sound precision | pattern (diagnosis) precision | native words |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, s in named:
        for kind, t in s["report_types"].items():
            word = f"{t['wrong_words'] / t['words']:.0%} of {t['words']}" if t["words"] else "–"
            sound = f"{t['on_expert_sound'] / t['reports']:.0%} of {t['reports']}" if t["reports"] else "–"
            pattern = "–" if t["right_pattern"] is None or not t["reports"] else f"{t['right_pattern'] / t['reports']:.0%}"
            print(f"| {name} | {kind} | {word} | {sound} | {pattern} | {t['native_words']} of {t['native_correct_words']} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", type=Path, help="systems as frozen in v2-3c (experiments/v1.json ...)")
    parser.add_argument("--frozen", default="v2", help="the frozen v2 among them (the decision compares with it)")
    parser.add_argument("--write-experiments", action="store_true", help="write experiments/<name>-gen.json")
    args = parser.parse_args()

    results = [run(load_system(path)) for path in args.systems]
    named, units = [], {}
    for r in results:
        name, gen = r["system"].name, r["gen"].name
        print(f"{name}: v2-3c cross-validation reproduced: {reproduced(name, r['old_cv']['rows'])}")
        r["old_summary"], r["gen_summary"] = summary(r["old_cv"]["rows"]), summary(r["gen_cv"]["rows"])
        units[name], units[gen] = evaluate_saa.saa_units(r["old_cv"]["rows"]), evaluate_saa.saa_units(r["gen_cv"]["rows"])
        named += [(f"{name} (v2-3c rules)", r["old_summary"]), (gen, r["gen_summary"])]

    print_gop_step(results)
    print_pattern_thresholds(results)
    print_dev_table(named)
    print("\n## 4. Differences\n")
    frozen = next(r for r in results if r["system"].name == args.frozen)
    pairs = [(frozen["system"].name, units[frozen["system"].name], frozen["gen"].name, units[frozen["gen"].name])]
    for a, b in itertools.combinations(results, 2):
        pairs += [(a[k].name, units[a[k].name], b[k].name, units[b[k].name]) for k in ("system", "gen")]
    compared = print_differences(pairs)
    decision = decide(frozen["old_summary"], frozen["gen_summary"], compared[(frozen["system"].name, frozen["gen"].name)])
    print(f"\nDecision rule ({frozen['gen'].name} replaces frozen {frozen['system'].name} only if all hold):")
    for c in decision["checks"]:
        print(f"  [{'x' if c['met'] else ' '}] {c['check']}: {c['value']}")
    print(f"  -> {'replace frozen v2' if decision['replace'] else 'keep frozen v2'}")
    print_patterns(named)
    print_report_types(named)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        gen = r["gen"]
        out = {"name": gen.name, "recognizer": gen.recognizer, "settings": gen.settings, "fallback": r["pick"]["fallback"],
               "selected": summary(r["selected"]), "cv": r["gen_summary"], "v2_3c_cv": r["old_summary"],
               "folds": r["gen_cv"]["folds"], "units": {"cv": units[gen.name], "v2_3c_cv": units[r["system"].name]}}
        (OUT_DIR / f"{gen.name}.json").write_text(json.dumps(out, indent=2, default=float))
        if args.write_experiments:
            system = {"name": gen.name, "recognizer": gen.recognizer, "decision": "rules", "settings": gen.settings}
            (ROOT / "experiments" / f"{gen.name}.json").write_text(json.dumps(system, indent=2) + "\n")
    (OUT_DIR / "decision.json").write_text(json.dumps({"frozen": frozen["system"].name, "generalized": frozen["gen"].name,
                                                       **decision}, indent=2))
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
