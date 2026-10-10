#!/usr/bin/env python3
"""Build the frozen contrast table that `analyze_registered_tests.py` consumes.

This script is the only path from per-run evaluation output to the decision table, and it is
deliberately the *only* place where comparators are named.  Producing the table is separate from
running the tests so that no analysis code can choose a comparator after seeing an outcome: the
arm pairs below are transcribed from `refine-logs/DECISION_TABLE.json`, and the aggregation order
is transcribed from `MATH_SPEC.json` eq4 -> eq6.

What it refuses to do
---------------------
* It never pads, imputes or drops a seed to reach 18.  A contrast whose two arms do not both have
  all 18 accepted replicates is reported under `not_computable` with the missing run names, not
  as a short vector.
* It never invents the LowRes/HighRes language partition.  `refine-logs/LANGUAGE_FREEZE.json`
  — which T13 requires to be frozen *before any evaluation label exists* — does not exist, and
  evaluation labels already do.  Creating it now would be a post-hoc choice of the statistic's own
  trimming set, so T11, T12 and T13 are reported as not computable and gate (d) is not evaluable.
* It never reports a confirmatory verdict.  The block runs under the 2026-09-19 operator hard-open
  stamp with no KEY_AUDIT and no frozen split: everything here is descriptive.

Aggregation order (eq4 -> eq6, unchanged)
-----------------------------------------
per-language selective risk at kappa = 0.8 under the top-k rank acceptance rule (written by
`evaluate_vqa.py` as `per_language_selective_risk`)
  -> per-language contrast, oriented beneficial-positive as SR(comparator) - SR(Joint) because
     selective risk is a loss
  -> trimmed language mean with T = ceil(|L|/4) over the frozen endpoint language list
     (|L| = 12, T = 3 as frozen in the endpoint receipt; for |L| = 12 this trimmed mean is exactly
     the interquartile mean eq6 registers)
  -> 18 matched seeds, one value per seed, paired within seed.

Validation pass
---------------
`queue_manager_project40.py::job_status_check` marks a job completed whenever its
`expected_output` path exists, without inspecting the exit code, so a truncated or failed
evaluation can be carried as "completed" by the queue.  Every summary read here is therefore
re-checked against the frozen endpoint receipt (item count, input sha256, coverage, language set,
finiteness, `full_generation`) and anything that fails is excluded and named in `validation`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path

# Transcribed from refine-logs/DECISION_TABLE.json.  `Joint` is the innovated arm in every pair and
# the comparator is the second element; the orientation below turns each into beneficial-positive.
PAIRED_TESTS = {
    "T01": ("Joint", "Random"),
    "T02": ("Joint", "LangOnlyCond"),
    "T03": ("Joint", "LangOnly"),
    "T04": ("Joint", "ConceptOnly"),
    "T05": ("Joint", "EnglishHeavy"),
}

# eq6: I = IQM_l(R_Joint - R_LangOnly - R_ConceptOnly + R_NatNat), a negative I being the
# beneficial direction, so the beneficial-positive per-language value is the negation.
INTERACTION_ARMS = ("Joint", "LangOnly", "ConceptOnly", "NatNat")

NOT_COMPUTABLE = {
    "T06": (
        "the registered estimand is the median WITHIN-UNIT lambda effect over (E4-eligible vertex "
        "x training seed) pairs, both lambda drawn on the SAME vertex (OCCUPANCY_VERTICES.json, "
        "occupancy/vertex_sampler.py). The executed block has arm-level HighOcc and LowOcc runs "
        "over the ordinary seed list and no vertex draw, so it contains no within-vertex pair. "
        "Gate (c) is therefore not evaluable; the arm-level HighOcc-LowOcc difference is reported "
        "below under `descriptive_only` and is NOT T06."
    ),
    "T08": "arm COINCIDE was never run (PREREG_DEVIATIONS D8: dropped, not aliased onto another arm).",
    "T09": "arm CGMix was never run (PREREG_DEVIATIONS D8: dropped, not aliased onto another arm).",
    "T10": (
        "the accuracy-mediated decomposition has no executable mediator binding in MATH_SPEC.json; "
        "per-language accuracy is emitted below as a descriptive table instead of being aggregated "
        "into a test whose form is not registered executably."
    ),
    "T11": (
        "requires the aggregate HIGH-resource language set. No frozen high/low-resource partition "
        "of the 12 endpoint languages exists (refine-logs/LANGUAGE_FREEZE.json is absent) and "
        "evaluation labels already exist, so any partition chosen now would be post-hoc. Gate (d) "
        "is not evaluable."
    ),
    "T12": (
        "requires an unseen-entity flag per evaluation item. predictions.jsonl carries id, "
        "language, gold/prediction index and confidence only; the endpoint receipt records "
        "train_concept_overlap = 0 for the whole set, which is a set-level property and does not "
        "induce the per-item seen/unseen split this test reads."
    ),
    "T13": (
        "requires T = ceil(|L|/4) over the LowRes languages surviving the R19-F1 retention floor, "
        "'computed once and FROZEN in refine-logs/LANGUAGE_FREEZE.json before any evaluation label "
        "exists, never recomputed per replicate'. That file does not exist and evaluation labels "
        "already exist. Gate (d) is not evaluable."
    ),
}


def trimmed_language_mean(values: list[float], trim: int) -> float:
    ordered = sorted(values)
    kept = ordered[trim: len(ordered) - trim]
    if not kept:
        raise RuntimeError(f"trim {trim} leaves nothing of {len(values)} languages")
    return sum(kept) / len(kept)


def run_name(arm: str, seed: int) -> str:
    return f"{arm.lower()}-s{seed}"


def load_summary(path: Path, receipt: dict) -> tuple[dict | None, list[str]]:
    """Return (summary, reasons-it-was-rejected).  A rejected summary is never used."""
    if not path.exists():
        return None, ["eval_summary.json absent"]
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"eval_summary.json is not valid JSON ({exc.msg})"]

    reasons: list[str] = []
    if summary.get("status") != "completed":
        reasons.append(f"status={summary.get('status')!r}")
    if not summary.get("full_generation"):
        reasons.append("full_generation is not true")
    if summary.get("n_items") != receipt["items"]:
        reasons.append(f"n_items={summary.get('n_items')} != endpoint {receipt['items']}")
    if summary.get("input_sha256") != receipt["sha256"]:
        reasons.append("input_sha256 does not match the frozen endpoint")
    if summary.get("target_coverage") != receipt["target_coverage"]:
        reasons.append(
            f"target_coverage={summary.get('target_coverage')} != {receipt['target_coverage']}"
        )
    risk = summary.get("per_language_selective_risk") or {}
    if set(risk) != set(receipt["languages"]):
        reasons.append("per_language_selective_risk does not cover exactly the frozen language list")
    elif any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in risk.values()):
        reasons.append("per_language_selective_risk holds a non-finite value")
    return (None, reasons) if reasons else (summary, [])


def per_language_accuracy(path: Path, languages: list[str]) -> dict[str, float] | None:
    """Descriptive only — the accuracy table the paper reports beside the endpoint."""
    if not path.exists():
        return None
    hits: dict[str, int] = {language: 0 for language in languages}
    seen: dict[str, int] = {language: 0 for language in languages}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            language = record.get("language")
            if language not in seen:
                continue
            seen[language] += 1
            hits[language] += 1 if record.get("correct") else 0
    if not all(seen.values()):
        return None
    return {language: hits[language] / seen[language] for language in languages}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals-root", type=Path, required=True)
    parser.add_argument("--endpoint-receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--with-accuracy",
        action="store_true",
        help="also read every predictions.jsonl to emit the descriptive per-language accuracy table",
    )
    args = parser.parse_args()

    receipt = json.loads(args.endpoint_receipt.read_text(encoding="utf-8"))
    languages = list(receipt["languages"])
    trim = int(receipt["trim_T"])
    if trim != math.ceil(len(languages) / 4):
        raise RuntimeError(
            f"endpoint receipt trim_T={trim} contradicts the registered T = ceil(|L|/4) = "
            f"{math.ceil(len(languages) / 4)} for |L| = {len(languages)}"
        )

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry.get("version") != 2 or not registry.get("budget_is_binding"):
        raise SystemExit("refusing: registry is not a binding-budget registry")
    arms = sorted({subset["arm"] for subset in registry["subsets"]})
    seeds = sorted({subset["seed"] for subset in registry["subsets"]})

    accepted: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    rejected: dict[str, list[str]] = {}
    conformance_failures: dict[str, int] = {}
    accuracy: dict[str, dict[str, float]] = {}

    for arm in arms:
        for seed in seeds:
            name = run_name(arm, seed)
            directory = args.evals_root / name
            summary, reasons = load_summary(directory / "eval_summary.json", receipt)
            if summary is None:
                if reasons == ["eval_summary.json absent"]:
                    missing.append(name)
                else:
                    rejected[name] = reasons
                continue
            accepted[(arm, seed)] = summary
            failures = summary.get("conformance_failures") or []
            if failures:
                conformance_failures[name] = len(failures)
            if args.with_accuracy:
                table = per_language_accuracy(directory / "predictions.jsonl", languages)
                if table is not None:
                    accuracy[name] = table

    tests: dict[str, dict] = {}
    not_computable = dict(NOT_COMPUTABLE)

    def complete(required: tuple[str, ...]) -> tuple[list[int], list[str]]:
        usable, absent = [], []
        for seed in seeds:
            gaps = [run_name(arm, seed) for arm in required if (arm, seed) not in accepted]
            if gaps:
                absent.extend(gaps)
            else:
                usable.append(seed)
        return usable, absent

    for identifier, (innovated, comparator) in PAIRED_TESTS.items():
        usable, absent = complete((innovated, comparator))
        if len(usable) != len(seeds):
            not_computable[identifier] = (
                f"{len(usable)} of {len(seeds)} matched replicates available; "
                f"no value is emitted for a short vector. Missing runs: {sorted(set(absent))}"
            )
            continue
        per_language: dict[int, dict[str, float]] = {}
        values: list[float] = []
        for seed in usable:
            innovated_risk = accepted[(innovated, seed)]["per_language_selective_risk"]
            comparator_risk = accepted[(comparator, seed)]["per_language_selective_risk"]
            contrast = {
                language: comparator_risk[language] - innovated_risk[language]
                for language in languages
            }
            per_language[seed] = contrast
            values.append(trimmed_language_mean(list(contrast.values()), trim))
        tests[identifier] = {
            "contrast": f"{innovated} minus {comparator} on per-language selective risk at kappa=0.8",
            "orientation": f"beneficial-positive as SR({comparator}) - SR({innovated})",
            "seeds": usable,
            "beneficial_positive_values": values,
            "per_language_contrast": {str(seed): per_language[seed] for seed in usable},
        }

    usable, absent = complete(INTERACTION_ARMS)
    if len(usable) != len(seeds):
        not_computable["T07"] = (
            f"{len(usable)} of {len(seeds)} matched replicates available across "
            f"{INTERACTION_ARMS}. Missing runs: {sorted(set(absent))}"
        )
    else:
        per_language = {}
        values = []
        for seed in usable:
            risk = {arm: accepted[(arm, seed)]["per_language_selective_risk"] for arm in INTERACTION_ARMS}
            contrast = {
                language: -(
                    risk["Joint"][language]
                    - risk["LangOnly"][language]
                    - risk["ConceptOnly"][language]
                    + risk["NatNat"][language]
                )
                for language in languages
            }
            per_language[seed] = contrast
            values.append(trimmed_language_mean(list(contrast.values()), trim))
        tests["T07"] = {
            "contrast": "eq6 two-factor marginal-balance interaction over Joint/LangOnly/ConceptOnly/NatNat",
            "orientation": "beneficial-positive as -(R_Joint - R_LangOnly - R_ConceptOnly + R_NatNat)",
            "seeds": usable,
            "beneficial_positive_values": values,
            "per_language_contrast": {str(seed): per_language[seed] for seed in usable},
        }

    descriptive_only: dict[str, dict] = {}
    occupancy_usable, occupancy_absent = complete(("HighOcc", "LowOcc"))
    if len(occupancy_usable) == len(seeds):
        values = []
        for seed in occupancy_usable:
            high = accepted[("HighOcc", seed)]["per_language_selective_risk"]
            low = accepted[("LowOcc", seed)]["per_language_selective_risk"]
            contrast = {language: low[language] - high[language] for language in languages}
            values.append(trimmed_language_mean(list(contrast.values()), trim))
        descriptive_only["arm_level_occupancy"] = {
            "contrast": "HighOcc minus LowOcc at the arm level, matched by seed",
            "orientation": "beneficial-positive as SR(LowOcc) - SR(HighOcc)",
            "seeds": occupancy_usable,
            "values": values,
            "warning": (
                "this is NOT test T06 and may not be substituted for it: T06 is a within-vertex "
                "contrast and no vertex was drawn. It carries no gate role and supports no "
                "occupancy claim."
            ),
        }
    elif occupancy_absent:
        descriptive_only["arm_level_occupancy"] = {
            "unavailable": f"missing runs: {sorted(set(occupancy_absent))}"
        }

    payload = {
        "version": 1,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "descriptive_hard_open_not_confirmatory",
        "status_note": (
            "Produced under the 2026-09-19 operator hard-open stamp with no KEY_AUDIT and no frozen "
            "split. SUCCESS = (b) AND (c) AND (d) is NOT evaluable from this block: gate (b) needs "
            "the whole ten-id Holm family and T08/T09 were never run, gate (c) needs the within-vertex "
            "T06, and gate (d) needs T11 and T13, both of which need a language freeze that does not "
            "exist. Nothing here may be reported as confirmatory."
        ),
        "endpoint": {
            "sha256": receipt["sha256"],
            "languages": languages,
            "language_count": len(languages),
            "trim_T": trim,
            "target_coverage": receipt["target_coverage"],
            "items": receipt["items"],
        },
        "aggregation": (
            "per-language selective risk at kappa=0.8 (eq4) -> beneficial-positive per-language "
            f"contrast -> trimmed language mean with T={trim} over the {len(languages)} frozen "
            "endpoint languages (for |L|=12 this equals the eq6 interquartile mean) -> one value "
            "per matched seed"
        ),
        "arms": arms,
        "seeds": seeds,
        "tests": tests,
        "not_computable": not_computable,
        "descriptive_only": descriptive_only,
        "validation": {
            "expected_runs": len(arms) * len(seeds),
            "accepted": len(accepted),
            "missing": sorted(missing),
            "rejected": rejected,
            "conformance_failure_counts": conformance_failures,
            "note": (
                "the queue marks a job completed on expected_output existence alone, so every "
                "summary is re-checked here against the frozen endpoint receipt"
            ),
        },
    }
    if accuracy:
        payload["descriptive_per_language_accuracy"] = accuracy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "accepted": len(accepted),
        "expected": len(arms) * len(seeds),
        "missing": len(missing),
        "rejected": len(rejected),
        "tests_emitted": sorted(tests),
        "not_computable": sorted(not_computable),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
