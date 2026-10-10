#!/usr/bin/env python3
"""Execute the registered T01--T13 sign-test/TOST decision table.

Input is a frozen contrast table whose values are all oriented beneficial-
positive.  Producing that table is separate from inference so no analysis code
can choose comparators after observing outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HOLM_IDS = [f"T{index:02d}" for index in range(1, 11)]
OUTSIDE_IDS = ["T11", "T12", "T13"]


def binomial_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


def sign_test(values: list[float], *, two_sided: bool = True, boundary: float = 0.0) -> dict:
    ties = sum(value == boundary for value in values)
    if ties > 2:
        return {"n": len(values), "positive": 0, "negative": len(values), "ties": ties, "untestable": True, "p_value": 1.0}
    positive = sum(value > boundary for value in values)
    negative = len(values) - positive
    tail = binomial_cdf(min(positive, negative), len(values)) if values else 1.0
    p_value = min(1.0, 2.0 * tail) if two_sided else (binomial_cdf(negative, len(values)) if values else 1.0)
    return {"n": len(values), "positive": positive, "negative": negative, "ties": ties, "untestable": False, "p_value": p_value}


def order_statistic_interval(values: list[float], alpha: float) -> list[float | None]:
    ordered = sorted(values)
    n = len(ordered)
    admissible = [k for k in range(n // 2 + 1) if 2.0 * binomial_cdf(k, n) <= alpha]
    if not admissible:
        return [None, None]
    k = max(admissible)
    return [ordered[k], ordered[n - k - 1]]


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    active = True
    result = {}
    for rank, key in enumerate(ordered):
        threshold = alpha / (len(ordered) - rank)
        reject = active and p_values[key] <= threshold
        active = active and reject
        result[key] = {"p_value": p_values[key], "threshold": threshold, "reject": reject}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--decision-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.contrasts.read_text(encoding="utf-8"))
    registered = json.loads(args.decision_table.read_text(encoding="utf-8"))
    # DECISION_TABLE.json carries no flat `test_ids` key; the registered ids and their order live
    # in `family_of_record` (the ten-id Holm family) and `outside_family`, with `tests` a list of
    # definitions. Both are checked, and cross-checked against each other, so id drift in either
    # place still stops the run — which is what the original guard was for.
    if registered.get("family_of_record") != HOLM_IDS:
        raise RuntimeError("DECISION_TABLE family_of_record test-id order drift")
    if registered.get("outside_family") != OUTSIDE_IDS:
        raise RuntimeError("DECISION_TABLE outside_family test-id order drift")
    defined = [test["id"] for test in registered["tests"]]
    if defined != HOLM_IDS + OUTSIDE_IDS:
        raise RuntimeError("DECISION_TABLE tests[].id order drift against the registered families")

    tests = payload.get("tests", {})
    unknown = set(tests) - set(HOLM_IDS + OUTSIDE_IDS)
    if unknown:
        raise RuntimeError(f"contrast input holds unregistered test ids {sorted(unknown)}")
    # A test whose runs were never executed is reported as not_executed with the builder's own
    # reason. It is never silently dropped, never imputed, and never allowed to read as a pass:
    # any gate that depends on it becomes null, and `success` becomes null with it.
    not_computable = payload.get("not_computable", {})
    absent = [key for key in HOLM_IDS + OUTSIDE_IDS if key not in tests]
    not_executed = {
        key: {
            "status": "not_executed",
            "reason": not_computable.get(key, "absent from the contrast table with no reason recorded"),
        }
        for key in absent
    }

    results = {}
    superiority_p = {}
    equivalence_p = {}
    for identifier in [key for key in HOLM_IDS if key in tests]:
        values = [float(value) for value in tests[identifier]["beneficial_positive_values"]]
        if len(values) != 18 or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{identifier} requires 18 finite matched-replicate contrasts")
        test = sign_test(values)
        superiority_p[identifier] = test["p_value"]
        lower = sign_test(values, two_sided=False, boundary=-0.03)["p_value"]
        upper = sign_test([-value for value in values], two_sided=False, boundary=-0.03)["p_value"]
        equivalence_p[identifier] = max(lower, upper)
        results[identifier] = {
            **test,
            "median": (sorted(values)[8] + sorted(values)[9]) / 2.0,
            "order_statistic_interval_alpha_0.005": order_statistic_interval(values, 0.005),
            "tost_raw_p": max(lower, upper),
        }
    # Holm over a partial family is NOT the registered procedure: the registered cutoff is
    # alpha/10 over all ten ids, and correcting over fewer tests would make each surviving test
    # easier to reject. The adjustment is still reported for description, but the family is marked
    # incomplete and gate (b) is refused below rather than computed on a smaller family.
    holm_family_complete = set(superiority_p) == set(HOLM_IDS)
    superiority_holm = holm(superiority_p)
    equivalence_holm = holm(equivalence_p)
    for identifier in superiority_p:
        results[identifier]["superiority_holm"] = superiority_holm[identifier]
        results[identifier]["equivalence_holm"] = equivalence_holm[identifier]
        results[identifier]["holm_family_complete"] = holm_family_complete
    for identifier in [key for key in OUTSIDE_IDS if key in tests]:
        values = [float(value) for value in tests[identifier]["beneficial_positive_values"]]
        if len(values) != 18 or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{identifier} requires 18 finite matched-replicate contrasts")
        margin = float(tests[identifier].get("margin", 0.0))
        results[identifier] = sign_test(values, two_sided=False, boundary=-margin)
        results[identifier]["margin"] = margin
    def gate(required: list[str], decide) -> tuple[bool | None, str | None]:
        """A gate over a test that was never executed is null, never False and never True.

        False would read as 'the registered criterion was evaluated and not met'; True would be a
        fabricated pass. Neither is true of a contrast whose arms never ran, so the gate reports
        null and names what is missing.
        """
        blocked = [key for key in required if key not in results]
        if blocked:
            return None, "; ".join(f"{key}: {not_executed[key]['reason']}" for key in blocked)
        return decide(), None

    gate_b, gate_b_reason = gate(
        HOLM_IDS[:5],
        lambda: holm_family_complete
        and all(results[key]["superiority_holm"]["reject"] and results[key]["positive"] >= 16
                for key in HOLM_IDS[:5]),
    )
    if gate_b is not None and not holm_family_complete:
        # T01-T05 may all be present while T06-T10 are not; the Holm cutoff of record is defined
        # over the whole ten-id family, so the gate is not evaluable even though its five tests are.
        gate_b, gate_b_reason = None, (
            "the registered Holm cutoff alpha/10 is defined over the complete ten-id family; "
            f"absent: {sorted(set(HOLM_IDS) - set(superiority_p))}"
        )
    gate_c, gate_c_reason = gate(
        ["T06"],
        lambda: results["T06"]["superiority_holm"]["reject"] and results["T06"]["positive"] >= 16,
    )
    gate_d, gate_d_reason = gate(
        ["T11", "T13"],
        lambda: results["T11"]["positive"] >= 13 and results["T13"]["positive"] >= 13,
    )
    gates = {"b": gate_b, "c": gate_c, "d": gate_d}
    blocked_gates = {
        key: reason
        for key, reason in (("b", gate_b_reason), ("c", gate_c_reason), ("d", gate_d_reason))
        if reason is not None
    }
    output = {
        "version": 1,
        "results": results,
        "not_executed": not_executed,
        "gates": gates,
        # SUCCESS = (b) AND (c) AND (d) is the only success rule. With any gate null the
        # conjunction is undetermined, so `success` is null and `outcome` says so explicitly
        # rather than degrading to false, which would read as a registered refutation.
        "success": None if any(value is None for value in gates.values())
        else bool(gate_b and gate_c and gate_d),
        "outcome": "NOT_EVALUABLE" if any(value is None for value in gates.values()) else None,
        "gates_not_evaluable": blocked_gates,
        "holm_family_complete": holm_family_complete,
        "residual_warning": "T06 interval coverage under the registered dependence band remains unresolved.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["gates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
