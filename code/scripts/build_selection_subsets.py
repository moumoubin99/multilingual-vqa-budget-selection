#!/usr/bin/env python3
"""Rebuilt fixed-budget selection-policy subset generator.

Replaces `scripts/materialize_hardopen_subsets.py`, which produced 61 byte-distinct
files that were all the same 9 940 rows in different order. That script is kept
unmodified as the artifact of record for the 2026-09-22 blocking finding.

Two defects are fixed here, and both are turned into hard refusals rather than
silent degradations:

1.  **Budget was not binding.** `budget = min(args.budget, len(pool))` with
    `--budget 20000` against a 9 940-row pool made the budget equal to the pool, so
    every `take(pool, budget)` returned the entire pool, merely shuffled. A selection
    policy that is handed its whole universe is not a selection policy. Here the
    budget must be strictly smaller than the pool, and smaller than the tightest
    per-arm feasibility bound, or the run refuses to start.

2.  **Three named arms had no implementation.** `cgmix`, `coincide` and `nnselect`
    matched no branch in `sample_rows` and fell through to the uniform-random
    default, so three "baselines" were literally the random policy under another
    name. They now raise instead of silently aliasing.

A third defect is fixed non-fatally: rows carry a real `image_sha256`, so
train/eval disjointness is checkable by content instead of by path.

Every arm returns exactly `--budget` rows, which is what makes the occupancy
contrast a contrast in occupancy (lambda) alone rather than in training-set size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Arms named in the queue manifest that have no defined selection rule. Listing
# them here is what keeps them from quietly becoming `random` again.
UNIMPLEMENTED_KINDS = {
    "cgmix": "CG-Mix has no selection rule in METHOD_DESIGN.md or FINAL_PROPOSAL.md",
    "coincide": "COINCIDE requires the clustered-influence scoring it is named for",
    "nnselect": "NNSelect requires a nearest-neighbour embedding score",
}

IMPLEMENTED_KINDS = (
    "random", "joint", "englishheavy", "lowocc", "highocc",
    "natnat", "langonly", "conceptonly", "langonlycond",
)

ENGLISHHEAVY_EN_SHARE = 0.70
LOWOCC_CONCEPT_FRACTION = 0.25
HIGHOCC_CONCEPT_FRACTION = 0.85


class InfeasibleBudget(RuntimeError):
    """The requested budget cannot be met by this arm without degenerating."""


def sha256_file(path: Path, _cache: dict[str, str] = {}) -> str | None:
    key = str(path)
    if key in _cache:
        return _cache[key]
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    if digest is not None:
        _cache[key] = digest
    return digest


def row_id_signature(rows: list[dict]) -> str:
    """Order-independent fingerprint of WHICH rows a subset contains.

    Order-dependent hashes (what `input_sha256` records) cannot tell two arms
    apart when they hold the same rows in different order — that is precisely how
    the identical-dataset defect stayed invisible for 48 completed runs.
    """
    joined = "\n".join(sorted(str(r["id"]) for r in rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_pool(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty pool: {path}")
    return rows


def enrich(rows: list[dict]) -> list[dict]:
    for row in rows:
        if not row.get("image_sha256"):
            row["image_sha256"] = sha256_file(Path(row["image_path"]))
    return rows


def sample_random(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    return rng.sample(pool, budget)


def sample_joint(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    """Maximise joint (concept x language) support at the fixed budget.

    Round-robin one row per occupied cell per pass, so the number of distinct
    cells touched is as large as the budget allows. This is the arm the idea card
    describes as joint stratification by entity and language.
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pool:
        cells[(str(row["concept_id"]), str(row["language"]))].append(row)
    order = sorted(cells)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(cells[key])
    chosen: list[dict] = []
    depth = 0
    while len(chosen) < budget:
        progressed = False
        for key in order:
            bucket = cells[key]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                progressed = True
                if len(chosen) == budget:
                    return chosen
        if not progressed:
            raise InfeasibleBudget(f"joint: pool exhausted at {len(chosen)} < budget {budget}")
        depth += 1
    return chosen


def sample_englishheavy(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    """A genuine language skew: ENGLISHHEAVY_EN_SHARE of the budget is English.

    The old code capped `n_en` at the number of English rows available, which at a
    non-binding budget silently produced a perfectly uniform language marginal —
    the arm was uniform, not English-heavy. Refusing is the correct behaviour when
    the skew cannot be realised.
    """
    english = [r for r in pool if str(r["language"]) == "en"]
    rest = [r for r in pool if str(r["language"]) != "en"]
    n_en = round(budget * ENGLISHHEAVY_EN_SHARE)
    n_rest = budget - n_en
    if n_en > len(english):
        raise InfeasibleBudget(
            f"englishheavy: needs {n_en} English rows for a {ENGLISHHEAVY_EN_SHARE:.0%} share "
            f"at budget {budget}, pool has {len(english)}. Lower --budget to at most "
            f"{int(len(english) / ENGLISHHEAVY_EN_SHARE)}."
        )
    if n_rest > len(rest):
        raise InfeasibleBudget(f"englishheavy: needs {n_rest} non-English rows, pool has {len(rest)}")
    return rng.sample(english, n_en) + rng.sample(rest, n_rest)


def sample_occupancy(pool: list[dict], rng: random.Random, budget: int, fraction: float, kind: str) -> list[dict]:
    """Restrict concept support to `fraction` of concepts, then draw exactly `budget` rows.

    Occupancy (lambda) is rows-per-retained-concept. Because the budget is fixed and
    identical across both levels, narrow support means deeper per-concept dose and
    wide support means shallower dose — which is the intended contrast. The old code
    drew *all* rows of the retained concepts, so the two levels differed 3-4x in size
    and the contrast was confounded with training budget.
    """
    concepts = sorted({str(r["concept_id"]) for r in pool})
    keep_n = max(1, round(len(concepts) * fraction))
    shuffled = list(concepts)
    rng.shuffle(shuffled)
    keep = set(shuffled[:keep_n])
    candidates = [r for r in pool if str(r["concept_id"]) in keep]
    if len(candidates) < budget:
        raise InfeasibleBudget(
            f"{kind}: {keep_n} retained concepts hold {len(candidates)} rows < budget {budget}. "
            f"Lower --budget or raise the concept fraction."
        )
    return rng.sample(candidates, budget)


def _weighted_sample_without_replacement(
    rows: list[dict], weights: list[float], rng: random.Random, budget: int
) -> list[dict]:
    """Efraimidis-Spirakis: key_i = u_i ** (1 / w_i), take the `budget` largest keys.

    Exact weighted sampling without replacement, which is what a fixed budget needs —
    with-replacement draws would let one row occupy several slots and quietly shrink
    the realised training set below the budget.
    """
    keyed = []
    for row, weight in zip(rows, weights):
        if weight <= 0.0:
            continue
        keyed.append((rng.random() ** (1.0 / weight), row))
    if len(keyed) < budget:
        raise InfeasibleBudget(
            f"weighted draw: only {len(keyed)} rows carry positive weight < budget {budget}"
        )
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in keyed[:budget]]


def sample_natnat(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    """Neither factor stratified: language and concept drawn from their MARGINALS.

    This is the (off, off) cell of the 2x2 factorial, and it is not the same policy
    as `random`. `random` is uniform over rows, i.e. it reproduces the pool's natural
    *joint* language-by-concept distribution. NatNat reproduces the two natural
    *marginals* while breaking the dependence between them, by weighting each row by
    P(l) * P(c) / P(l, c). Keeping both in the design is what separates "no
    stratification" from "the joint structure the pool happens to have".
    """
    n = len(pool)
    lang_count = Counter(str(r["language"]) for r in pool)
    concept_count = Counter(str(r["concept_id"]) for r in pool)
    cell_count = Counter((str(r["language"]), str(r["concept_id"])) for r in pool)
    weights = []
    for row in pool:
        lang, concept = str(row["language"]), str(row["concept_id"])
        # P(l) * P(c) / P(l, c), with the shared 1/n factors cancelled.
        weights.append(lang_count[lang] * concept_count[concept] / (cell_count[(lang, concept)] * n))
    return _weighted_sample_without_replacement(pool, weights, rng, budget)


def _stratified_round_robin(
    pool: list[dict], rng: random.Random, budget: int, key: str, kind: str
) -> list[dict]:
    """Uniform over the levels of one factor; the other factor is left natural.

    Round-robin one row per level per pass, so the realised marginal over `key` is as
    flat as the budget and the per-level supply allow, while rows *within* a level are
    drawn uniformly and therefore keep that level's natural composition on every other
    factor. This is the single-factor arm of the 2x2.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        buckets[str(row[key])].append(row)
    order = sorted(buckets)
    rng.shuffle(order)
    for level in order:
        rng.shuffle(buckets[level])
    chosen: list[dict] = []
    depth = 0
    while len(chosen) < budget:
        progressed = False
        for level in order:
            bucket = buckets[level]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                progressed = True
                if len(chosen) == budget:
                    return chosen
        if not progressed:
            raise InfeasibleBudget(f"{kind}: pool exhausted at {len(chosen)} < budget {budget}")
        depth += 1
    return chosen


def sample_langonly(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    return _stratified_round_robin(pool, rng, budget, "language", "langonly")


def sample_conceptonly(pool: list[dict], rng: random.Random, budget: int) -> list[dict]:
    return _stratified_round_robin(pool, rng, budget, "concept_id", "conceptonly")


def sample_langonlycond(pool: list[dict], rng: random.Random, budget: int, seed: int) -> list[dict]:
    """Joint's language marginal, but concepts left natural within each language.

    METHOD_DESIGN.md §1 carries the added-concept claim on `Joint - LangOnlyCond`,
    not on `Joint - LangOnly`. That only isolates concept stratification if the two
    arms agree on the language marginal: otherwise the difference also contains
    whatever the language marginals differ by. So this arm reconstructs Joint's
    realised per-language counts at the SAME seed and then fills each language by
    uniform draw, leaving concept composition natural. The remaining difference
    between the two arms is concept stratification and nothing else.
    """
    joint_rows = sample_joint(pool, random.Random(seed), budget)
    target = Counter(str(r["language"]) for r in joint_rows)
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_language[str(row["language"])].append(row)
    chosen: list[dict] = []
    for language in sorted(target):
        want = target[language]
        available = by_language.get(language, [])
        if want > len(available):
            raise InfeasibleBudget(
                f"langonlycond: language '{language}' needs {want} rows to match Joint's "
                f"marginal at seed {seed}, pool has {len(available)}"
            )
        chosen.extend(rng.sample(available, want))
    if len(chosen) != budget:
        raise InfeasibleBudget(f"langonlycond: assembled {len(chosen)} rows != budget {budget}")
    return chosen


def sample_rows(pool: list[dict], seed: int, budget: int, kind: str) -> list[dict]:
    if kind in UNIMPLEMENTED_KINDS:
        raise NotImplementedError(
            f"arm '{kind}' has no selection rule: {UNIMPLEMENTED_KINDS[kind]}. "
            f"It previously fell through to uniform random, which made it a duplicate "
            f"of the 'random' arm. Implement it or drop it from the manifest."
        )
    rng = random.Random(seed)
    if kind == "random":
        return sample_random(pool, rng, budget)
    if kind == "joint":
        return sample_joint(pool, rng, budget)
    if kind == "englishheavy":
        return sample_englishheavy(pool, rng, budget)
    if kind == "lowocc":
        return sample_occupancy(pool, rng, budget, LOWOCC_CONCEPT_FRACTION, kind)
    if kind == "highocc":
        return sample_occupancy(pool, rng, budget, HIGHOCC_CONCEPT_FRACTION, kind)
    if kind == "natnat":
        return sample_natnat(pool, rng, budget)
    if kind == "langonly":
        return sample_langonly(pool, rng, budget)
    if kind == "conceptonly":
        return sample_conceptonly(pool, rng, budget)
    if kind == "langonlycond":
        return sample_langonlycond(pool, rng, budget, seed)
    raise NotImplementedError(f"unknown arm kind '{kind}'; implemented: {IMPLEMENTED_KINDS}")


def describe(rows: list[dict]) -> dict:
    concepts = Counter(str(r["concept_id"]) for r in rows)
    return {
        "items": len(rows),
        "row_id_signature": row_id_signature(rows),
        "language_marginal": dict(sorted(Counter(str(r["language"]) for r in rows).items())),
        "distinct_concepts": len(concepts),
        "distinct_images": len({str(r["image_path"]) for r in rows}),
        "rows_per_concept_mean": round(len(rows) / max(1, len(concepts)), 3),
        "image_sha256_present": sum(1 for r in rows if r.get("image_sha256")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-jsonl", type=Path, required=True,
                        help="the candidate pool, one row per line")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True,
                        help="fixed number of rows per arm; must be strictly < pool size")
    parser.add_argument("--arm", action="append", required=True, metavar="KIND:NAME",
                        help="repeatable, e.g. --arm random:Random --arm lowocc:LowOcc")
    parser.add_argument("--seed", action="append", type=int, required=True,
                        help="repeatable; every arm is materialised at every seed")
    parser.add_argument("--exclude-ids-jsonl", type=Path, action="append", default=[],
                        help="repeatable; rows whose id appears here are removed from the pool")
    parser.add_argument("--exclude-concepts-jsonl", type=Path, action="append", default=[],
                        help="repeatable; rows whose concept_id appears here are removed "
                             "(use for the eval inventory: 7/8 eval concepts sit in this pool)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report feasibility and per-arm composition without writing jsonl")
    args = parser.parse_args()

    pool = load_pool(args.pool_jsonl)
    pool_before = len(pool)

    excluded_ids: set[str] = set()
    for path in args.exclude_ids_jsonl:
        excluded_ids |= {str(json.loads(l)["id"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    excluded_concepts: set[str] = set()
    for path in args.exclude_concepts_jsonl:
        excluded_concepts |= {str(json.loads(l)["concept_id"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    if excluded_ids or excluded_concepts:
        pool = [r for r in pool
                if str(r["id"]) not in excluded_ids and str(r["concept_id"]) not in excluded_concepts]
    if not pool:
        raise RuntimeError("exclusions removed the entire pool")

    # --- the guard that would have caught the 2026-09-22 defect at generation time ---
    if args.budget >= len(pool):
        raise SystemExit(
            f"refusing: --budget {args.budget} >= pool {len(pool)}. A budget that is not "
            f"binding makes every selection policy return the whole pool, which is exactly "
            f"the defect this script replaces. Choose a budget strictly below {len(pool)}."
        )

    if not args.dry_run:
        enrich(pool)

    records = []
    signatures: dict[int, dict[str, str]] = defaultdict(dict)
    for spec in args.arm:
        kind, _, arm_name = spec.partition(":")
        kind = kind.lower()
        arm_name = arm_name or kind
        for seed in args.seed:
            rows = sample_rows(pool, seed, args.budget, kind)
            if len(rows) != args.budget:
                raise RuntimeError(f"{arm_name}/{seed}: got {len(rows)} rows, expected {args.budget}")
            summary = describe(rows)
            signatures[seed][arm_name] = summary["row_id_signature"]
            destination = args.out_root / arm_name / f"seed_{seed}.jsonl"
            if not args.dry_run:
                write_jsonl(destination, rows)
            records.append({"arm": arm_name, "kind": kind, "seed": seed,
                            "path": str(destination), **summary})

    # --- post-condition: no two arms at the same seed may hold the same row set ---
    collisions = []
    for seed, by_arm in signatures.items():
        seen: dict[str, str] = {}
        for arm_name, sig in sorted(by_arm.items()):
            if sig in seen:
                collisions.append({"seed": seed, "arms": [seen[sig], arm_name], "signature": sig})
            seen[sig] = arm_name
    if collisions:
        raise SystemExit(
            "refusing: arms with identical row sets at the same seed "
            f"(this is the 2026-09-22 defect): {json.dumps(collisions, indent=2)}"
        )

    payload = {
        "version": 2,
        "supersedes": "scripts/materialize_hardopen_subsets.py",
        "operator_hard_open": True,
        "not_key_audit": True,
        "not_split_locked": True,
        "pool_jsonl": str(args.pool_jsonl),
        "pool_items_before_exclusions": pool_before,
        "pool_items": len(pool),
        "excluded_ids": len(excluded_ids),
        "excluded_concepts": len(excluded_concepts),
        "budget": args.budget,
        "budget_is_binding": True,
        "distinct_row_sets_verified": True,
        "subsets": records,
    }
    if not args.dry_run:
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        args.registry.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "subsets"}, ensure_ascii=False))
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
