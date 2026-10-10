#!/usr/bin/env python3
"""Build the multilingual selection POOL that the fixed-budget policies draw from.

WHY THIS SCRIPT CONSTRUCTS AN IMBALANCE INSTEAD OF SAMPLING THE SPLIT AS-IS
--------------------------------------------------------------------------
`train_task1` is perfectly balanced by construction: every one of the 30 language
variants carries the same 1626 `food_id`s at ~16 rows each (measured, not assumed --
`en`, `yo` and `sc` all report rows 27000 / concepts 1626 / max 22 / p50 16 / min 15).

A pool drawn uniformly from that split has a flat language marginal AND a flat
concept marginal. Every policy in the 2x2 then returns the same distribution:
stratifying a factor that is already uniform is a no-op, so Joint, LangOnly,
ConceptOnly and NatNat would all be Random with a different random seed. The
experiment would run to completion and measure nothing. This is the same inertness
already observed on the old 4-language pool, where LangOnly returned 500/500/500/500
against a Random draw that was also ~500 each.

Selection policies are only distinguishable when supply is uneven, which is the
situation they exist for: real multilingual corpora are steeply skewed by language
and long-tailed by concept, and high-resource languages cover the concept tail that
low-resource languages do not. WorldCuisines deliberately removed that skew.

So the skew is re-imposed here, EXPLICITLY AND ON THE RECORD:

  * it is a *declared construction parameter*, not a measured property of any
    corpus, and this script does not pretend otherwise. The full realised profile
    is written into the receipt so the paper reports the pool's imbalance as a
    stipulated design condition;
  * the language and concept profiles are geometric with ratios passed on the
    command line, and the concept ranking is a seeded permutation, so "head" and
    "tail" concepts carry no semantic content that could be confounded with the
    endpoint;
  * `--tail-floor` couples the two factors: a language at resource rank j may only
    draw from the top `f_j` fraction of the concept ranking, f_j decaying from 1.0
    at the highest-resource language to `--tail-floor` at the lowest. This coupling
    is what makes NatNat (marginals preserved, dependence broken) a different policy
    from Random (joint preserved) rather than a relabelling of it.

Any claim read off these runs is therefore a claim about selection policies under a
supply profile of the stated shape -- not a claim about WorldCuisines, whose own
supply is uniform. That scope limit belongs in the paper and is recorded in the
receipt's `scope_limit` field so it cannot be silently dropped.

Concept disjointness from the endpoint is enforced, not assumed: the run is refused
if any pool concept also appears in the endpoint set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_worldcuisines_pilot import file_hash, freeze_image


def read_train(root: Path, language: str) -> list[dict]:
    path = root / "hf_prompt" / "train_task1" / f"{language}_train_task1.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing train split: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def geometric_profile(n: int, ratio: float) -> list[float]:
    """Weights w_0..w_{n-1} with w_0 / w_{n-1} == ratio, decaying geometrically."""
    if n == 1:
        return [1.0]
    return [ratio ** (-rank / (n - 1)) for rank in range(n)]


def weighted_sample(rows: list[dict], weights: list[float], rng: random.Random, k: int) -> list[dict]:
    """Efraimidis-Spirakis weighted sampling WITHOUT replacement.

    Without replacement is required: a fixed quota filled with repeats would shrink
    the realised pool below the quota while the receipt still claimed the quota.
    """
    keyed = [(rng.random() ** (1.0 / w), row) for row, w in zip(rows, weights) if w > 0.0]
    if len(keyed) < k:
        raise SystemExit(f"quota {k} exceeds {len(keyed)} eligible rows")
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in keyed[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-out", type=Path, required=True)
    parser.add_argument("--runtime-image-root", type=Path)
    parser.add_argument("--endpoint", type=Path, required=True,
                        help="endpoint jsonl; its concepts are refused in the pool")
    parser.add_argument("--languages", required=True,
                        help="comma-separated, ordered HIGHEST resource first -- the order IS the profile")
    parser.add_argument("--total", type=int, default=30000, help="target pool size before rounding")
    parser.add_argument("--lang-ratio", type=float, default=10.0,
                        help="supply of the highest-resource language / the lowest")
    parser.add_argument("--concept-ratio", type=float, default=20.0,
                        help="supply of the head concept / the tail concept")
    parser.add_argument("--tail-floor", type=float, default=0.25,
                        help="fraction of the concept ranking the LOWEST-resource language may draw from")
    parser.add_argument("--min-budget-headroom", type=float, default=4.0,
                        help="pool must be at least this many times the largest planned budget")
    parser.add_argument("--budget", type=int, default=2000, help="planned per-subset budget B")
    parser.add_argument("--max-missing-images", type=float, default=0.01,
                        help="largest fraction of referenced images that may be absent from the archive")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--resize-factor", type=int, default=28)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    rng = random.Random(args.seed)

    if args.total < args.min_budget_headroom * args.budget:
        raise SystemExit(
            f"refusing: pool target {args.total} < {args.min_budget_headroom}x budget {args.budget}. "
            f"With little headroom every policy is forced to take nearly the whole pool and the "
            f"arms converge for an arithmetic reason, not a scientific one."
        )

    endpoint_concepts = set()
    with args.endpoint.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                endpoint_concepts.add(str(json.loads(line)["concept_id"]))

    # --- language supply profile: the CLI order is the declared resource ranking ---
    lang_weights = geometric_profile(len(languages), args.lang_ratio)
    scale = args.total / sum(lang_weights)
    quota = {language: max(1, round(weight * scale)) for language, weight in zip(languages, lang_weights)}

    # --- concept supply profile: seeded ranking, so head/tail carry no semantics ---
    source = read_train(args.world_root, languages[0])
    concepts = sorted({str(row["food_id"]) for row in source})
    leak = sorted(set(concepts) & endpoint_concepts)
    if leak:
        raise SystemExit(
            f"refusing: {len(leak)} concepts appear in BOTH the pool source and the endpoint "
            f"(e.g. {leak[:10]}); the selection pool would not be concept-disjoint from evaluation."
        )
    rng.shuffle(concepts)
    concept_rank = {concept: rank for rank, concept in enumerate(concepts)}
    concept_weight = dict(zip(concepts, geometric_profile(len(concepts), args.concept_ratio)))

    # --- coupling: low-resource languages see only the head of the concept ranking ---
    reach = geometric_profile(len(languages), 1.0 / args.tail_floor)  # 1.0 ... tail_floor
    reach = [1.0 / value for value in reach]
    allowed_depth = {
        language: max(1, round(fraction * len(concepts)))
        for language, fraction in zip(languages, reach)
    }

    # The released archive is missing a handful of images (measured: 3 of 4297 distinct
    # train images). Eligibility is decided BEFORE the quota is drawn, so a missing file
    # can never silently shrink a language below its declared quota -- it only removes a
    # row from the candidate set. A material loss is still refused rather than absorbed.
    present: dict[str, bool] = {}

    def has_image(row: dict) -> bool:
        key = str(row["image_path"])
        if key not in present:
            present[key] = (args.image_root / key).is_file()
        return present[key]

    dropped = 0
    rows_out: list[dict] = []
    for language in languages:
        depth = allowed_depth[language]
        candidates, weights = [], []
        for row in read_train(args.world_root, language):
            concept = str(row["food_id"])
            if concept_rank[concept] < depth:
                if not has_image(row):
                    dropped += 1
                    continue
                candidates.append(row)
                weights.append(concept_weight[concept])
        drawn = weighted_sample(candidates, weights, rng, quota[language])
        for row in drawn:
            rows_out.append({"__lang": language, "__row": row})

    missing_images = sorted(key for key, ok in present.items() if not ok)
    missing_fraction = len(missing_images) / max(1, len(present))
    if missing_fraction > args.max_missing_images:
        raise SystemExit(
            f"refusing: {len(missing_images)} of {len(present)} referenced images are absent "
            f"({missing_fraction:.3%}), above --max-missing-images {args.max_missing_images:.3%}; "
            f"the archive is incomplete, not merely imperfect."
        )

    # --- freeze each distinct image once, at the deployed processor's geometry ---
    image_out = args.image_out
    runtime_root = args.runtime_image_root or image_out
    frozen: dict[str, str] = {}
    final: list[dict] = []
    for entry in rows_out:
        language, row = entry["__lang"], entry["__row"]
        relative = Path(row["image_path"])
        key = str(relative)
        if key not in frozen:
            src = args.image_root / relative
            if not src.is_file():
                raise SystemExit(f"refusing: pool row references a missing image: {src}")
            freeze_image(src, image_out / relative, factor=args.resize_factor,
                         min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            frozen[key] = file_hash(image_out / relative)
        final.append({
            "id": f"worldcuisines:train_task1:{language}:{row['qa_id']}",
            "language": language,
            "concept_id": str(row["food_id"]),
            "question": row["question"],
            "options": row["options"],
            "answer_index": int(row["mcq_answer_index"]),
            "answer": row["answer"],
            "image_path": str((runtime_root / relative).resolve()),
            "image_sha256": frozen[key],
            "source": "worldcuisines/vqa-v1.1",
            "hard_open": True,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in final)
    args.out.write_text(body, encoding="utf-8")

    per_language = Counter(row["language"] for row in final)
    per_concept = Counter(row["concept_id"] for row in final)
    by_language_concepts = defaultdict(set)
    for row in final:
        by_language_concepts[row["language"]].add(row["concept_id"])
    tail = sorted(per_concept.values(), reverse=True)
    receipt = {
        "version": 1,
        "role": "selection_pool",
        "supersedes": "data/frozen/subsets/Random/seed_646616.jsonl (4 languages, 9940 rows)",
        "seed": args.seed,
        "languages_by_declared_resource_rank": languages,
        "rows": len(final),
        "rows_per_language": dict(sorted(per_language.items())),
        "concepts": len(per_concept),
        "concepts_per_language": {k: len(v) for k, v in sorted(by_language_concepts.items())},
        "concept_rows_max": tail[0],
        "concept_rows_median": tail[len(tail) // 2],
        "concept_rows_min": tail[-1],
        "distinct_images": len(frozen),
        "candidate_rows_dropped_missing_image": dropped,
        "referenced_images_absent_from_archive": len(missing_images),
        "referenced_images_total": len(present),
        "endpoint": str(args.endpoint),
        "endpoint_concept_overlap": 0,
        "planned_budget": args.budget,
        "budget_headroom": round(len(final) / args.budget, 2),
        "supply_profile": {
            "status": "STIPULATED, NOT MEASURED",
            "language_ratio_high_over_low": args.lang_ratio,
            "concept_ratio_head_over_tail": args.concept_ratio,
            "tail_floor_lowest_resource_language": args.tail_floor,
            "concept_ranking": "seeded permutation; head/tail carry no semantic content",
        },
        "scope_limit": (
            "WorldCuisines' own train split is uniform across languages and concepts "
            "(1626 concepts x ~16 rows in every language). The imbalance this pool exhibits "
            "was imposed by this script with the parameters above. Results therefore speak to "
            "selection policies under a supply profile of this declared shape, NOT to "
            "WorldCuisines as distributed, and not to any measured corpus."
        ),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "source": "worldcuisines/vqa-v1.1 train_task1",
        "source_root": str(args.world_root.resolve()),
    }
    args.out.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
