#!/usr/bin/env python3
"""Freeze the multilingual WorldCuisines ENDPOINT set from the held-out eval split.

This replaces `data/pilot/worldcuisines_sanity.jsonl` as the evaluation set. That
file was never an endpoint set: `prepare_worldcuisines_pilot.py` reads only
`train_task1`/`train_task2`, so the 32 "held-out" items were carved out of the
*training* split — which is why 7 of its 8 concepts sat in the selection pool and
had to be removed by a hand-passed `--exclude-concepts-jsonl`.

WorldCuisines ships a real held-out split. Two properties are verified here rather
than asserted, because each is a silent-vacuity failure mode:

* **Concept disjointness is checked, not assumed.** The script refuses to write if
  any `food_id` in the chosen eval languages also appears in `train_task1` for the
  same languages. On the released v1.1 data the overlap is 0 — the split is already
  concept-disjoint — so the exclusion hack is not replaced by a subtler one, it is
  retired. The refusal is what keeps that true if the source data ever changes.

* **Every language must have a nonempty accepted set.** MATH_SPEC eq4 accepts
  `k_l = floor(target_coverage * n_l)` items per language; the run is refused unless
  `k_l >= --min-accepted` for every language actually written.

Image freezing is inherited from `prepare_worldcuisines_pilot.py` unchanged in
substance: each copy is written at the geometry the deployed Qwen2.5-VL
`smart_resize` selects, so the pinned bytes are the bytes the vision tower reads.
Because the eval languages share one concept set, each distinct image is frozen
once and its receipt reused across languages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from prepare_worldcuisines_pilot import file_hash, freeze_image


def read_split(root: Path, split: str, language: str) -> list[dict]:
    path = root / "hf_prompt" / split / f"{language}_{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing split file: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_concepts(root: Path, languages: list[str]) -> set[str]:
    concepts: set[str] = set()
    for language in languages:
        path = root / "hf_prompt" / "train_task1" / f"{language}_train_task1.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing train split for disjointness check: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    concepts.add(str(json.loads(line)["food_id"]))
    return concepts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path, required=True,
                        help="directory holding hf_prompt/<split>/<lang>_<split>.jsonl")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="directory the rows' image_path values resolve against")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-out", type=Path,
                        help="where frozen resized copies are written (default: <out parent>/images)")
    parser.add_argument("--runtime-image-root", type=Path,
                        help="path the launch will pass as --image-root; recorded, not read here")
    parser.add_argument("--split", default="small_eval_task1")
    parser.add_argument("--languages", required=True,
                        help="comma-separated; the endpoint's language set L")
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--min-accepted", type=int, default=2)
    parser.add_argument("--min-languages", type=int, default=8,
                        help="refuse a set too small for the trimmed language mean T(|L|)=ceil(|L|/4)")
    parser.add_argument("--resize-factor", type=int, default=28)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if len(languages) < args.min_languages:
        raise SystemExit(
            f"refusing: {len(languages)} languages < --min-languages {args.min_languages}. "
            f"The endpoint is a trimmed mean over languages with T(|L|)=ceil(|L|/4); below 8 "
            f"languages the trim removes a large fraction of the sample."
        )

    by_language = {language: read_split(args.world_root, args.split, language) for language in languages}

    # --- refusal 1: the split must actually be held out at the concept level ---
    eval_concepts = {str(row["food_id"]) for rows in by_language.values() for row in rows}
    overlap = sorted(eval_concepts & train_concepts(args.world_root, languages))
    if overlap:
        raise SystemExit(
            f"refusing: {len(overlap)} concepts appear in BOTH {args.split} and train_task1 "
            f"for these languages (e.g. {overlap[:10]}). The endpoint would not be concept-disjoint "
            f"from the selection pool."
        )

    # --- refusal 2: eq4 must accept something in every language ---
    for language, rows in by_language.items():
        k_l = math.floor(args.target_coverage * len(rows))
        if k_l < args.min_accepted:
            raise SystemExit(
                f"refusing: language '{language}' has {len(rows)} items; at coverage "
                f"{args.target_coverage} eq4 accepts k_l={k_l} < --min-accepted {args.min_accepted}."
            )

    image_out = args.image_out or (args.out.parent / "images")
    runtime_root = args.runtime_image_root or image_out

    frozen_cache: dict[str, tuple[str, dict]] = {}
    rows_out: list[dict] = []
    for language in languages:
        for row in by_language[language]:
            relative = Path(row["image_path"])
            key = str(relative)
            if key not in frozen_cache:
                source = args.image_root / relative
                if not source.is_file():
                    raise SystemExit(f"refusing: eval row references a missing image: {source}")
                geometry = freeze_image(
                    source, image_out / relative,
                    factor=args.resize_factor, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                )
                frozen_cache[key] = (file_hash(image_out / relative), geometry)
            digest, geometry = frozen_cache[key]
            rows_out.append({
                "id": f"worldcuisines:{args.split}:{language}:{row['qa_id']}",
                "language": language,
                "concept_id": str(row["food_id"]),
                "question": row["question"],
                "options": row["options"],
                "answer_index": int(row["mcq_answer_index"]),
                "answer": row["answer"],
                "image_path": str((runtime_root / relative).resolve()),
                "image_sha256": digest,
                "image_geometry": geometry,
                "source": "worldcuisines/vqa-v1.1",
                "split": args.split,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows_out)
    args.out.write_text(body, encoding="utf-8")

    per_language = Counter(row["language"] for row in rows_out)
    receipt = {
        "version": 1,
        "role": "endpoint",
        "supersedes": "data/pilot/worldcuisines_sanity.jsonl",
        "split": args.split,
        "languages": languages,
        "language_count": len(languages),
        "trim_T": math.ceil(len(languages) / 4),
        "items": len(rows_out),
        "items_per_language": dict(sorted(per_language.items())),
        "distinct_concepts": len(eval_concepts),
        "distinct_images": len(frozen_cache),
        "train_concept_overlap": 0,
        "target_coverage": args.target_coverage,
        "accepted_per_language": {
            language: math.floor(args.target_coverage * count) for language, count in sorted(per_language.items())
        },
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "source": "worldcuisines/vqa-v1.1",
        "source_root": str(args.world_root.resolve()),
        "image_transform": (
            f"frozen PNG copies written at the geometry the deployed Qwen2.5-VL smart_resize "
            f"selects (factor {args.resize_factor}, min_pixels {args.min_pixels}, "
            f"max_pixels {args.max_pixels}), so the processor's resize is a fixed point on them"
        ),
        "image_sha256": {key: digest for key, (digest, _) in sorted(frozen_cache.items())},
    }
    args.out.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "image_sha256"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
