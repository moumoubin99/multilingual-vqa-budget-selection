#!/usr/bin/env python3
"""Build the CVQA zero-shot transfer endpoint for the v3 generalization check.

Why this file exists
--------------------
Every v3 selection policy is defined over *WorldCuisines* concepts and over a supply
profile this project stipulated (`prepare_worldcuisines_pool.py`), and the v3 endpoint is
drawn from the same corpus family. A policy ranking that holds only there is a fact about
WorldCuisines, not about selection. CVQA (Romero et al., `afaji/cvqa`, 10 374 items over 39
country-language subsets) is an independently constructed cultural VQA benchmark -- other
authors, other images, other concept inventory, natively authored questions -- so a ranking
that survives it is a ranking about selection policy. CVQA is additionally the benchmark
`METHOD_DESIGN.md` §3 names as held fixed, which the executed block departed from.

The adapters are NOT retrained. This file only produces an evaluation set; the transfer
manifest runs the existing `scripts/evaluate_vqa.py` against the already-trained v3
adapters with `--force-full-generation`, exactly as the main block does.

Ground truth
------------
CVQA's `Label` column carries the real answer index. Verified on shard 0 before this script
was written: 1 038 rows, Counter({3: 286, 2: 275, 0: 249, 1: 228}) -- a real four-way answer
distribution, not a masked placeholder. The README's "submit to eval.ai" line refers to the
leaderboard, not to withheld labels. No model output is ever used as ground truth here.

Declared construction choices (all of them limitations, all reported)
---------------------------------------------------------------------
1. Only the languages shared with the v3 endpoint are kept. `fr`, `sc` (Sardinian) and `yo`
   (Yoruba) have no CVQA subset and are dropped -- the transfer check therefore covers 8 of
   the 12 evaluated languages, and the two lowest-resource ones (sc, yo) are among the
   missing. Transfer evidence is consequently weakest exactly where the study's claim is
   most interesting; this is stated rather than worked around.
2. `ar` is mapped to CVQA's ('Egyptian_Arabic', 'Egypt'). The v3 endpoint's `ar` is MSA, so
   this stratum carries a dialect shift on top of the corpus shift.
3. `es` pools CVQA's seven Spanish country subsets; `zh_cn` takes ('Chinese', 'China') only,
   excluding ('Chinese', 'Singapore').
4. CVQA has no native English subset. The `en` stratum is built from the *English
   translations* (`Translated Question` / `Translated Options`) of rows drawn from the
   matched subsets, round-robin across source languages so no single language dominates,
   and -- critically -- from rows DISJOINT from those used in any native stratum, so no
   image and no question appears twice in the file. It is labelled `en_translated`, never
   `en`: it is translationese, and the EnglishHeavy arm's advantage on it should be read
   with that in mind.
5. Strata are capped to a common size so per-language accuracy is comparable across
   languages and the file stays cheap to decode.
6. CVQA items have four options; the v3 endpoint has five. `evaluate_vqa.py` is
   option-count generic (`n_options = len(row["options"])`), but chance accuracy is 0.25
   here against 0.20 on the v3 endpoint, so absolute accuracies are NOT comparable across
   the two endpoints. Only between-arm differences within this file are readable.

Output
------
  data/transfer/cvqa_transfer_v1.jsonl   rows in the v3 endpoint schema
  data/transfer/cvqa_images/<id>.jpg     one decoded image per row
  data/transfer/cvqa_transfer_receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

# v3 endpoint language -> the CVQA subsets that stand in for it.
LANGUAGE_MAP: dict[str, tuple[str, ...]] = {
    "ar": ("('Egyptian_Arabic', 'Egypt')",),
    "es": (
        "('Spanish', 'Argentina')", "('Spanish', 'Chile')", "('Spanish', 'Colombia')",
        "('Spanish', 'Ecuador')", "('Spanish', 'Mexico')", "('Spanish', 'Spain')",
        "('Spanish', 'Uruguay')",
    ),
    "hi": ("('Hindi', 'India')",),
    "id_formal": ("('Indonesian', 'Indonesia')",),
    "jv_krama": ("('Javanese', 'Indonesia')",),
    "ko_formal": ("('Korean', 'South Korea')",),
    "su_loma": ("('Sundanese', 'Indonesia')",),
    "zh_cn": ("('Chinese', 'China')",),
}
# v3 endpoint languages with no CVQA counterpart, recorded so the gap is visible.
LANGUAGES_UNAVAILABLE = ["fr", "sc", "yo"]

ENGLISH_STRATUM = "en_translated"
PROMPT_COLUMNS = ("Question", "Options")
ENGLISH_COLUMNS = ("Translated Question", "Translated Options")


def load_rows(shards: list[Path]) -> list[dict]:
    """Read every shard once, keeping only the columns this endpoint needs."""
    import pyarrow.parquet as pq

    columns = [
        "image", "ID", "Subset", "Question", "Translated Question",
        "Options", "Translated Options", "Label", "Category",
    ]
    wanted = {subset for subsets in LANGUAGE_MAP.values() for subset in subsets}
    out: list[dict] = []
    for shard in shards:
        handle = pq.ParquetFile(shard)
        for batch in handle.iter_batches(batch_size=256, columns=columns):
            table = batch.to_pylist()
            for row in table:
                if row["Subset"] in wanted:
                    out.append(row)
    return out


def stratify(rows: list[dict], per_language: int, seed: int):
    """Split the matched rows into disjoint native strata plus an English stratum.

    Native strata are drawn first; the English stratum is drawn round-robin from what is
    left, so the two never share a row and no source language dominates the English set.
    """
    subset_to_language = {
        subset: language for language, subsets in LANGUAGE_MAP.items() for subset in subsets
    }
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_language[subset_to_language[row["Subset"]]].append(row)

    rng = random.Random(seed)
    native: dict[str, list[dict]] = {}
    leftover: dict[str, list[dict]] = {}
    for language in sorted(by_language):
        pool = sorted(by_language[language], key=lambda r: str(r["ID"]))
        rng.shuffle(pool)
        take = min(per_language, len(pool))
        native[language] = pool[:take]
        leftover[language] = pool[take:]

    english: list[dict] = []
    cursors = {language: 0 for language in leftover}
    while len(english) < per_language:
        progressed = False
        for language in sorted(leftover):
            if len(english) >= per_language:
                break
            index = cursors[language]
            if index < len(leftover[language]):
                english.append(leftover[language][index])
                cursors[language] = index + 1
                progressed = True
        if not progressed:
            break
    return native, english


def emit(row: dict, language: str, index: int, images_dir: Path, translated: bool) -> dict:
    """One output record in the v3 endpoint schema."""
    from PIL import Image

    question_column, options_column = ENGLISH_COLUMNS if translated else PROMPT_COLUMNS
    question = row[question_column]
    options = list(row[options_column])
    gold = int(row["Label"])
    if not 0 <= gold < len(options):
        raise ValueError(f"CVQA row {row['ID']} has label {gold} outside its {len(options)} options")

    payload = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
    digest = hashlib.sha256(payload).hexdigest()
    item_id = f"cvqa:test:{language}:{index}"
    image_path = images_dir / f"{item_id.replace(':', '_')}.jpg"
    if not image_path.exists():
        image = Image.open(io.BytesIO(payload))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=95)
    return {
        "id": item_id,
        "language": language,
        "question": question,
        "options": options,
        "answer": options[gold],
        "answer_index": gold,
        "image_path": str(image_path.resolve()),
        "image_sha256": digest,
        "concept_id": str(row.get("Category") or "unknown"),
        "source": {
            "dataset": "afaji/cvqa",
            "split": "test",
            "subset": row["Subset"],
            "cvqa_id": str(row["ID"]),
            "prompt_language": "english_translation" if translated else "native",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=646616)
    args = parser.parse_args()

    shards = sorted(args.shards_dir.glob("test-*-of-*.parquet"))
    if len(shards) != 10:
        raise SystemExit(f"refusing: expected 10 CVQA shards, found {len(shards)}")

    rows = load_rows(shards)
    native, english = stratify(rows, args.per_language, args.seed)

    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for language in sorted(native):
        for index, row in enumerate(native[language]):
            records.append(emit(row, language, index, args.images_dir, translated=False))
    for index, row in enumerate(english):
        records.append(emit(row, ENGLISH_STRATUM, index, args.images_dir, translated=True))

    ids = {record["id"] for record in records}
    if len(ids) != len(records):
        raise SystemExit("refusing: duplicate item ids")
    cvqa_ids = [record["source"]["cvqa_id"] for record in records]
    if len(set(cvqa_ids)) != len(cvqa_ids):
        raise SystemExit("refusing: a CVQA row was used in more than one stratum")

    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = defaultdict(int)
    for record in records:
        counts[record["language"]] += 1
    receipt = {
        "built": "scripts/prepare_cvqa_transfer.py",
        "source": {
            "dataset": "afaji/cvqa",
            "split": "test",
            "mirror": "https://hf-mirror.com/datasets/afaji/cvqa",
            "shards": [shard.name for shard in shards],
            "shard_sha256": {shard.name: hashlib.sha256(shard.read_bytes()).hexdigest() for shard in shards},
        },
        "seed": args.seed,
        "per_language_target": args.per_language,
        "items": len(records),
        "items_per_language": dict(sorted(counts.items())),
        "languages_mapped": {k: list(v) for k, v in sorted(LANGUAGE_MAP.items())},
        "languages_unavailable": LANGUAGES_UNAVAILABLE,
        "ground_truth": (
            "CVQA's own `Label` column. Verified non-masked on shard 0 before this script was "
            "written: Counter({3: 286, 2: 275, 0: 249, 1: 228}) over 1038 rows. No model output "
            "is used as ground truth anywhere in this file."
        ),
        "english_stratum": (
            "`en_translated` is built from CVQA's English translations of rows disjoint from every "
            "native stratum, round-robin across the 8 source languages. It is translationese, not "
            "native English, and is labelled as such."
        ),
        "scope_limit": (
            "This is a ZERO-SHOT TRANSFER endpoint: the v3 adapters were trained on a "
            "WorldCuisines-derived pool and have never seen CVQA. Absolute accuracy is expected to "
            "fall for every arm and is NOT comparable to the v3 endpoint -- CVQA items have four "
            "options (chance 0.25) against the v3 endpoint's five (chance 0.20), and the task, "
            "images and concept inventory all differ. The only readable quantity is the sign and "
            "size of BETWEEN-ARM differences within this file, paired within seed. Coverage is 8 "
            "of the 12 evaluated languages; fr, sc and yo have no CVQA subset, so the two "
            "lowest-resource languages of the study are absent from the transfer check. `ar` is "
            "Egyptian Arabic here against MSA on the v3 endpoint. Descriptive, gates nothing."
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "jsonl": str(args.out_jsonl),
        "items": len(records),
        "items_per_language": dict(sorted(counts.items())),
        "languages_unavailable": LANGUAGES_UNAVAILABLE,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
