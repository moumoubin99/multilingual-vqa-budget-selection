#!/usr/bin/env python3
"""Ground-truth VQA evaluation with complementary option permutations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ccmvqa.formulas import selective_risk
from ccmvqa.authorization import authorize_model_command, sha256_file
from ccmvqa.runtime_env import activate_vendored_torch, torch_build_receipt

# Must happen before the first torch import: the Qwen2.5-VL processor cannot be built
# without torchvision, which is only available in the vendored build.
TORCH_SELECTION = activate_vendored_torch()

# What a --pilot-mode launch is allowed to compute and claim. The pilot is a receipt-bound
# sanity and timing run, so it evaluates MATH_SPEC eq4 per language and nothing else. The
# study's headline aggregate — the interquartile mean of eq4 over the pre-registered
# low-resource languages — is deliberately not computed here: the registered trim is
# inactive below eight languages, so a four-language pilot could only emit a plain mean
# wearing the headline's name. AURC and the multiclass Brier score are confirmatory
# secondary endpoints with no pilot-stage estimand, so they are withheld as well.
PILOT_SCOPE = {
    "computes": [
        "eq4 selective risk per language at the target coverage",
        "eq4 acceptance counts and report-only per-language thresholds",
        "unweighted accuracy as an operational sanity statistic",
        "shortcut-vs-greedy conformance on the sampled conformance ids",
        "wall-clock timing",
    ],
    "withheld": {
        "headline_lowres_iqm": (
            "needs the pre-registered low-resource language set; the registered interquartile"
            " trim is inactive below eight languages, so this pilot cannot produce it"
        ),
        "aurc_secondary": "confirmatory secondary endpoint, not a pilot estimand",
        "brier_secondary": "confirmatory secondary endpoint, not a pilot estimand",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--conformance-fraction", type=float, default=0.02)
    parser.add_argument("--pilot-mode", action="store_true")
    parser.add_argument("--force-full-generation", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--image-root",
        help="pilot only: directory holding the receipt-hashed images when the frozen store was relocated",
    )
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in values:
        missing = {"id", "language", "question", "options", "answer_index", "image_path"} - row.keys()
        if missing:
            raise ValueError(f"evaluation item lacks fields {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"duplicate evaluation id {row['id']}")
        seen.add(row["id"])
        if not 0 <= int(row["answer_index"]) < len(row["options"]):
            raise ValueError(f"answer index out of range for {row['id']}")
    return values


def open_image(path: str, max_pixels: int):
    import math
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


def option_token_ids(tokenizer, n_options: int) -> list[int]:
    result = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]:
        alternatives = [tokenizer.encode(letter, add_special_tokens=False), tokenizer.encode(" " + letter, add_special_tokens=False)]
        one = next((candidate for candidate in alternatives if len(candidate) == 1), None)
        if one is None:
            raise RuntimeError(f"option letter {letter} is not a single tokenizer token")
        result.append(one[0])
    return result


def prompt_for(row: dict, permutation: list[int]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"{letters[position]}. {row['options'][semantic]}" for position, semantic in enumerate(permutation)
    )
    return f"{row['question']}\n{options}\nAnswer with one option letter only."


def semantic_probabilities(
    model, processor, row: dict, permutation: list[int], max_pixels: int, device: str, image_path: str
):
    import torch

    prompt = prompt_for(row, permutation)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = open_image(image_path, max_pixels)
    batch = processor(text=[text], images=[image], padding=False, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    # bfloat16 autocast is a CUDA-only fast path; CPU runs the same math in float32.
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits = model(**batch, use_cache=False).logits[0, -1].float()
    ids = option_token_ids(processor.tokenizer, len(permutation))
    displayed = torch.softmax(logits[ids], dim=-1).cpu().numpy()
    semantic = np.zeros(len(permutation), dtype=np.float64)
    for displayed_position, semantic_position in enumerate(permutation):
        semantic[semantic_position] = displayed[displayed_position]
    return semantic, batch, ids


def pin_greedy_generation_config(model) -> dict:
    """Record the checkpoint defaults, then pin and assert the reviewed decode mode.

    The explicit ``generate`` arguments below are the operative controls.  Checking
    the loaded configuration as well prevents a checkpoint-side beam/sampling default
    from silently changing the conformance path if those arguments are ever removed.
    """
    config = model.generation_config
    loaded_num_beams = getattr(config, "num_beams", 1)
    loaded_do_sample = getattr(config, "do_sample", False)
    if loaded_num_beams is not None and (isinstance(loaded_num_beams, bool) or loaded_num_beams < 1):
        raise RuntimeError(f"checkpoint generation config has invalid num_beams={loaded_num_beams}")
    if loaded_do_sample not in (None, False, True):
        raise RuntimeError(f"checkpoint generation config has invalid do_sample={loaded_do_sample!r}")
    config.num_beams = 1
    config.do_sample = False
    if config.num_beams != 1 or config.do_sample is not False:
        raise RuntimeError("could not pin greedy generation configuration")
    return {
        "loaded_num_beams": loaded_num_beams,
        "loaded_do_sample": loaded_do_sample,
        "effective_num_beams": config.num_beams,
        "effective_do_sample": config.do_sample,
    }


def generated_semantic(model, processor, batch: dict, permutation: list[int]) -> int:
    """Run reviewed greedy decode and verify every emitted token against its score."""
    import torch

    greedy_config = pin_greedy_generation_config(model)
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=16,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequences = generated.sequences
    continuation = sequences[0, batch["input_ids"].shape[1] :]
    processed_scores = tuple(generated.scores)
    if greedy_config["effective_num_beams"] != 1 or greedy_config["effective_do_sample"] is not False:
        raise RuntimeError("generation configuration changed after it was pinned")
    if not 0 < continuation.numel() <= 16 or len(processed_scores) != continuation.numel():
        raise RuntimeError(
            "greedy generation did not stop within max_new_tokens or omitted per-step processed scores"
        )
    emitted = continuation.tolist()
    production_argmax = [int(step[0].argmax().item()) for step in processed_scores]
    if emitted != production_argmax:
        raise RuntimeError(
            f"do_sample=false emitted {emitted}, but processed-logit argmax tokens were {production_argmax}"
        )
    parsed = processor.tokenizer.decode(continuation, skip_special_tokens=True).strip()
    return parse_generated_semantic(parsed, permutation)


def parse_generated_semantic(decoded: str, permutation: list[int]) -> int:
    """Parse the first standalone *in-range* option letter and invert the permutation.

    Generation is allowed to return a short sentence (for example, ``"Answer: C"``).
    Letters embedded in words are not answers, and an earlier standalone letter outside
    the displayed option range must not hide a later valid answer.
    """
    parsed = str(decoded).strip()
    for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", parsed.upper()):
        displayed = ord(match.group(1)) - ord("A")
        if 0 <= displayed < len(permutation):
            return int(permutation[displayed])
    raise RuntimeError(f"could not parse option letter from generated answer {parsed!r}")


def complementary_permutations(n_options: int, seed: int, item_id: str) -> tuple[list[int], list[int]]:
    """Registered seed-derived display order and its complementary reversal."""
    if n_options <= 0 or seed < 0 or not str(item_id):
        raise ValueError("n_options and item_id must be nonempty and seed must be nonnegative")
    first = list(range(n_options))
    random.Random(f"{seed}:{item_id}").shuffle(first)
    return first, list(reversed(first))


def pilot_conformance_ids(data: list[dict], seed: int, fraction: float) -> set[str]:
    """Select ceil(fraction*N), at least one, by SHA-256(seed:item-id) rank."""
    if seed < 0 or not 0.0 <= fraction <= 1.0:
        raise ValueError("seed must be nonnegative and conformance fraction must be in [0, 1]")
    if not data:
        return set()
    count = max(1, int(np.ceil(fraction * len(data))))
    ranked = sorted(data, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    return {row["id"] for row in ranked[:count]}


def language_matrices(predictions: list[dict]):
    """Lay per-item records out on the global frozen evaluation-item axis eq4 consumes.

    Columns are the frozen evaluation items themselves, in ascending `frozen_index` —
    the integer position each item holds in the frozen evaluation file, which is the
    frozen item id eq4 breaks confidence ties on. Every language shares that one axis,
    so `item_ids[column]` is the actual id of the item in that column for every language,
    never a within-language rank and never a lexicographic ordering of string ids
    (where "10" would sort before "2"). Raggedness lives entirely in `mask`: a language
    occupies only the columns of the items it was evaluated on.
    """
    by_language: dict[str, list[dict]] = defaultdict(list)
    owner_of_frozen_index: dict[int, str] = {}
    for record in predictions:
        frozen_index = int(record["frozen_index"])
        owner = owner_of_frozen_index.setdefault(frozen_index, record["id"])
        if owner != record["id"]:
            raise ValueError(f"frozen index {frozen_index} is claimed by both {owner} and {record['id']}")
        by_language[record["language"]].append(record)
    language_names = sorted(by_language)
    item_ids = np.asarray(sorted(owner_of_frozen_index), dtype=np.int64)
    column_of_frozen_index = {frozen_index: column for column, frozen_index in enumerate(item_ids.tolist())}
    correctness = np.zeros((len(language_names), item_ids.size), dtype=np.float64)
    confidence = np.zeros_like(correctness)
    mask = np.zeros_like(correctness, dtype=bool)
    for language_index, language in enumerate(language_names):
        for record in by_language[language]:
            column = column_of_frozen_index[int(record["frozen_index"])]
            correctness[language_index, column] = record["correct"]
            confidence[language_index, column] = record["confidence"]
            mask[language_index, column] = True
    return language_names, correctness, confidence, mask, item_ids


def pair_averaged_probability(p_first, p_second) -> np.ndarray:
    """Average the two complementary-permutation distributions into eq4's confidence input."""
    first = np.asarray(p_first, dtype=np.float64)
    second = np.asarray(p_second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("complementary-permutation probabilities must be equal-length vectors")
    return (first + second) / 2.0


def prediction_record(row: dict, frozen_index: int, probability: np.ndarray, permutations: list[list[int]]) -> dict:
    """Extract eq4's per-item inputs: the argmax option, its correctness and its confidence."""
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (len(row["options"]),):
        raise ValueError(f"probability vector does not cover the options of {row['id']}")
    prediction = int(np.argmax(probability))
    gold = int(row["answer_index"])
    return {
        "id": row["id"],
        "frozen_index": int(frozen_index),
        "language": row["language"],
        "prediction_index": prediction,
        "gold_index": gold,
        "correct": int(prediction == gold),
        "confidence": float(probability[prediction]),
        "semantic_probability": probability.tolist(),
        "permutations": [list(permutation) for permutation in permutations],
    }


def unweighted_accuracy(predictions: list[dict]) -> float:
    """Operational sanity statistic: unweighted mean correctness over the evaluated items."""
    if not predictions:
        raise ValueError("no predictions to score")
    return float(np.mean([record["correct"] for record in predictions]))


def confirmatory_secondary_metrics(predictions: list[dict]) -> dict:
    """AURC and the multiclass Brier score: confirmatory-only, never computed in pilot mode.

    Both are study-level secondary endpoints with no pilot-stage estimand, so a
    ``--pilot-mode`` launch never reaches this function (see ``PILOT_SCOPE``).
    """
    correctness_flat = np.asarray([record["correct"] for record in predictions], dtype=np.float64)
    confidences_flat = np.asarray([record["confidence"] for record in predictions], dtype=np.float64)
    order = np.argsort(-confidences_flat, kind="stable")
    cumulative_error = np.cumsum(1.0 - correctness_flat[order]) / np.arange(1, len(order) + 1)
    brier = np.mean(
        [
            np.sum(
                (
                    np.asarray(record["semantic_probability"])
                    - np.eye(len(record["semantic_probability"]))[record["gold_index"]]
                )
                ** 2
            )
            for record in predictions
        ]
    )
    return {"aurc_secondary": float(np.mean(cumulative_error)), "brier_secondary": float(brier)}


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    data = rows(Path(args.eval_jsonl))
    if not data:
        raise ValueError("evaluation set is empty")
    authorization = authorize_model_command(
        PROJECT_ROOT,
        Path(args.eval_jsonl),
        purpose="eval",
        pilot_mode=bool(args.pilot_mode),
        image_root=Path(args.image_root) if args.image_root else None,
    )
    # Read every image from the path authorization hashed against the receipt.
    image_paths = authorization.get("image_paths") or {row["id"]: row["image_path"] for row in data}
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(args.device)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    greedy_generation_config = pin_greedy_generation_config(model)

    started = time.time()
    predictions = []
    conformance_failures = []
    if args.pilot_mode:
        conformance_ids = pilot_conformance_ids(data, args.seed, args.conformance_fraction)
        conformance_source = "derived_pilot_receipt"
    elif authorization.get("mode") == "operator_hard_open":
        # A hard-open run has no registered ENDPOINT_TABLE.json, and cannot have one: the
        # freeze that would produce it is exactly what `.aris/OPERATOR_HARD_OPEN` authorizes
        # running without. Rather than plant a hand-made file in `refine-logs/freeze/` — which
        # a later confirmatory run would silently consume as if it were registered — the
        # conformance sample is derived here by the same deterministic sha256(seed:item-id)
        # rank rule, and the summary records that it was derived, not registered. The
        # relaxation is bound to the hard-open stamp, so the confirmatory path below is
        # unchanged and still requires the frozen table.
        conformance_ids = pilot_conformance_ids(data, args.seed, args.conformance_fraction)
        conformance_source = "derived_operator_hard_open"
    else:
        endpoint = json.loads((PROJECT_ROOT / "refine-logs" / "freeze" / "ENDPOINT_TABLE.json").read_text(encoding="utf-8"))
        conformance_ids = set(endpoint["conformance_ids"])
        conformance_source = "registered_endpoint_table"
    # `frozen_index` is the item's position in the frozen evaluation file: the global,
    # integer evaluation-item axis eq4 orders confidence ties on.
    for frozen_index, row in enumerate(data):
        n_options = len(row["options"])
        first, second = complementary_permutations(n_options, args.seed, row["id"])
        image_path = image_paths[row["id"]]
        p_first, first_batch, _ = semantic_probabilities(
            model, processor, row, first, args.max_pixels, args.device, image_path
        )
        p_second, second_batch, _ = semantic_probabilities(
            model, processor, row, second, args.max_pixels, args.device, image_path
        )
        probability = pair_averaged_probability(p_first, p_second)
        if args.force_full_generation:
            generated_pair = [
                generated_semantic(model, processor, first_batch, first),
                generated_semantic(model, processor, second_batch, second),
            ]
            probability = np.bincount(generated_pair, minlength=n_options).astype(np.float64) / 2.0
        record = prediction_record(row, frozen_index, probability, [first, second])
        predictions.append(record)
        if row["id"] in conformance_ids:
            generated_predictions = [
                generated_semantic(model, processor, first_batch, first),
                generated_semantic(model, processor, second_batch, second),
            ]
            shortcut_predictions = [int(np.argmax(p_first)), int(np.argmax(p_second))]
            if generated_predictions != shortcut_predictions:
                conformance_failures.append(
                    {"id": row["id"], "generated_semantic": generated_predictions, "shortcut_semantic": shortcut_predictions}
                )

    language_names, correctness, confidence, mask, item_ids = language_matrices(predictions)
    risks, accepted, thresholds = selective_risk(
        correctness, confidence, mask, item_ids, args.target_coverage
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_tmp = output / "predictions.jsonl.tmp"
    predictions_tmp.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in predictions), encoding="utf-8"
    )
    predictions_tmp.replace(output / "predictions.jsonl")
    elapsed = time.time() - started
    # Confirmatory secondaries are study-level endpoints; a pilot has no estimand for them
    # and therefore never computes them (see PILOT_SCOPE).
    secondary = {"aurc_secondary": None, "brier_secondary": None}
    if not args.pilot_mode:
        secondary = confirmatory_secondary_metrics(predictions)
    summary = {
        "status": "completed" if not conformance_failures or args.force_full_generation else "shortcut_withdrawn_rerun_full_generation",
        "pilot_only": bool(args.pilot_mode),
        "n_items": len(predictions),
        "accuracy": unweighted_accuracy(predictions),
        "aurc_secondary": secondary["aurc_secondary"],
        "brier_secondary": secondary["brier_secondary"],
        # The study headline (interquartile mean of eq4 over the pre-registered low-resource
        # languages) is not a pilot output: the registered trim is inactive below eight
        # languages, so emitting it here would be a plain mean under the headline's name.
        "headline_lowres_iqm": None,
        "scope": PILOT_SCOPE if args.pilot_mode else {"computes": ["full selected run specification"], "withheld": {}},
        "evaluation_item_ids": item_ids.tolist(),
        "target_coverage": args.target_coverage,
        "per_language_selective_risk": dict(zip(language_names, risks.tolist())),
        "per_language_threshold_report_only": dict(zip(language_names, thresholds.tolist())),
        "accepted_counts": dict(zip(language_names, accepted.sum(axis=1).astype(int).tolist())),
        "conformance_failures": conformance_failures,
        "conformance_ids": sorted(conformance_ids),
        "conformance_ids_source": conformance_source,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if args.device == "cuda" else 0.0,
        "authorization": authorization,
        "input_sha256": sha256_file(Path(args.eval_jsonl)),
        "full_generation": bool(args.force_full_generation),
        "greedy_generation_config": greedy_generation_config,
        "torch_version": torch.__version__,
        "torch_selection": TORCH_SELECTION,
        "torch_build": torch_build_receipt(),
        "device": args.device,
        # A CPU run reports no accelerator and its wall clock does not transfer to GPU-hour budgeting.
        "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "execution_order_clause": {
            "frozen_vit_feature_cache": "withdrawn_unimplemented",
            "image_prefix_kv_cache": "withdrawn_unimplemented",
            "cost_paid_by_full_recompute": True,
        },
    }
    summary_tmp = output / "eval_summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(output / "eval_summary.json")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not conformance_failures or args.force_full_generation else 2


if __name__ == "__main__":
    raise SystemExit(main())
