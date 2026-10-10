#!/usr/bin/env python3
"""Single-recipe Qwen2.5-VL LoRA trainer for pilot and frozen arm subsets."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ccmvqa.formulas import (
    accumulated_loss,
    balanced_independence_target,
    build_adamw,
    build_cosine_warmup_scheduler,
    causal_lm_cross_entropy,
    clip_gradient_global_norm,
    lora_scaling,
    optimizer_step_due,
    qwen_forward_logits,
    seeded_epoch_permutation,
)
from ccmvqa.authorization import authorize_model_command, operator_hard_open, sha256_file


def preflight_checkpoint_files(model_path: str | Path) -> dict:
    """Verify that every indexed safetensors shard is complete and readable."""
    from safetensors import safe_open

    root = Path(model_path)
    incomplete = sorted(path.name for path in root.glob("*.incomplete"))
    if incomplete:
        raise RuntimeError(f"partial checkpoint files present: {incomplete}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in index["weight_map"].items():
        expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
    shards = {}
    for shard_name, expected_tensors in sorted(expected_by_shard.items()):
        shard = root / shard_name
        if not shard.is_file() or shard.stat().st_size <= 8:
            raise RuntimeError(f"checkpoint shard missing or truncated: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            observed_tensors = set(handle.keys())
        if observed_tensors != expected_tensors:
            raise RuntimeError(f"checkpoint index/header mismatch for {shard_name}")
        shards[shard_name] = {"bytes": shard.stat().st_size, "sha256": sha256_file(shard)}
    return {"index_sha256": sha256_file(root / "model.safetensors.index.json"), "shards": shards}


def load_base_model(model_path: str | Path, *, device: str):
    """Exact production checkpoint loader shared by the launch and preflight test."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    preflight_checkpoint_files(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--save-checkpoints", type=int, default=0)
    parser.add_argument("--pilot-mode", action="store_true")
    parser.add_argument("--allow-unlocked-freeze", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training subset is empty")
    required = {"id", "language", "concept_id", "question", "options", "answer_index", "image_path"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"training row {row.get('id')} lacks {sorted(missing)}")
    return rows


def validate_balanced_pilot(rows: list[dict]) -> None:
    languages = sorted({str(row["language"]) for row in rows})
    concepts = sorted({str(row["concept_id"]) for row in rows})
    language_weight = np.full(len(languages), 1.0 / len(languages))
    concept_weight = np.full(len(concepts), 1.0 / len(concepts))
    target = balanced_independence_target(len(rows), language_weight, concept_weight)
    observed = np.zeros((len(languages), len(concepts)), dtype=np.float64)
    language_index = {value: index for index, value in enumerate(languages)}
    concept_index = {value: index for index, value in enumerate(concepts)}
    for row in rows:
        observed[language_index[str(row["language"])], concept_index[str(row["concept_id"])]] += 1.0
    if not np.isclose(target.sum(), len(rows)) or not np.allclose(observed, target, atol=1e-12):
        raise RuntimeError("pilot subset is not the observed balanced language-by-concept eq1 table")


def assert_run_permission(args: argparse.Namespace, rows: list[dict]) -> dict:
    pilot_rows = all(bool(row.get("pilot_only")) for row in rows)
    if args.pilot_mode:
        if not pilot_rows:
            raise RuntimeError("pilot mode accepts only rows explicitly marked pilot_only")
        validate_balanced_pilot(rows)
    elif pilot_rows:
        raise RuntimeError("pilot-only rows cannot enter confirmatory training")
    hard_open = operator_hard_open(PROJECT_ROOT) is not None
    if args.allow_unlocked_freeze and not hard_open:
        raise RuntimeError("--allow-unlocked-freeze is reserved for test harnesses and cannot authorize model execution")
    if hard_open and not args.pilot_mode and pilot_rows:
        raise RuntimeError("pilot-only rows cannot enter hard-open training")
    return authorize_model_command(
        PROJECT_ROOT, Path(args.train_jsonl), purpose="train", pilot_mode=bool(args.pilot_mode)
    )


def open_image(path: str, max_pixels: int, *, require_identity: bool = False):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    pixels = image.width * image.height
    if pixels > max_pixels:
        if require_identity:
            raise RuntimeError(f"frozen launch image exceeds max_pixels: {path}")
        scale = math.sqrt(max_pixels / pixels)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


def encode_row(processor, row: dict, max_pixels: int):
    import torch

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"{letters[index]}. {option}" for index, option in enumerate(row["options"]))
    prompt = f"{row['question']}\n{options}\nAnswer with one option letter only."
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    answer = letters[int(row["answer_index"])]
    full = [*user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    image = open_image(row["image_path"], max_pixels, require_identity=True)
    image_processor = processor.image_processor
    alignment = int(image_processor.patch_size) * int(image_processor.merge_size)
    if image.width % alignment or image.height % alignment:
        raise RuntimeError(f"frozen launch image is not factor-{alignment} aligned: {row['id']}")
    prompt_inputs = processor(
        text=[prompt_text], images=[image], padding=False, return_tensors="pt", do_resize=False
    )
    inputs = processor(
        text=[full_text], images=[image], padding=False, return_tensors="pt", do_resize=False
    )
    labels = inputs["input_ids"].clone()
    prompt_tokens = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_tokens] = -100
    inputs["labels"] = labels
    if labels.ne(-100).sum().item() == 0:
        raise RuntimeError(f"assistant target disappeared for row {row['id']}")
    return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def execute_one_update(
    model,
    batches: list[dict],
    optimizer,
    scheduler,
    *,
    autocast_device: str | None,
    autocast_dtype=None,
    max_gradient_norm: float = 1.0,
) -> dict:
    """Execute the exact composed forward/loss/accumulate/backward/clip/update path."""
    from contextlib import nullcontext

    import torch

    if not batches:
        raise ValueError("one update requires at least one micro-batch")
    raw_losses: list[float] = []
    target_tokens = 0
    for batch in batches:
        labels = batch["labels"]
        model_batch = {key: value for key, value in batch.items() if key != "labels"}
        context = (
            torch.autocast(device_type=autocast_device, dtype=autocast_dtype)
            if autocast_device is not None and autocast_dtype is not None
            else nullcontext()
        )
        with context:
            logits = qwen_forward_logits(model, model_batch)
            raw_loss = causal_lm_cross_entropy(logits, labels)
            loss = accumulated_loss(raw_loss, len(batches))
        if not torch.isfinite(raw_loss):
            raise FloatingPointError("non-finite loss in composed update")
        loss.backward()
        raw_losses.append(float(raw_loss.detach().cpu()))
        target_tokens += int(labels.ne(-100).sum().item())
    gradient_norm = clip_gradient_global_norm(model.parameters(), max_gradient_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite gradient norm in composed update")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "raw_losses": raw_losses,
        "target_tokens": target_tokens,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.micro_batch != 1:
        raise ValueError("this deterministic multimodal runner currently requires --micro-batch 1")
    if args.effective_batch < 1 or args.effective_batch % args.micro_batch:
        raise ValueError("effective batch must be a positive multiple of micro batch")
    rows = load_rows(Path(args.train_jsonl))
    authorization = assert_run_permission(args, rows)
    set_seed(args.seed)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this registered model command")
    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = load_base_model(args.model_path, device="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    expected_scale = lora_scaling(args.alpha, args.rank)
    actual_scales = {
        float(scaling["default"])
        for module in model.modules()
        if isinstance((scaling := getattr(module, "scaling", None)), Mapping)
        and "default" in scaling
    }
    if actual_scales != {expected_scale}:
        raise RuntimeError(f"PEFT LoRA scaling drift: expected {expected_scale}, observed {actual_scales}")
    for name, parameter in model.named_parameters():
        if ".visual." in name or name.startswith("visual."):
            parameter.requires_grad_(False)
    model.train()

    accumulation = args.effective_batch // args.micro_batch
    examples_per_epoch = len(rows)
    planned_updates = max(1, math.ceil(examples_per_epoch * args.epochs / accumulation))
    update_limit = min(planned_updates, args.max_steps) if args.max_steps > 0 else planned_updates
    if args.save_checkpoints < 0 or args.save_checkpoints > update_limit:
        raise ValueError("save-checkpoints must be between zero and the number of updates")
    checkpoint_updates = (
        sorted(set(int(value) for value in np.linspace(1, update_limit, args.save_checkpoints)))
        if args.save_checkpoints
        else []
    )
    optimizer = build_adamw((p for p in model.parameters() if p.requires_grad), learning_rate=args.learning_rate)
    warmup = max(0, int(update_limit * args.warmup_ratio))
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=warmup, training_steps=update_limit)
    optimizer.zero_grad(set_to_none=True)

    update = 0
    losses: list[float] = []
    token_count = 0
    epoch = 0
    while update < update_limit:
        order = seeded_epoch_permutation(len(rows), args.seed, epoch)
        for group_start in range(0, len(order), accumulation):
            group_indices = order[group_start : group_start + accumulation]
            batches = [
                {
                    key: value.cuda(non_blocking=True)
                    for key, value in encode_row(processor, rows[index], args.max_pixels).items()
                }
                for index in group_indices
            ]
            result = execute_one_update(
                model,
                batches,
                optimizer,
                scheduler,
                autocast_device="cuda",
                autocast_dtype=torch.bfloat16,
            )
            losses.extend(result["raw_losses"])
            token_count += result["target_tokens"]
            update += 1
            print(json.dumps({"update": update, "loss": losses[-1], "tokens": token_count}), flush=True)
            if update in checkpoint_updates:
                checkpoint = Path(args.output_dir) / "checkpoints" / f"update_{update:05d}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint / "adapter")
            if update >= update_limit:
                break
        epoch += 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    processor.save_pretrained(output / "processor")
    elapsed = time.time() - started
    summary = {
        "status": "completed",
        "pilot_only": bool(args.pilot_mode),
        "seed": args.seed,
        "examples": len(rows),
        "updates": update,
        "micro_steps": len(losses),
        "target_tokens": token_count,
        "mean_loss": float(np.mean(losses)),
        "last_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "max_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_updates": checkpoint_updates,
        "authorization": authorization,
        "input_sha256": sha256_file(Path(args.train_jsonl)),
        "model_path": str(Path(args.model_path).resolve()),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "command": sys.argv,
        "code_sha256": sha256_file(Path(__file__)),
        "recipe": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "effective_batch": args.effective_batch,
            "warmup_ratio": args.warmup_ratio,
            "vision_frozen": True,
            "optimizer": "AdamW",
            "optimizer_betas": list(optimizer.defaults["betas"]),
            "optimizer_eps": optimizer.defaults["eps"],
            "optimizer_weight_decay": optimizer.defaults["weight_decay"],
            "gradient_clip_norm": 1.0,
            "attention": "sdpa",
            "max_pixels": args.max_pixels,
            "gradient_checkpointing": True,
        },
    }
    temporary = output / "train_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "train_summary.json")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            destination = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "oom": "out of memory" in str(error).casefold(),
                "traceback": traceback.format_exc(),
                "command": sys.argv,
            }
            temporary = destination / "failure.json.tmp"
            temporary.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination / "failure.json")
        raise
