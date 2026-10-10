"""Fail-closed authorization checks for pilot and confirmatory model commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"authorization artifact missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"authorization artifact is not an object: {path}")
    return value


def _pilot_image_path(row: dict, image_root: Path | None) -> Path:
    """Locate a pilot image; a relocated store may only change the directory."""
    recorded = Path(row["image_path"])
    return recorded if image_root is None else Path(image_root) / recorded.name


def authorize_pilot(project: Path, data_path: Path, *, image_root: Path | None = None) -> dict:
    """Authorize only a receipt-bound, machine-screened WorldCuisines pilot."""
    data_path = data_path.resolve()
    receipt_path = data_path.with_suffix(".receipt.json")
    receipt = _read(receipt_path)
    actual_hash = sha256_file(data_path)
    if receipt.get("pilot_only") is not True or receipt.get("source") != "worldcuisines/vqa-v1.1":
        raise RuntimeError("pilot receipt must identify the WorldCuisines pilot-only source")
    if receipt.get("sha256") != actual_hash:
        raise RuntimeError("pilot JSONL does not match its frozen receipt")
    if int(receipt.get("examples", 0)) <= 0:
        raise RuntimeError("pilot receipt has no examples")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("id") for row in rows] != receipt.get("item_ids"):
        raise RuntimeError("pilot row ids do not match the receipt")
    images: dict[str, str] = {}
    for row in rows:
        image_path = _pilot_image_path(row, image_root)
        expected_image = receipt.get("image_sha256", {}).get(row["id"])
        if not image_path.is_file() or sha256_file(image_path) != expected_image or row.get("image_sha256") != expected_image:
            raise RuntimeError(f"pilot image missing or stale for {row['id']}")
        images[row["id"]] = str(image_path)

    screen = _read(project / "refine-logs" / "freeze" / "CONTAMINATION_MACHINE.json")
    if screen.get("pilot_jsonl_sha256") != actual_hash:
        raise RuntimeError("pilot contamination screen is absent or stale")
    if screen.get("n_overlap") != 0:
        raise RuntimeError("pilot overlaps the evaluation question inventory")
    if screen.get("machine_scan_complete") is not True:
        raise RuntimeError("pilot machine contamination screen is incomplete")
    return {"mode": "pilot", "data_sha256": actual_hash, "receipt": str(receipt_path), "image_paths": images}


def _key_audit_valid(value: dict) -> bool:
    status = str(value.get("status", value.get("verdict", ""))).casefold()
    return status in {"pass", "passed", "complete", "approved"} and value.get("native_speaker_reviewed") is True


def authorize_confirmatory(project: Path, data_path: Path, *, purpose: str) -> dict:
    """Check every registered freeze/authorization condition before train or eval."""
    prereg = _read(project / "refine-logs" / "PREREG_OF_RECORD.json")
    if prereg.get("ok") is not True:
        raise RuntimeError("confirmatory execution refused: preregistration is not PASS")

    freeze_dir = project / "refine-logs" / "freeze"
    status = _read(freeze_dir / "FREEZE_STATUS.json")
    required_true = ("split_locked", "confirmatory_training_authorized", "machine_artifacts_complete")
    missing = [key for key in required_true if status.get(key) is not True]
    if missing:
        raise RuntimeError(f"confirmatory execution refused: false freeze gates {missing}")
    key_audit_path = freeze_dir / "KEY_AUDIT.json"
    key_audit = _read(key_audit_path)
    if not _key_audit_valid(key_audit):
        raise RuntimeError("confirmatory execution refused: KEY_AUDIT lacks native-speaker PASS")
    premise_path = freeze_dir / "PREMISE_CHECK.json"
    if _read(premise_path).get("premise_pass") is not True:
        raise RuntimeError("confirmatory execution refused: section-F premise did not pass")
    allocation_path = freeze_dir / "ALLOCATION_CERTS.json"
    allocation = _read(allocation_path)
    if not allocation.get("certificates") or not all(value.get("feasible") is True for value in allocation["certificates"]):
        raise RuntimeError("confirmatory execution refused: allocation feasibility is uncertified")

    registry_name = "TRAIN_SUBSETS.json" if purpose == "train" else "ENDPOINT_TABLE.json"
    registry_path = freeze_dir / registry_name
    registry = _read(registry_path)
    current = sha256_file(data_path.resolve())
    entries = registry.get("files", registry.get("subsets", []))
    if isinstance(entries, dict):
        entries = list(entries.values())
    matching = [entry for entry in entries if isinstance(entry, dict) and entry.get("sha256") == current]
    if len(matching) != 1:
        raise RuntimeError(f"{purpose} data hash is not uniquely registered in {registry_name}")
    entry = matching[0]
    if purpose == "train" and not str(entry.get("arm", "")).startswith("Prov_"):
        if int(entry.get("items", entry.get("examples", 0))) != 20_000:
            raise RuntimeError("confirmatory training subset must contain exactly B=20000 items")
    if purpose == "eval":
        counts = registry.get("items_per_language", {})
        if len(counts) < 8 or any(int(count) < 120 for count in counts.values()):
            raise RuntimeError("endpoint violates the >=8 languages and >=120 items/language floors")

    expected = status.get("artifact_sha256", {})
    for name, path in {
        "KEY_AUDIT.json": key_audit_path,
        registry_name: registry_path,
        "DEDUP_VALIDATION.json": freeze_dir / "DEDUP_VALIDATION.json",
        "PREMISE_CHECK.json": premise_path,
        "ALLOCATION_CERTS.json": allocation_path,
    }.items():
        if expected.get(name) != sha256_file(path):
            raise RuntimeError(f"freeze hash missing or stale for {name}")
    return {"mode": "confirmatory", "data_sha256": current, "registry": str(registry_path)}


def operator_hard_open(project: Path) -> dict | None:
    """Operator A can launch without KEY_AUDIT / freeze files. This is not a PASS."""
    path = project / ".aris" / "OPERATOR_HARD_OPEN"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or str(value.get("by") or "").strip() != "A":
        return None
    if value.get("hard_open_without_key_audit") is not True:
        return None
    if value.get("hard_open_without_freeze") is not True:
        return None
    return value


def authorize_model_command(
    project: Path, data_path: Path, *, purpose: str, pilot_mode: bool, image_root: Path | None = None
) -> dict:
    if purpose not in {"train", "eval"}:
        raise ValueError("purpose must be train or eval")
    stamp = operator_hard_open(project)
    if stamp is not None and not pilot_mode:
        resolved = data_path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"hard-open {purpose} data missing: {resolved}")
        return {
            "mode": "operator_hard_open",
            "data_sha256": sha256_file(resolved),
            "stamp": str((project / ".aris" / "OPERATOR_HARD_OPEN").resolve()),
            "note": "Operator A authorized launch without KEY_AUDIT/freeze; not confirmatory.",
        }
    if image_root is not None and not pilot_mode:
        raise RuntimeError("image_root relocation is allowed only for the pilot receipt path")
    return (
        authorize_pilot(project, data_path, image_root=image_root)
        if pilot_mode
        else authorize_confirmatory(project, data_path, purpose=purpose)
    )
