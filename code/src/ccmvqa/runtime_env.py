"""Select the torch/torchvision build the model path is allowed to import.

The interpreter's own ``site-packages`` holds ``torch 2.12.1+cpu`` with no
``torchvision``.  Transformers 5.9 routes every Qwen2.5-VL image processor through
``AutoImageProcessor``/``Qwen2VLVideoProcessor``, both of which hard-require
torchvision, so the *processor cannot even be constructed* without it, and the only
torchvision wheel reachable from this host (PyPI ``0.27.1``) is linked against
``libtorch_cuda.so``/``libcudart.so.13`` and therefore refuses to load next to the
``+cpu`` torch.

Rather than overwrite the interpreter's torch — which could not be restored if the
swap went wrong — the matching CUDA-built pair is unpacked side by side in
``VENDORED_TORCH_ROOT`` and prepended to ``sys.path`` here, before anything imports
torch.  The build is CUDA-linked but runs on CPU on this host (``torch.cuda`` simply
reports no device); it changes which shared objects are loaded, not what is computed.

This module is imported for its effect by the evaluation launch and by the contract
tests, so the choice of build is part of the reviewed source rather than an invisible
environment variable at launch time.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDORED_TORCH_ROOT = Path.home() / ".cache" / "aris_torch_cuda"


def activate_vendored_torch(root: Path = VENDORED_TORCH_ROOT) -> dict:
    """Prepend the vendored torch build to ``sys.path`` and describe what was selected.

    Returns a small receipt so a run summary can record which build executed. Raises
    if a *different* torch has already been imported: the path would then have no
    effect and the process would silently run on a torchvision-less build. An already
    imported torch that is itself the vendored build is the intended end state, so it
    is reported as active rather than refused — this is what lets one process import
    the evaluation module first and the dependency capture afterwards.
    """
    root = Path(root)
    if not (root / "torch").is_dir():
        return {"vendored_torch_root": str(root), "active": False, "reason": "not installed"}
    imported = sys.modules.get("torch")
    already_vendored = False
    if imported is not None:
        imported_file = getattr(imported, "__file__", None)
        if imported_file is None or not Path(imported_file).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(
                "torch was imported before activate_vendored_torch(); the vendored build cannot take effect"
            )
        already_vendored = True
    entry = str(root)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    reason = (
        "the vendored build was already imported in this process"
        if already_vendored
        else "torchvision is required by the Qwen2.5-VL processor"
    )
    return {"vendored_torch_root": entry, "active": True, "reason": reason}


def torch_build_receipt() -> dict:
    """Report the torch/torchvision actually imported, for the run summary."""
    import torch
    import torchvision

    return {
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
