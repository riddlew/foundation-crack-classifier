from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import timm
import torch
from torch import nn

from foundation_crack_classifier.labels import LABELS


DEFAULT_BACKBONE = "mobilenetv3_small_100"
CHECKPOINT_VERSION = 1
REQUIRED_CHECKPOINT_KEYS = {
    "checkpoint_version",
    "model_state",
    "labels",
    "backbone",
    "image_size",
    "threshold_config",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(
    backbone: str = DEFAULT_BACKBONE, pretrained: bool = True
) -> nn.Module:
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=len(LABELS),
    )


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    backbone: str,
    image_size: int,
    threshold_config: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "labels": list(LABELS),
            "backbone": backbone,
            "image_size": image_size,
            "threshold_config": threshold_config,
        },
        output_path,
    )


def load_checkpoint(
    model_path: Path, map_location: torch.device | str | None = None
) -> tuple[nn.Module, dict]:
    checkpoint = _load_checkpoint_payload(model_path, map_location or "cpu")
    _validate_checkpoint(checkpoint)
    model = create_model(backbone=checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_checkpoint_payload(
    model_path: Path, map_location: torch.device | str
) -> dict:
    try:
        return torch.load(model_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


def _validate_checkpoint(checkpoint: object) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")

    missing_keys = sorted(REQUIRED_CHECKPOINT_KEYS - checkpoint.keys())
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: " + ", ".join(missing_keys)
        )

    checkpoint_version = checkpoint["checkpoint_version"]
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint_version}. "
            f"Expected {CHECKPOINT_VERSION}."
        )

    if checkpoint["labels"] != list(LABELS):
        raise ValueError("Checkpoint labels do not match LABELS.")
