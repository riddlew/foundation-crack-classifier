import json

import pytest
import torch

from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.model import (
    CHECKPOINT_VERSION,
    DEFAULT_BACKBONE,
    create_model,
    get_device,
    load_checkpoint,
    save_checkpoint,
    write_json,
)


def test_get_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _FakeMps(is_available=True), raising=False)

    assert get_device().type == "cuda"


def test_get_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _FakeMps(is_available=True), raising=False)

    assert get_device().type == "mps"


def test_get_device_uses_cpu_when_accelerators_are_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _FakeMps(is_available=False), raising=False)

    assert get_device().type == "cpu"


def test_create_model_uses_project_label_count():
    model = create_model(pretrained=False)

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))

    assert output.shape == (1, len(LABELS))


def test_save_checkpoint_writes_model_and_training_metadata(tmp_path):
    model = create_model(pretrained=False)
    output_path = tmp_path / "models" / "checkpoint.pt"
    threshold_config = {"minimum_confidence": 0.65}

    save_checkpoint(
        model=model,
        output_path=output_path,
        backbone=DEFAULT_BACKBONE,
        image_size=224,
        threshold_config=threshold_config,
    )

    checkpoint = _safe_torch_load(output_path)
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["labels"] == list(LABELS)
    assert checkpoint["backbone"] == DEFAULT_BACKBONE
    assert checkpoint["image_size"] == 224
    assert checkpoint["threshold_config"] == threshold_config
    assert checkpoint["model_state"].keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(checkpoint["model_state"][key], value)


def test_load_checkpoint_round_trips_model_in_eval_mode(tmp_path):
    model = create_model(pretrained=False)
    model.train()
    output_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        output_path=output_path,
        backbone=DEFAULT_BACKBONE,
        image_size=224,
        threshold_config={"minimum_confidence": 0.65},
    )

    loaded_model, checkpoint = load_checkpoint(output_path)

    assert checkpoint["backbone"] == DEFAULT_BACKBONE
    assert loaded_model.training is False
    assert loaded_model.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(loaded_model.state_dict()[key], value)


def test_load_checkpoint_rejects_label_mismatch(tmp_path):
    output_path = _save_test_checkpoint(tmp_path)
    checkpoint = _safe_torch_load(output_path)
    checkpoint["labels"] = list(reversed(LABELS))
    torch.save(checkpoint, output_path)

    with pytest.raises(ValueError, match="Checkpoint labels do not match LABELS."):
        load_checkpoint(output_path)


def test_load_checkpoint_rejects_missing_required_key(tmp_path):
    output_path = _save_test_checkpoint(tmp_path)
    checkpoint = _safe_torch_load(output_path)
    del checkpoint["threshold_config"]
    torch.save(checkpoint, output_path)

    with pytest.raises(
        ValueError, match="Checkpoint is missing required keys: threshold_config"
    ):
        load_checkpoint(output_path)


def test_load_checkpoint_rejects_non_mapping_payload(tmp_path):
    output_path = tmp_path / "checkpoint.pt"
    torch.save(["bad"], output_path)

    with pytest.raises(ValueError, match="Checkpoint payload must be a mapping."):
        load_checkpoint(output_path)


def test_load_checkpoint_rejects_wrong_checkpoint_version(tmp_path):
    output_path = _save_test_checkpoint(tmp_path)
    checkpoint = _safe_torch_load(output_path)
    checkpoint["checkpoint_version"] = CHECKPOINT_VERSION + 1
    torch.save(checkpoint, output_path)

    with pytest.raises(
        ValueError,
        match=(
            f"Unsupported checkpoint version: {CHECKPOINT_VERSION + 1}. "
            f"Expected {CHECKPOINT_VERSION}."
        ),
    ):
        load_checkpoint(output_path)


def test_write_json_writes_sorted_indented_json_with_trailing_newline(tmp_path):
    output_path = tmp_path / "reports" / "payload.json"

    write_json(output_path, {"z": 1, "a": {"b": 2}})

    assert output_path.read_text() == '{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n'
    assert json.loads(output_path.read_text()) == {"a": {"b": 2}, "z": 1}


def _save_test_checkpoint(tmp_path):
    output_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model=create_model(pretrained=False),
        output_path=output_path,
        backbone=DEFAULT_BACKBONE,
        image_size=224,
        threshold_config={"minimum_confidence": 0.65},
    )
    return output_path


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class _FakeMps:
    def __init__(self, is_available):
        self._is_available = is_available

    def is_available(self):
        return self._is_available
