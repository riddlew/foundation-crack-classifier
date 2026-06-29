from pathlib import Path
import json

import pytest
import torch
from PIL import Image

import foundation_crack_classifier.infer as infer_module
from foundation_crack_classifier.infer import main, parse_args, predict_probabilities
from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.schemas import InferenceResult


def test_parse_args_uses_environment_default_model_path(monkeypatch):
    monkeypatch.setenv("FCC_MODEL_DIR", "/tmp/custom-models")

    args = parse_args(["input.jpg"])

    assert args.image_path == Path("input.jpg")
    assert args.model_path == Path("/tmp/custom-models") / "crack_severity_model.pt"


def test_main_exits_clearly_when_image_file_is_missing(tmp_path):
    missing_image = tmp_path / "missing.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"

    with pytest.raises(SystemExit) as exc_info:
        main([str(missing_image), "--model-path", str(model_path)])

    assert str(exc_info.value) == f"Image file not found: {missing_image}"


def test_main_exits_clearly_when_model_file_is_missing(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"image")
    missing_model = tmp_path / "models" / "missing.pt"

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(missing_model)])

    assert str(exc_info.value) == f"Model file not found: {missing_model}"


def test_main_exits_clearly_when_image_path_is_directory(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.mkdir()
    model_path = tmp_path / "models" / "crack_severity_model.pt"

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == f"Image file not found: {image_path}"


def test_main_exits_clearly_when_model_path_is_directory(tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    image_path.write_bytes(b"image")
    model_path.mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == f"Model file not found: {model_path}"


def test_main_exits_clearly_when_checkpoint_is_invalid(monkeypatch, tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "invalid.pt"
    image_path.write_bytes(b"image")
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(infer_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module,
        "load_checkpoint",
        lambda model_path, map_location: (_ for _ in ()).throw(
            ValueError("Checkpoint labels do not match LABELS.")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == "Checkpoint labels do not match LABELS."


def test_main_exits_clearly_when_checkpoint_payload_is_not_mapping(tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "invalid.pt"
    image_path.write_bytes(b"image")
    model_path.parent.mkdir()
    torch.save(["bad"], model_path)

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == "Checkpoint payload must be a mapping."


def test_main_exits_clearly_when_checkpoint_file_is_corrupt(tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "corrupt.pt"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    model_path.parent.mkdir()
    model_path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == f"Unable to load model checkpoint: {model_path}"


def test_main_exits_clearly_when_image_file_is_corrupt(monkeypatch, tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    image_path.write_bytes(b"not an image")
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(infer_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module,
        "load_checkpoint",
        lambda model_path, map_location: (_FakeModel(), {"image_size": 128}),
    )
    monkeypatch.setattr(
        infer_module,
        "build_eval_transform",
        lambda image_size: lambda image: torch.ones(3, 4, 4),
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value) == f"Unable to read image file: {image_path}"


def test_main_exits_clearly_when_threshold_config_is_invalid(monkeypatch, tmp_path):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    image_path.write_bytes(b"image")
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(infer_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module,
        "load_checkpoint",
        lambda model_path, map_location: (
            _FakeModel(),
            {"image_size": 128, "threshold_config": ["bad"]},
        ),
    )
    monkeypatch.setattr(
        infer_module,
        "predict_probabilities",
        lambda model, path, transform, device: {
            "level1": 0.1,
            "level2": 0.7,
            "level3": 0.1,
            "unclear": 0.1,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert (
        str(exc_info.value)
        == f"Invalid checkpoint threshold configuration: {model_path}"
    )


def test_main_exits_clearly_when_model_probabilities_are_invalid(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    image_path.write_bytes(b"image")
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(infer_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module,
        "load_checkpoint",
        lambda model_path, map_location: (
            _FakeModel(),
            {"image_size": 128, "threshold_config": {}},
        ),
    )
    monkeypatch.setattr(
        infer_module,
        "predict_probabilities",
        lambda model, path, transform, device: {
            "level1": 0.2,
            "level2": 0.8,
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(image_path), "--model-path", str(model_path)])

    assert str(exc_info.value).startswith("Invalid model probabilities: ")


def test_predict_probabilities_maps_softmax_outputs_to_label_order(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    model = _FakeLogitModel(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    probabilities = predict_probabilities(
        model,
        image_path,
        lambda image: torch.ones(3, 4, 4),
        torch.device("cpu"),
    )

    expected = torch.softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    assert probabilities == {
        label: pytest.approx(float(expected[index]))
        for index, label in enumerate(LABELS)
    }
    assert model.eval_called is True
    assert model.input_shape == (1, 3, 4, 4)


def test_predict_probabilities_converts_images_to_rgb_before_transform(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGBA", (4, 4), color=(255, 255, 255, 128)).save(image_path)
    model = _FakeLogitModel(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    def assert_rgb_transform(image):
        assert image.mode == "RGB"
        return torch.ones(3, 4, 4)

    predict_probabilities(
        model,
        image_path,
        assert_rgb_transform,
        torch.device("cpu"),
    )


def test_main_loads_checkpoint_applies_thresholds_and_prints_json(
    monkeypatch, tmp_path, capsys
):
    image_path = tmp_path / "input.jpg"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    image_path.write_bytes(b"image")
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")
    fake_model = _FakeModel()
    result = InferenceResult.from_label(
        final_label="level2",
        confidence=0.7,
        raw_probabilities={
            "level1": 0.1,
            "level2": 0.7,
            "level3": 0.1,
            "unclear": 0.1,
        },
        why_this_result="test reason",
    )
    probabilities = {
        "level1": 0.1,
        "level2": 0.7,
        "level3": 0.1,
        "unclear": 0.1,
    }
    calls: dict[str, object] = {}

    def fake_load_checkpoint(path, map_location):
        calls["checkpoint_path"] = path
        calls["checkpoint_map_location"] = map_location
        return fake_model, {
            "image_size": 128,
            "threshold_config": {
                "minimum_confidence": 0.50,
                "urgent_threshold": 0.20,
            },
        }

    def fake_build_eval_transform(image_size):
        calls["image_size"] = image_size
        return ("eval-transform", image_size)

    def fake_predict_probabilities(model, path, transform, device):
        calls["predict"] = {
            "model": model,
            "path": path,
            "transform": transform,
            "device": device,
        }
        return probabilities

    def fake_apply_thresholds(raw_probabilities, config):
        calls["thresholds"] = {
            "probabilities": raw_probabilities,
            "config": config,
        }
        return result

    monkeypatch.setattr(infer_module, "get_device", lambda: "test-device")
    monkeypatch.setattr(infer_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(infer_module, "build_eval_transform", fake_build_eval_transform)
    monkeypatch.setattr(infer_module, "predict_probabilities", fake_predict_probabilities)
    monkeypatch.setattr(infer_module, "apply_thresholds", fake_apply_thresholds)

    main([str(image_path), "--model-path", str(model_path)])

    assert calls["checkpoint_path"] == model_path
    assert calls["checkpoint_map_location"] == "test-device"
    assert fake_model.device == "test-device"
    assert calls["image_size"] == 128
    assert calls["predict"] == {
        "model": fake_model,
        "path": image_path,
        "transform": ("eval-transform", 128),
        "device": "test-device",
    }
    assert calls["thresholds"]["probabilities"] == probabilities
    assert calls["thresholds"]["config"].minimum_confidence == 0.50
    assert calls["thresholds"]["config"].urgent_threshold == 0.20
    assert json.loads(capsys.readouterr().out) == result.to_dict()


class _FakeLogitModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.eval_called = False
        self.input_shape = None

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(tensor.shape)
        return self.logits


class _FakeModel:
    def __init__(self) -> None:
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self
