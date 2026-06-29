from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import foundation_crack_classifier.train as train_module
from foundation_crack_classifier.dataset import DatasetSplit, ImageRecord
from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.model import DEFAULT_BACKBONE
from foundation_crack_classifier.thresholds import ThresholdConfig
from foundation_crack_classifier.train import (
    _class_weights,
    _parse_args,
    _run_eval_epoch,
    _run_train_epoch,
    main,
)


def test_parse_args_uses_environment_defaults(monkeypatch):
    monkeypatch.setenv("FCC_DATA_DIR", "/tmp/custom-data")
    monkeypatch.setenv("FCC_MODEL_DIR", "/tmp/custom-models")
    monkeypatch.setenv("FCC_BACKBONE", "resnet18")
    monkeypatch.setenv("FCC_EPOCHS", "9")
    monkeypatch.setenv("FCC_BATCH_SIZE", "7")
    monkeypatch.setenv("FCC_LEARNING_RATE", "0.001")
    monkeypatch.setenv("FCC_IMAGE_SIZE", "128")
    monkeypatch.setenv("FCC_SEED", "99")
    monkeypatch.setenv("FCC_PRETRAINED", "no")

    args = _parse_args([])

    assert args.data_dir == Path("/tmp/custom-data")
    assert args.model_dir == Path("/tmp/custom-models")
    assert args.backbone == "resnet18"
    assert args.epochs == 9
    assert args.batch_size == 7
    assert args.learning_rate == 0.001
    assert args.image_size == 128
    assert args.seed == 99
    assert args.pretrained is False


def test_parse_args_prefers_cli_args_over_environment_defaults(monkeypatch):
    monkeypatch.setenv("FCC_DATA_DIR", "/tmp/env-data")
    monkeypatch.setenv("FCC_MODEL_DIR", "/tmp/env-models")
    monkeypatch.setenv("FCC_BACKBONE", "env-backbone")
    monkeypatch.setenv("FCC_EPOCHS", "9")
    monkeypatch.setenv("FCC_BATCH_SIZE", "7")
    monkeypatch.setenv("FCC_LEARNING_RATE", "0.001")
    monkeypatch.setenv("FCC_IMAGE_SIZE", "128")
    monkeypatch.setenv("FCC_SEED", "99")
    monkeypatch.setenv("FCC_PRETRAINED", "false")

    args = _parse_args(
        [
            "--data-dir",
            "/tmp/cli-data",
            "--model-dir",
            "/tmp/cli-models",
            "--backbone",
            DEFAULT_BACKBONE,
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.02",
            "--image-size",
            "64",
            "--seed",
            "11",
            "--pretrained",
        ]
    )

    assert args.data_dir == Path("/tmp/cli-data")
    assert args.model_dir == Path("/tmp/cli-models")
    assert args.backbone == DEFAULT_BACKBONE
    assert args.epochs == 3
    assert args.batch_size == 4
    assert args.learning_rate == 0.02
    assert args.image_size == 64
    assert args.seed == 11
    assert args.pretrained is True


def test_parse_args_defaults_to_pretrained_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("FCC_PRETRAINED", raising=False)

    args = _parse_args([])

    assert args.pretrained is True


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_parse_args_accepts_boolean_pretrained_env_values(
    monkeypatch, env_value, expected
):
    monkeypatch.setenv("FCC_PRETRAINED", env_value)

    args = _parse_args([])

    assert args.pretrained is expected


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_error"),
    [
        ("FCC_EPOCHS", "later", "FCC_EPOCHS must be an integer"),
        ("FCC_BATCH_SIZE", "many", "FCC_BATCH_SIZE must be an integer"),
        ("FCC_LEARNING_RATE", "fast", "FCC_LEARNING_RATE must be a number"),
        ("FCC_IMAGE_SIZE", "large", "FCC_IMAGE_SIZE must be an integer"),
        ("FCC_SEED", "random", "FCC_SEED must be an integer"),
        ("FCC_PRETRAINED", "sometimes", "FCC_PRETRAINED must be a boolean value"),
    ],
)
def test_parse_args_rejects_invalid_environment_defaults(
    monkeypatch, capsys, env_name, env_value, expected_error
):
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(SystemExit) as exc_info:
        _parse_args([])

    assert exc_info.value.code == 2
    assert expected_error in capsys.readouterr().err


def test_parse_args_cli_values_override_invalid_environment_defaults(monkeypatch):
    monkeypatch.setenv("FCC_EPOCHS", "later")
    monkeypatch.setenv("FCC_BATCH_SIZE", "many")
    monkeypatch.setenv("FCC_LEARNING_RATE", "fast")
    monkeypatch.setenv("FCC_IMAGE_SIZE", "large")
    monkeypatch.setenv("FCC_SEED", "random")
    monkeypatch.setenv("FCC_PRETRAINED", "sometimes")

    args = _parse_args(
        [
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.01",
            "--image-size",
            "32",
            "--seed",
            "7",
            "--no-pretrained",
        ]
    )

    assert args.epochs == 2
    assert args.batch_size == 3
    assert args.learning_rate == 0.01
    assert args.image_size == 32
    assert args.seed == 7
    assert args.pretrained is False


@pytest.mark.parametrize(
    ("cli_args", "expected_error"),
    [
        (["--epochs", "0"], "--epochs must be >= 1"),
        (["--batch-size", "0"], "--batch-size must be >= 1"),
        (["--learning-rate", "0"], "--learning-rate must be a finite number > 0"),
        (["--learning-rate", "nan"], "--learning-rate must be a finite number > 0"),
        (["--learning-rate", "inf"], "--learning-rate must be a finite number > 0"),
        (["--image-size", "0"], "--image-size must be >= 1"),
    ],
)
def test_parse_args_rejects_invalid_cli_values(capsys, cli_args, expected_error):
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(cli_args)

    assert exc_info.value.code == 2
    assert expected_error in capsys.readouterr().err


@pytest.mark.parametrize("env_value", ["nan", "inf"])
def test_parse_args_rejects_non_finite_environment_learning_rate(
    monkeypatch, capsys, env_value
):
    monkeypatch.setenv("FCC_LEARNING_RATE", env_value)

    with pytest.raises(SystemExit) as exc_info:
        _parse_args([])

    assert exc_info.value.code == 2
    assert "--learning-rate must be a finite number > 0" in capsys.readouterr().err


def test_class_weights_are_equal_for_balanced_records(tmp_path):
    records = [
        ImageRecord(path=tmp_path / label / f"{index}.jpg", label=label)
        for label in LABELS
        for index in range(2)
    ]

    weights = _class_weights(records)

    assert torch.allclose(weights, torch.ones(len(LABELS)))


def test_class_weights_are_larger_for_rarer_classes(tmp_path):
    records = [
        ImageRecord(path=tmp_path / "level1" / f"{index}.jpg", label="level1")
        for index in range(4)
    ]
    records.extend(
        ImageRecord(path=tmp_path / label / f"{index}.jpg", label=label)
        for label in ["level2", "level3", "unclear"]
        for index in range(2)
    )

    weights = _class_weights(records)

    level1_weight = weights[LABELS.index("level1")]
    for label in ["level2", "level3", "unclear"]:
        assert weights[LABELS.index(label)] > level1_weight
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_run_train_epoch_updates_model_and_returns_average_loss():
    torch.manual_seed(123)
    loader = _tiny_loader()
    model = _tiny_model()
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    loss = _run_train_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loss > 0
    assert model.training is True
    assert not torch.equal(model.weight.detach(), before)


def test_run_eval_epoch_returns_average_loss_without_training():
    torch.manual_seed(123)
    loader = _tiny_loader()
    model = _tiny_model()
    before = model.weight.detach().clone()
    loss_fn = nn.CrossEntropyLoss()

    loss = _run_eval_epoch(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    assert loss > 0
    assert model.training is False
    assert torch.equal(model.weight.detach(), before)


def test_main_exits_clearly_for_empty_required_label_folders(tmp_path):
    data_dir = tmp_path / "training_images"
    for label in LABELS:
        (data_dir / label).mkdir(parents=True)

    with pytest.raises(SystemExit, match="No supported images found"):
        main(["--data-dir", str(data_dir), "--epochs", "1"])


def test_main_exits_clearly_for_unreadable_training_image(tmp_path):
    data_dir = tmp_path / "training_images"
    bad_path = data_dir / LABELS[0] / "bad.jpg"
    for label in LABELS:
        label_dir = data_dir / label
        label_dir.mkdir(parents=True)
        (label_dir / "bad.jpg").write_text("not an image")

    with pytest.raises(SystemExit) as exc_info:
        main(["--data-dir", str(data_dir), "--epochs", "1"])

    assert str(bad_path) in str(exc_info.value)


def test_main_saves_best_checkpoint_and_writes_metadata(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    save_calls: list[dict] = []
    written_json: dict[Path, dict] = {}
    created_model: dict[str, object] = {}

    _patch_fast_main(
        monkeypatch,
        tmp_path,
        validation_loss=0.4,
        save_calls=save_calls,
        written_json=written_json,
        created_model=created_model,
    )

    main(
        [
            "--data-dir",
            str(tmp_path / "training_images"),
            "--model-dir",
            str(model_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--image-size",
            "64",
            "--seed",
            "123",
            "--no-pretrained",
        ]
    )

    assert created_model == {"backbone": DEFAULT_BACKBONE, "pretrained": False}
    assert len(save_calls) == 1
    assert save_calls[0]["output_path"] == model_dir / "crack_severity_model.pt"
    assert save_calls[0]["backbone"] == DEFAULT_BACKBONE
    assert save_calls[0]["image_size"] == 64
    assert save_calls[0]["threshold_config"] == ThresholdConfig().__dict__
    assert model_dir / "label_map.json" in written_json
    assert model_dir / "training_config.json" in written_json

    config = written_json[model_dir / "training_config.json"]
    assert config["pretrained"] is False
    assert config["train_count"] == 8
    assert config["validation_count"] == 4
    assert config["test_count"] == 4
    assert config["seed"] == 123
    assert config["image_size"] == 64
    assert config["threshold_config"] == ThresholdConfig().__dict__
    assert config["class_counts"] == {
        "train": {label: 2 for label in LABELS},
        "validation": {label: 1 for label in LABELS},
        "test": {label: 1 for label in LABELS},
    }
    assert isinstance(config["class_weights"], list)
    assert len(config["class_weights"]) == len(LABELS)


def test_main_exits_when_no_checkpoint_is_saved(monkeypatch, tmp_path):
    _patch_fast_main(
        monkeypatch,
        tmp_path,
        validation_loss=float("inf"),
        save_calls=[],
        written_json={},
        created_model={},
    )

    with pytest.raises(SystemExit, match="without saving a checkpoint"):
        main(["--data-dir", str(tmp_path / "training_images"), "--epochs", "1"])


def _tiny_loader():
    inputs = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    targets = torch.tensor([0, 1, 2, 3])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def _tiny_model():
    return nn.Linear(2, len(LABELS))


def _patch_fast_main(
    monkeypatch,
    tmp_path,
    validation_loss,
    save_calls: list[dict],
    written_json: dict[Path, dict],
    created_model: dict[str, object],
) -> None:
    split = DatasetSplit(
        train=_records(tmp_path, "train", 2),
        validation=_records(tmp_path, "validation", 1),
        test=_records(tmp_path, "test", 1),
    )
    records = split.train + split.validation + split.test

    monkeypatch.setattr(
        train_module,
        "discover_images",
        lambda data_dir, require_all_labels: records,
    )
    monkeypatch.setattr(train_module, "_verify_image_files", lambda records: None)
    monkeypatch.setattr(train_module, "split_records", lambda records, seed: split)
    monkeypatch.setattr(
        train_module,
        "build_train_transform",
        lambda image_size: ("train-transform", image_size),
    )
    monkeypatch.setattr(
        train_module,
        "build_eval_transform",
        lambda image_size: ("eval-transform", image_size),
    )
    monkeypatch.setattr(
        train_module,
        "CrackImageDataset",
        lambda records, transform: {"records": records, "transform": transform},
    )
    monkeypatch.setattr(
        train_module,
        "DataLoader",
        lambda dataset, batch_size, shuffle: {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
        },
    )

    def fake_create_model(backbone, pretrained):
        created_model["backbone"] = backbone
        created_model["pretrained"] = pretrained
        return _tiny_model()

    monkeypatch.setattr(train_module, "create_model", fake_create_model)
    monkeypatch.setattr(train_module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train_module, "_run_train_epoch", lambda *args: 0.5)
    monkeypatch.setattr(train_module, "_run_eval_epoch", lambda *args: validation_loss)
    monkeypatch.setattr(
        train_module,
        "save_checkpoint",
        lambda **kwargs: save_calls.append(kwargs),
    )
    monkeypatch.setattr(
        train_module,
        "write_json",
        lambda path, payload: written_json.setdefault(path, payload),
    )


def _records(tmp_path, split_name: str, count_per_label: int) -> list[ImageRecord]:
    return [
        ImageRecord(
            path=tmp_path / "training_images" / label / split_name / f"{index}.jpg",
            label=label,
        )
        for label in LABELS
        for index in range(count_per_label)
    ]
