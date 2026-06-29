from pathlib import Path

import pytest

import foundation_crack_classifier.evaluate as evaluate_module
from foundation_crack_classifier.dataset import DatasetSplit, ImageRecord
from foundation_crack_classifier.evaluate import _build_report, main, parse_args
from foundation_crack_classifier.labels import LABELS


def test_parse_args_uses_environment_defaults(monkeypatch):
    monkeypatch.setenv("FCC_DATA_DIR", "/tmp/custom-data")
    monkeypatch.setenv("FCC_MODEL_DIR", "/tmp/custom-models")
    monkeypatch.setenv("FCC_REPORT_DIR", "/tmp/custom-reports")

    args = parse_args([])

    assert args.data_dir == Path("/tmp/custom-data")
    assert args.model_path == Path("/tmp/custom-models") / "crack_severity_model.pt"
    assert args.report_dir == Path("/tmp/custom-reports")
    assert args.batch_size == 16
    assert args.seed == 42


def test_parse_args_rejects_invalid_batch_size(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--batch-size", "0"])

    assert exc_info.value.code == 2
    assert "--batch-size must be >= 1" in capsys.readouterr().err


def test_main_exits_clearly_when_model_file_is_missing(tmp_path):
    missing_model = tmp_path / "models" / "missing.pt"

    with pytest.raises(SystemExit) as exc_info:
        main(["--model-path", str(missing_model)])

    assert str(exc_info.value) == f"Model file not found: {missing_model}"


def test_main_exits_clearly_when_checkpoint_is_invalid(monkeypatch, tmp_path):
    model_path = tmp_path / "models" / "invalid.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(evaluate_module, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint",
        lambda model_path, map_location: (_ for _ in ()).throw(
            ValueError("Checkpoint labels do not match LABELS.")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--model-path", str(model_path)])

    assert str(exc_info.value) == "Checkpoint labels do not match LABELS."


def test_main_loads_checkpoint_and_writes_evaluation_report(
    monkeypatch, tmp_path, capsys
):
    data_dir = tmp_path / "training_images"
    model_path = tmp_path / "models" / "crack_severity_model.pt"
    report_dir = tmp_path / "reports"
    model_path.parent.mkdir()
    model_path.write_bytes(b"checkpoint")
    fake_model = _FakeModel()
    report = {"accuracy": 1.0}
    y_true = [0, 1]
    y_pred = [0, 1]
    split = DatasetSplit(
        train=_records(tmp_path, "train", 1),
        validation=_records(tmp_path, "validation", 1),
        test=_records(tmp_path, "test", 1),
    )
    records = split.train + split.validation + split.test
    calls: dict[str, object] = {}

    def fake_load_checkpoint(path, map_location):
        calls["checkpoint_path"] = path
        calls["checkpoint_map_location"] = map_location
        return fake_model, {"image_size": 128}

    def fake_discover_images(path, require_all_labels):
        calls["data_dir"] = path
        calls["require_all_labels"] = require_all_labels
        return records

    def fake_split_records(discovered_records, seed):
        calls["split_records"] = discovered_records
        calls["seed"] = seed
        return split

    def fake_build_eval_transform(image_size):
        calls["image_size"] = image_size
        return ("eval-transform", image_size)

    def fake_dataset(dataset_records, transform):
        dataset = {"records": dataset_records, "transform": transform}
        calls["dataset"] = dataset
        return dataset

    def fake_loader(dataset, batch_size, shuffle):
        loader = {"dataset": dataset, "batch_size": batch_size, "shuffle": shuffle}
        calls["loader"] = loader
        return loader

    def fake_predict(model, loader, device):
        calls["predict"] = {"model": model, "loader": loader, "device": device}
        return y_true, y_pred

    def fake_build_report(report_y_true, report_y_pred):
        calls["report_inputs"] = (report_y_true, report_y_pred)
        return report

    def fake_write_confusion_matrix(matrix_y_true, matrix_y_pred, output_path):
        calls["confusion_matrix"] = {
            "y_true": matrix_y_true,
            "y_pred": matrix_y_pred,
            "output_path": output_path,
        }

    def fake_write_json(path, payload):
        calls["json"] = {"path": path, "payload": payload}

    monkeypatch.setattr(evaluate_module, "get_device", lambda: "test-device")
    monkeypatch.setattr(evaluate_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(evaluate_module, "discover_images", fake_discover_images)
    monkeypatch.setattr(evaluate_module, "split_records", fake_split_records)
    monkeypatch.setattr(
        evaluate_module, "build_eval_transform", fake_build_eval_transform
    )
    monkeypatch.setattr(evaluate_module, "CrackImageDataset", fake_dataset)
    monkeypatch.setattr(evaluate_module, "DataLoader", fake_loader)
    monkeypatch.setattr(evaluate_module, "_predict", fake_predict)
    monkeypatch.setattr(evaluate_module, "_build_report", fake_build_report)
    monkeypatch.setattr(
        evaluate_module, "_write_confusion_matrix", fake_write_confusion_matrix
    )
    monkeypatch.setattr(evaluate_module, "write_json", fake_write_json)

    main(
        [
            "--data-dir",
            str(data_dir),
            "--model-path",
            str(model_path),
            "--report-dir",
            str(report_dir),
            "--batch-size",
            "2",
            "--seed",
            "123",
        ]
    )

    assert calls["checkpoint_path"] == model_path
    assert calls["checkpoint_map_location"] == "test-device"
    assert fake_model.device == "test-device"
    assert calls["data_dir"] == data_dir
    assert calls["require_all_labels"] is True
    assert calls["split_records"] == records
    assert calls["seed"] == 123
    assert calls["image_size"] == 128
    assert calls["dataset"] == {
        "records": split.test,
        "transform": ("eval-transform", 128),
    }
    assert calls["loader"] == {
        "dataset": calls["dataset"],
        "batch_size": 2,
        "shuffle": False,
    }
    assert calls["predict"] == {
        "model": fake_model,
        "loader": calls["loader"],
        "device": "test-device",
    }
    assert calls["report_inputs"] == (y_true, y_pred)
    assert calls["json"] == {
        "path": report_dir / "evaluation.json",
        "payload": report,
    }
    assert calls["confusion_matrix"] == {
        "y_true": y_true,
        "y_pred": y_pred,
        "output_path": report_dir / "confusion_matrix.png",
    }
    assert report_dir.exists()
    stdout = capsys.readouterr().out
    assert f"evaluation_report={report_dir / 'evaluation.json'}" in stdout


def test_build_report_includes_metrics_and_high_risk_misses():
    y_true = [
        LABELS.index("level1"),
        LABELS.index("level2"),
        LABELS.index("level2"),
        LABELS.index("level3"),
        LABELS.index("level3"),
        LABELS.index("unclear"),
    ]
    y_pred = [
        LABELS.index("level1"),
        LABELS.index("level1"),
        LABELS.index("level2"),
        LABELS.index("level1"),
        LABELS.index("level3"),
        LABELS.index("unclear"),
    ]

    report = _build_report(y_true, y_pred)

    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["confusion_matrix"] == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert report["high_risk_misses"] == {
        "actual_level3_predicted_level1": 1,
        "actual_level2_predicted_level1": 1,
    }
    for label in LABELS:
        assert label in report["classification_report"]


class _FakeModel:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = device
        return self


def _records(tmp_path, split_name: str, count_per_label: int) -> list[ImageRecord]:
    return [
        ImageRecord(
            path=tmp_path / "training_images" / label / split_name / f"{index}.jpg",
            label=label,
        )
        for label in LABELS
        for index in range(count_per_label)
    ]
