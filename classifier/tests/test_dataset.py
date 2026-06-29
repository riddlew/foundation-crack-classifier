from pathlib import Path

import pytest
from PIL import Image

from foundation_crack_classifier.dataset import (
    CrackImageDataset,
    ImageRecord,
    build_eval_transform,
    discover_images,
    split_records,
)
from foundation_crack_classifier.labels import LABELS, LABEL_TO_INDEX


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=(120, 120, 120))
    image.save(path)


def _write_records(root: Path, count_per_label: int = 5) -> None:
    for label in LABELS:
        for index in range(count_per_label):
            _write_image(root / label / f"{index}.jpg")


def test_discover_images_uses_expected_label_folders(tmp_path):
    _write_image(tmp_path / "level1" / "a.jpg")
    _write_image(tmp_path / "level2" / "b.png")
    _write_image(tmp_path / "level3" / "c.webp")
    _write_image(tmp_path / "unclear" / "d.jpeg")
    (tmp_path / "level1" / "notes.txt").write_text("not an image")

    records = discover_images(tmp_path)

    assert [record.label for record in records] == [
        "level1",
        "level2",
        "level3",
        "unclear",
    ]
    assert all(record.path.exists() for record in records)


def test_discover_images_finds_nested_images_with_uppercase_extensions(tmp_path):
    _write_image(tmp_path / "level1" / "nested" / "a.JPG")

    records = discover_images(tmp_path)

    assert len(records) == 1
    assert records[0].label == "level1"
    assert records[0].path.name == "a.JPG"


def test_discover_images_can_require_all_label_folders(tmp_path):
    for label in ["level1", "level2", "level3"]:
        _write_image(tmp_path / label / "a.jpg")

    with pytest.raises(ValueError, match="Missing label folder.*unclear"):
        discover_images(tmp_path, require_all_labels=True)


def test_discover_images_can_require_supported_images_in_each_label_folder(tmp_path):
    for label in LABELS:
        (tmp_path / label).mkdir()
    for label in ["level2", "level3", "unclear"]:
        _write_image(tmp_path / label / "a.jpg")
    (tmp_path / "level1" / "notes.txt").write_text("not an image")

    with pytest.raises(ValueError, match="No supported images.*level1"):
        discover_images(tmp_path, require_all_labels=True)


def test_split_records_is_deterministic(tmp_path):
    _write_records(tmp_path)
    records = discover_images(tmp_path)

    first = split_records(records, seed=123)
    second = split_records(records, seed=123)

    assert first == second
    assert len(first.train) == 12
    assert len(first.validation) == 4
    assert len(first.test) == 4


def test_split_records_rejects_unknown_labels(tmp_path):
    record = ImageRecord(path=tmp_path / "unknown.jpg", label="other")

    with pytest.raises(ValueError, match="Unknown label: other"):
        split_records([record])


def test_split_records_requires_enough_records_per_label(tmp_path):
    _write_records(tmp_path, count_per_label=4)
    records = discover_images(tmp_path)

    with pytest.raises(ValueError, match="at least 5 records.*level1"):
        split_records(records)


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio"),
    [
        (-0.1, 0.2),
        (1.1, 0.2),
        (0.6, -0.1),
        (0.6, 1.1),
        (0.8, 0.3),
    ],
)
def test_split_records_rejects_invalid_ratios(tmp_path, train_ratio, validation_ratio):
    _write_records(tmp_path)
    records = discover_images(tmp_path)

    with pytest.raises(ValueError, match="split ratios"):
        split_records(
            records,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )


def test_crack_image_dataset_loads_tensor_and_label_index(tmp_path):
    image_path = tmp_path / "level2" / "a.jpg"
    _write_image(image_path)
    records = discover_images(tmp_path)
    dataset = CrackImageDataset(records, build_eval_transform(image_size=16))

    tensor, label_index = dataset[0]

    assert tuple(tensor.shape) == (3, 16, 16)
    assert label_index == LABEL_TO_INDEX["level2"]
