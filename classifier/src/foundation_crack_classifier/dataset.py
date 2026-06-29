from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from foundation_crack_classifier.labels import LABELS, LABEL_TO_INDEX


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_RECORDS_PER_LABEL = 5


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


@dataclass(frozen=True)
class DatasetSplit:
    train: list[ImageRecord]
    validation: list[ImageRecord]
    test: list[ImageRecord]


def discover_images(data_dir: Path, require_all_labels: bool = False) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []

    for label in LABELS:
        label_dir = data_dir / label
        if not label_dir.exists():
            if require_all_labels:
                missing_labels.append(label)
            continue

        label_records: list[ImageRecord] = []
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                label_records.append(ImageRecord(path=path, label=label))

        if require_all_labels and not label_records:
            empty_labels.append(label)
        records.extend(label_records)

    if missing_labels:
        raise ValueError(f"Missing label folder(s): {', '.join(missing_labels)}")
    if empty_labels:
        raise ValueError(
            f"No supported images found for label folder(s): {', '.join(empty_labels)}"
        )

    return records


def split_records(
    records: list[ImageRecord],
    seed: int = 42,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> DatasetSplit:
    if (
        not 0 <= train_ratio <= 1
        or not 0 <= validation_ratio <= 1
        or train_ratio + validation_ratio > 1
    ):
        raise ValueError(
            "Invalid split ratios: train_ratio and validation_ratio must be between "
            "0 and 1, and their sum must be less than or equal to 1."
        )

    grouped = {label: [] for label in LABELS}
    for record in records:
        if record.label not in grouped:
            raise ValueError(f"Unknown label: {record.label}")
        grouped[record.label].append(record)

    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    test: list[ImageRecord] = []
    rng = random.Random(seed)

    for label in LABELS:
        label_records = list(grouped[label])
        if len(label_records) < MIN_RECORDS_PER_LABEL:
            raise ValueError(
                f"Expected at least {MIN_RECORDS_PER_LABEL} records for label "
                f"{label}; found {len(label_records)}."
            )

        rng.shuffle(label_records)
        train_count = int(len(label_records) * train_ratio)
        validation_count = int(len(label_records) * validation_ratio)
        train.extend(label_records[:train_count])
        validation.extend(label_records[train_count : train_count + validation_count])
        test.extend(label_records[train_count + validation_count :])

    return DatasetSplit(
        train=sorted(train, key=lambda record: str(record.path)),
        validation=sorted(validation, key=lambda record: str(record.path)),
        test=sorted(test, key=lambda record: str(record.path)),
    )


def build_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class CrackImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, LABEL_TO_INDEX[record.label]
