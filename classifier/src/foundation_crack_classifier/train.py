from __future__ import annotations

from argparse import ArgumentParser
import math
from pathlib import Path
import os

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from foundation_crack_classifier.dataset import (
    CrackImageDataset,
    build_eval_transform,
    build_train_transform,
    discover_images,
    split_records,
)
from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.model import (
    DEFAULT_BACKBONE,
    create_model,
    get_device,
    save_checkpoint,
    write_json,
)
from foundation_crack_classifier.thresholds import ThresholdConfig


def _parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="Train the foundation crack severity classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Initialize the model with pretrained backbone weights.",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Initialize the model without downloading pretrained weights.",
    )
    args = parser.parse_args(argv)
    _apply_env_defaults(args, parser)
    _validate_args(args, parser)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    try:
        records = discover_images(args.data_dir, require_all_labels=True)
        _verify_image_files(records)
        split = split_records(records, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not split.train or not split.validation:
        raise SystemExit(
            "Training requires enough images to create train and validation splits."
        )

    train_dataset = CrackImageDataset(
        split.train,
        build_train_transform(args.image_size),
    )
    validation_dataset = CrackImageDataset(
        split.validation,
        build_eval_transform(args.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = get_device()
    model = create_model(args.backbone, pretrained=args.pretrained).to(device)
    class_weights = _class_weights(split.train)
    class_weights_for_config = class_weights.tolist()
    class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    output_path = args.model_dir / "crack_severity_model.pt"
    threshold_config = ThresholdConfig().__dict__
    best_validation_loss = float("inf")
    best_model_saved = False

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_train_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_loss = _run_eval_epoch(model, validation_loader, loss_fn, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                model=model,
                output_path=output_path,
                backbone=args.backbone,
                image_size=args.image_size,
                threshold_config=threshold_config,
            )
            best_model_saved = True

    if not best_model_saved:
        raise SystemExit("Training finished without saving a checkpoint.")

    write_json(
        args.model_dir / "label_map.json",
        {label: index for index, label in enumerate(LABELS)},
    )
    write_json(
        args.model_dir / "training_config.json",
        {
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "seed": args.seed,
            "pretrained": args.pretrained,
            "train_count": len(split.train),
            "validation_count": len(split.validation),
            "test_count": len(split.test),
            "class_counts": {
                "train": _class_counts(split.train),
                "validation": _class_counts(split.validation),
                "test": _class_counts(split.test),
            },
            "class_weights": class_weights_for_config,
            "threshold_config": threshold_config,
        },
    )
    print(f"saved_model={output_path}")


def _run_train_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)

    return total_loss / max(total_count, 1)


def _run_eval_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

    return total_loss / max(total_count, 1)


def _class_weights(records) -> torch.Tensor:
    counts = torch.ones(len(LABELS), dtype=torch.float32)
    for index, label in enumerate(LABELS):
        label_count = sum(1 for record in records if record.label == label)
        counts[index] = max(label_count, 1)

    weights = counts.sum() / counts
    return weights / weights.mean()


def _class_counts(records) -> dict[str, int]:
    return {
        label: sum(1 for record in records if record.label == label)
        for label in LABELS
    }


def _verify_image_files(records) -> None:
    for record in records:
        try:
            with Image.open(record.path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Unreadable image file: {record.path}") from exc


def _apply_env_defaults(args, parser: ArgumentParser) -> None:
    args.data_dir = args.data_dir or Path(os.getenv("FCC_DATA_DIR", "training_images"))
    args.model_dir = args.model_dir or Path(os.getenv("FCC_MODEL_DIR", "models"))
    args.backbone = args.backbone or os.getenv("FCC_BACKBONE", DEFAULT_BACKBONE)
    args.epochs = _value_or_env_int(args.epochs, "FCC_EPOCHS", 5, parser)
    args.batch_size = _value_or_env_int(args.batch_size, "FCC_BATCH_SIZE", 16, parser)
    args.learning_rate = _value_or_env_float(
        args.learning_rate, "FCC_LEARNING_RATE", 0.0003, parser
    )
    args.image_size = _value_or_env_int(args.image_size, "FCC_IMAGE_SIZE", 224, parser)
    args.seed = _value_or_env_int(args.seed, "FCC_SEED", 42, parser)
    args.pretrained = _value_or_env_bool(
        args.pretrained, "FCC_PRETRAINED", True, parser
    )


def _value_or_env_int(value, env_name: str, default: int, parser: ArgumentParser) -> int:
    if value is not None:
        return value
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        parser.error(f"{env_name} must be an integer; got {raw_value!r}")


def _value_or_env_float(
    value, env_name: str, default: float, parser: ArgumentParser
) -> float:
    if value is not None:
        return value
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        parser.error(f"{env_name} must be a number; got {raw_value!r}")


def _value_or_env_bool(
    value, env_name: str, default: bool, parser: ArgumentParser
) -> bool:
    if value is not None:
        return value
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    parser.error(
        f"{env_name} must be a boolean value "
        "(true/false, 1/0, yes/no, on/off); "
        f"got {raw_value!r}"
    )


def _validate_args(args, parser: ArgumentParser) -> None:
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be a finite number > 0")
    if args.image_size < 1:
        parser.error("--image-size must be >= 1")


if __name__ == "__main__":
    main()
