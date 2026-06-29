from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from foundation_crack_classifier.dataset import (
    CrackImageDataset,
    build_eval_transform,
    discover_images,
    split_records,
)
from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.model import get_device, load_checkpoint, write_json


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="Evaluate the foundation crack severity classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("FCC_DATA_DIR", "training_images")),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("FCC_MODEL_DIR", "models")) / "crack_severity_model.pt",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(os.getenv("FCC_REPORT_DIR", "reports")),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model_path.exists():
        raise SystemExit(f"Model file not found: {args.model_path}")

    device = get_device()
    try:
        model, checkpoint = load_checkpoint(args.model_path, map_location=device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model.to(device)

    try:
        records = discover_images(args.data_dir, require_all_labels=True)
        split = split_records(records, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not split.test:
        raise SystemExit("Evaluation requires a non-empty test split.")

    dataset = CrackImageDataset(
        split.test,
        build_eval_transform(checkpoint["image_size"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred = _predict(model, loader, device)
    report = _build_report(y_true, y_pred)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "evaluation.json", report)
    _write_confusion_matrix(y_true, y_pred, args.report_dir / "confusion_matrix.png")
    print(f"evaluation_report={args.report_dir / 'evaluation.json'}")


def _predict(model, loader, device) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return y_true, y_pred


def _build_report(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    level1 = LABELS.index("level1")
    level2 = LABELS.index("level2")
    level3 = LABELS.index("level3")
    label_indexes = list(range(len(LABELS)))
    matrix = confusion_matrix(y_true, y_pred, labels=label_indexes)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_indexes,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "high_risk_misses": {
            "actual_level3_predicted_level1": int(matrix[level3][level1]),
            "actual_level2_predicted_level1": int(matrix[level2][level1]),
        },
    }


def _write_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _validate_args(args, parser: ArgumentParser) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")


if __name__ == "__main__":
    main()
