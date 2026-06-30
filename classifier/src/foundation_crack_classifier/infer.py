from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import pickle

from PIL import Image
import torch

from foundation_crack_classifier.dataset import build_eval_transform
from foundation_crack_classifier.labels import LABELS
from foundation_crack_classifier.model import get_device, load_checkpoint
from foundation_crack_classifier.thresholds import ThresholdConfig, apply_thresholds


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="Run single-image foundation crack severity inference."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("FCC_MODEL_DIR", "models")) / "crack_severity_model.pt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.image_path.is_file():
        raise SystemExit(f"Image file not found: {args.image_path}")
    if not args.model_path.is_file():
        raise SystemExit(f"Model file not found: {args.model_path}")

    device = get_device()
    try:
        model, checkpoint = load_checkpoint(args.model_path, map_location=device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise SystemExit(f"Unable to load model checkpoint: {args.model_path}") from exc
    model.to(device)

    probabilities = predict_probabilities(
        model,
        args.image_path,
        build_eval_transform(checkpoint["image_size"]),
        device,
    )
    try:
        threshold_config = ThresholdConfig(**checkpoint.get("threshold_config", {}))
        result = apply_thresholds(probabilities, threshold_config)
    except TypeError as exc:
        raise SystemExit(
            f"Invalid checkpoint threshold configuration: {args.model_path}"
        ) from exc
    except ValueError as exc:
        raise SystemExit(f"Invalid model probabilities: {exc}") from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def predict_probabilities_from_image(model, image: Image.Image, transform, device) -> dict[str, float]:
    model.eval()
    tensor = transform(image.convert("RGB"))

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {
        label: float(probabilities[index])
        for index, label in enumerate(LABELS)
    }


def predict_probabilities(model, image_path: Path, transform, device) -> dict[str, float]:
    try:
        with Image.open(image_path) as image:
            return predict_probabilities_from_image(model, image, transform, device)
    except OSError as exc:
        raise SystemExit(f"Unable to read image file: {image_path}") from exc


if __name__ == "__main__":
    main()
