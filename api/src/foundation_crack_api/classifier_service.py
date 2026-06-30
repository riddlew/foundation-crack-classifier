from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pickle

from PIL import Image, UnidentifiedImageError
import torch

from foundation_crack_classifier.dataset import build_eval_transform
from foundation_crack_classifier.infer import predict_probabilities_from_image
from foundation_crack_classifier.model import get_device, load_checkpoint
from foundation_crack_classifier.thresholds import ThresholdConfig, apply_thresholds


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ImageDecodeError("Uploaded file was empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageDecodeError("Unable to read image file.") from exc


class ClassifierService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.device = get_device()
        try:
            self.model, self.checkpoint = load_checkpoint(model_path, map_location=self.device)
        except ValueError:
            raise
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise RuntimeError(f"Unable to load model checkpoint: {model_path}") from exc

        self.model.to(self.device)
        self.transform = build_eval_transform(self.checkpoint["image_size"])
        self.threshold_config = ThresholdConfig(
            **self.checkpoint.get("threshold_config", {})
        )

    def classify_bytes(self, image_bytes: bytes) -> dict[str, object]:
        image = decode_image(image_bytes)
        probabilities = predict_probabilities_from_image(
            self.model,
            image,
            self.transform,
            self.device,
        )
        result = apply_thresholds(probabilities, self.threshold_config)
        return result.to_dict()
