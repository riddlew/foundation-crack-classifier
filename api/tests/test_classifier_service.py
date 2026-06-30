from io import BytesIO

from PIL import Image
import pytest

from foundation_crack_api.classifier_service import ImageDecodeError, decode_image


def make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (12, 12), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_returns_rgb_pil_image():
    image = decode_image(make_png_bytes())

    assert image.mode == "RGB"
    assert image.size == (12, 12)


def test_decode_image_rejects_empty_bytes():
    with pytest.raises(ImageDecodeError, match=r"Uploaded file was empty\."):
        decode_image(b"")


def test_decode_image_rejects_invalid_image_bytes():
    with pytest.raises(ImageDecodeError, match=r"Unable to read image file\."):
        decode_image(b"not an image")
