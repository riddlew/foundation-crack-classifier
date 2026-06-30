from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from foundation_crack_api.classifier_service import ClassifierService, ImageDecodeError
from foundation_crack_api.config import Settings
from foundation_crack_api.schemas import (
    ClassifyResponse,
    FileClassificationResponse,
    HealthResponse,
)


class SupportsClassification(Protocol):
    def classify_bytes(self, image_bytes: bytes) -> dict[str, object]:
        ...


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_classifier_service() -> ClassifierService:
    settings = get_settings()
    return ClassifierService(settings.model_path)


app = FastAPI(title="Foundation Crack Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    files: Annotated[list[UploadFile], File()],
    classifier: Annotated[SupportsClassification, Depends(get_classifier_service)],
    notes: Annotated[str | None, Form()] = None,
) -> ClassifyResponse:
    # classify_bytes is synchronous and CPU-bound. For a local single-user
    # service this is fine. For concurrent use, move to run_in_threadpool.
    results: list[FileClassificationResponse] = []

    for upload in files:
        try:
            image_bytes = await upload.read()
            result = classifier.classify_bytes(image_bytes)
            results.append(
                FileClassificationResponse(
                    filename=upload.filename or "uploaded-image",
                    ok=True,
                    result=result,
                    error=None,
                )
            )
        except ImageDecodeError as exc:
            results.append(
                FileClassificationResponse(
                    filename=upload.filename or "uploaded-image",
                    ok=False,
                    result=None,
                    error=str(exc),
                )
            )
        finally:
            await upload.close()

    return ClassifyResponse(results=results)
