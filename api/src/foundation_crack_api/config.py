from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("/app/models/crack_severity_model.pt")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=Path(
                os.getenv("FCC_API_MODEL_PATH", "/app/models/crack_severity_model.pt")
            )
        )
