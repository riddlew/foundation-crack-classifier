from pathlib import Path

from foundation_crack_api.config import Settings


def test_settings_default_model_path():
    settings = Settings()

    assert settings.model_path == Path("/app/models/crack_severity_model.pt")


def test_settings_reads_model_path_from_environment(monkeypatch):
    monkeypatch.setenv("FCC_API_MODEL_PATH", "/tmp/model.pt")

    settings = Settings.from_env()

    assert settings.model_path == Path("/tmp/model.pt")
