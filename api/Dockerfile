FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install --upgrade pip \
  && pip install -e ".[dev]"

ENV PYTHONPATH=/app/src:/app/classifier_src
ENV FCC_API_MODEL_PATH=/app/models/crack_severity_model.pt

CMD ["uvicorn", "foundation_crack_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
