# Foundation Crack Classifier API

FastAPI service that classifies foundation crack photos and returns a severity assessment per image. Images are processed in memory — nothing is stored.

## Prerequisites

- Docker and Docker Compose
- A trained model checkpoint at `classifier/models/crack_severity_model.pt`

  Run training first if the model does not exist:

  ```bash
  docker compose run --rm classifier python -m foundation_crack_classifier.train
  ```

## Quick Start

Run from the **repository root** (not from `api/`):

```bash
docker compose up -d api
```

The service is ready when `docker compose logs api` shows:

```
Application startup complete.
```

Stop the service:

```bash
docker compose stop api
```

---

## Endpoints

### `GET /health`

Liveness check. Returns `200 OK` immediately — does not verify the model is loadable.

**Response**

```json
{"status": "ok"}
```

**curl**

```bash
curl http://localhost:8000/health
```

**HTTPie**

```bash
http GET http://localhost:8000/health
```

---

### `POST /classify`

Classify one or more crack photos. Accepts multipart form data. Returns one result per uploaded file.

**Request**

| Field   | Type                     | Required | Description                                      |
|---------|--------------------------|----------|--------------------------------------------------|
| `files` | one or more image files  | yes      | JPEG, PNG, or WebP. Sent as multipart form data. |

The model loads lazily on the first request — the first call may take a few seconds longer than subsequent ones.

**Response**

```json
{
  "results": [
    {
      "filename": "photo.jpg",
      "ok": true,
      "result": {
        "severity_level": "Level 1",
        "urgency": "inspection_recommended",
        "final_label": "level1",
        "confidence": 0.993209,
        "raw_probabilities": {
          "level1": 0.993209,
          "level2": 0.006465,
          "level3": 0.000034,
          "unclear": 0.000293
        },
        "why_this_result": "The model found the strongest match with Level 1.",
        "customer_summary": "The photo appears most consistent with a lower-severity crack...",
        "disclaimer": "This AI result is based on a single photo and is not a final structural diagnosis...",
        "recommended_action": "Consider scheduling an inspection."
      },
      "error": null
    }
  ]
}
```

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Name of the uploaded file |
| `ok` | bool | `true` if classification succeeded, `false` if the file could not be decoded |
| `result` | object \| null | Classification result (null when `ok` is false) |
| `error` | string \| null | Error message (null when `ok` is true) |

**`result` fields**

| Field | Type | Description |
|-------|------|-------------|
| `severity_level` | string | Human-readable severity: `"Level 1"`, `"Level 2"`, `"Level 3"`, or `"Unclear"` |
| `urgency` | string | Machine-readable urgency: `inspection_recommended`, `contact_soon`, `contact_immediately`, `unable_to_assess` |
| `final_label` | string | Raw label: `level1`, `level2`, `level3`, `unclear` |
| `confidence` | float [0–1] | Model confidence for the assigned label |
| `raw_probabilities` | object | Softmax probability for each of the four labels |
| `why_this_result` | string | One-sentence explanation of the threshold logic that produced this result |
| `customer_summary` | string | Plain-language summary suitable for showing to a homeowner |
| `disclaimer` | string | Standard disclaimer (not a structural diagnosis) |
| `recommended_action` | string | Recommended next step for the homeowner |

**Error response (per file)**

When a file cannot be decoded as an image, that file's entry has `ok: false` and a non-null `error`. The rest of the batch still returns results. The HTTP status is always `200`.

```json
{
  "filename": "document.pdf",
  "ok": false,
  "result": null,
  "error": "Unable to read image file."
}
```

A `422` response is returned if the `files` field is missing entirely from the request.

---

## Examples

### Single image

**curl**

```bash
curl -X POST http://localhost:8000/classify \
  -F "files=@/path/to/photo.jpg"
```

**HTTPie**

```bash
http --form POST http://localhost:8000/classify \
  files@/path/to/photo.jpg
```

---

### Multiple images in one request

**curl**

```bash
curl -X POST http://localhost:8000/classify \
  -F "files=@/path/to/photo-1.jpg" \
  -F "files=@/path/to/photo-2.jpg" \
  -F "files=@/path/to/photo-3.jpg"
```

**HTTPie**

```bash
http --form POST http://localhost:8000/classify \
  files@/path/to/photo-1.jpg \
  files@/path/to/photo-2.jpg \
  files@/path/to/photo-3.jpg
```

---

### Pretty-print the response

**curl + jq**

```bash
curl -s -X POST http://localhost:8000/classify \
  -F "files=@/path/to/photo.jpg" | jq .
```

**HTTPie** (pretty-prints by default)

```bash
http --form POST http://localhost:8000/classify \
  files@/path/to/photo.jpg
```

---

## Running Tests

```bash
docker compose run --rm api pytest -q
```

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `FCC_API_MODEL_PATH` | `/app/models/crack_severity_model.pt` | Path to the model checkpoint inside the container |
