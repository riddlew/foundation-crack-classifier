# Foundation Crack Classifier

Local model training and inference for foundation crack severity triage.

## Dataset Layout

Place labeled images in:

```text
training_images/
  level1/
  level2/
  level3/
  unclear/
```

Each image should belong to exactly one class. If one photo contains mixed severity signs, use the highest visible risk label.

- **Level 1:**
  - thin hairline cracks
  - short vertical cracks
  - small shrinkage cracks
  - no visible displacement
  - no wall bowing
  - no water staining
  - isolated crack, not spanning much of the wall
  - crack looks narrow and mostly cosmetic
- **Level 2:**
  - horizontal cracks
  - stair-step cracks in block/brick
  - diagonal cracks that are long or widening
  - cracks with water staining or efflorescence
  - multiple cracks in the same wall section
  - cracks near corners, windows, doors, or openings
  - cracks spanning several feet
  - visible separation, but no major bowing/collapse
- **Level 3:**
  - wall bowing inward
  - major displacement
  - blocks shifted significantly
  - partial collapse
  - full collapse
  - large open cracks with visible separation
  - bracing/stabilization already needed or visible
  - severe cracking plus wall deformation
- **Unclear:**
  - blurry photos
  - very dark or overexposed photos
  - photos taken too close to understand context
  - photos taken too far away to see the crack
  - photos where the crack is blocked by furniture, insulation, pipes, dirt, paint, etc.
  - photos with shadows/stains that look like cracks but are ambiguous
  - non-foundation surfaces someone might upload by mistake
  - photos of walls/slabs with no visible crack
  - screenshots, cropped images, or low-resolution images
  - images where multiple issues are visible but severity is impossible to judge from the shot

## Docker Commands

Run Docker commands from the repository root.

Run tests:

```bash
docker compose run --rm classifier pytest -q
```

Train:

```bash
docker compose run --rm classifier python -m foundation_crack_classifier.train
```

Evaluate:

```bash
docker compose run --rm classifier python -m foundation_crack_classifier.evaluate
```

Infer one image (placed in the `input` folder):

```bash
docker compose run --rm classifier python -m foundation_crack_classifier.infer /app/input/photo.jpg
```

## Outputs

Training writes:

- `models/crack_severity_model.pt`
- `models/label_map.json`
- `models/training_config.json`

Evaluation writes:

- `reports/evaluation.json`
- `reports/confusion_matrix.png`

## Local API

Start the API service from the repository root:

```bash
docker compose up api
```

Health check:

```bash
curl http://localhost:8000/health
```

Classify one or more images:

```bash
curl -X POST http://localhost:8000/classify \
  -F "files=@classifier/input/level-1_1.jpg"
```

```bash
curl -X POST http://localhost:8000/classify \
  -F "files=@classifier/input/level-1_1.jpg" \
  -F "files=@classifier/input/level-2_1.jpg"
```

The API does not store uploaded images or classification results. It reads uploaded files in memory, returns a response, and discards the file bytes.
