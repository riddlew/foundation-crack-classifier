def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_accepts_one_image(client):
    response = client.post(
        "/classify",
        files=[("files", ("photo.jpg", b"fake image bytes", "image/jpeg"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["results"][0]["filename"] == "photo.jpg"
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["result"]["final_label"] == "level2"
    assert payload["results"][0]["error"] is None


def test_classify_accepts_multiple_images(client):
    response = client.post(
        "/classify",
        files=[
            ("files", ("photo-1.jpg", b"image one", "image/jpeg")),
            ("files", ("photo-2.jpg", b"image two", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["filename"] for item in payload["results"]] == [
        "photo-1.jpg",
        "photo-2.jpg",
    ]
    assert all(item["ok"] for item in payload["results"])


def test_classify_returns_per_file_error(client):
    response = client.post(
        "/classify",
        files=[
            ("files", ("good.jpg", b"image one", "image/jpeg")),
            ("files", ("bad.txt", b"bad image", "text/plain")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["ok"] is True
    assert payload["results"][1] == {
        "filename": "bad.txt",
        "ok": False,
        "result": None,
        "error": "Unable to read image file.",
    }


def test_classify_requires_files_field(client):
    response = client.post("/classify")

    assert response.status_code == 422


def test_classify_accepts_optional_notes(client):
    response = client.post(
        "/classify",
        files=[("files", ("photo.jpg", b"fake image bytes", "image/jpeg"))],
        data={"notes": "image #1 shows the east wall"},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["ok"] is True


def test_cors_header_present_on_response(client):
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:8001"},
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8001"


def test_cors_preflight(client):
    response = client.options(
        "/classify",
        headers={
            "Origin": "http://localhost:8001",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8001"
