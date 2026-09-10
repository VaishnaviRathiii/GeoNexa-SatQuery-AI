"""
Integration test for FastAPI backend endpoint using real CV module.
"""

from fastapi.testclient import TestClient
from pathlib import Path
import pytest
from app.backend.main import app

client = TestClient(app)
SAMPLE_IMG = Path(__file__).resolve().parent.parent / "data" / "sample_images" / "river_01.jpg"


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "GeoNexa backend is running"}


@pytest.mark.skipif(not SAMPLE_IMG.exists(), reason="Sample image not found")
def test_analyze_endpoint_with_image_and_query():
    with open(SAMPLE_IMG, "rb") as f:
        response = client.post(
            "/analyze",
            files={"image": ("river_01.jpg", f, "image/jpeg")},
            data={"question": "Find water bodies in this satellite image"},
        )

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "cv_analysis" in data
    assert "geo_info" in data

    cv = data["cv_analysis"]
    assert cv["status"] == "success"
    assert "highlighted_image" in cv
    assert len(cv["segments"]) > 0
    assert len(cv["labels"]) > 0
    assert len(cv["grounded_regions"]) > 0
