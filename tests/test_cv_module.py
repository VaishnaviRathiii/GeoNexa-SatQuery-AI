"""
Integration and Interface Contract tests for app.computer_vision.cv_module.
"""

import io
import json
import pytest
import numpy as np
from PIL import Image
from app.computer_vision.cv_module import get_cv_analysis, analyze_image_file, get_pipeline


@pytest.fixture
def test_image_bytes():
    # Create synthetic 64x64 satellite image
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:30, 10:30] = [20, 140, 20]  # Vegetation
    img[35:55, 35:55] = [20, 80, 200]  # Water
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def test_get_cv_analysis_contract(test_image_bytes):
    """
    Test that get_cv_analysis returns the exact dictionary structure
    expected by the backend and frontend.
    """
    res = get_cv_analysis(test_image_bytes)

    assert isinstance(res, dict)
    assert res["status"] == "success"

    # Verify essential top-level keys
    assert "image_metadata" in res
    assert "detected_features" in res
    assert "detections" in res
    assert "segments" in res
    assert "grounded_regions" in res
    assert "highlighted_image" in res
    assert "summary" in res
    assert "detected_regions" in res
    assert "labels" in res

    # Verify backward compatibility with mock interface
    assert isinstance(res["detected_regions"], list)
    assert isinstance(res["labels"], list)

    # Verify JSON serializability
    json_output = json.dumps(res)
    assert isinstance(json_output, str)
    assert len(json_output) > 100


def test_get_cv_analysis_with_query(test_image_bytes):
    res = get_cv_analysis(test_image_bytes, query="Find water bodies")
    assert res["status"] == "success"
    assert res["query"] == "Find water bodies"
    assert isinstance(res["grounded_regions"], list)


def test_get_cv_analysis_invalid_bytes_graceful_handling():
    """
    Test that invalid/corrupt bytes do NOT crash the server,
    but return a structured error status dictionary.
    """
    res = get_cv_analysis(b"INVALID_IMAGE_BYTES")
    assert isinstance(res, dict)
    assert res["status"] == "error"
    assert res["error"] is not None
    assert "highlighted_image" in res


def test_get_pipeline():
    pipeline = get_pipeline()
    assert pipeline is not None
