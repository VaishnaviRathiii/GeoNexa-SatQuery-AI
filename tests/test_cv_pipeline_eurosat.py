"""
End-to-end evaluation and tests of CV pipeline against real EuroSAT satellite images.
"""

from pathlib import Path
import json
import pytest
from app.computer_vision.cv_module import get_cv_analysis

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_images"

EUROSAT_CLASSES = [
    "annualcrop",
    "forest",
    "herbaceousvegetation",
    "highway",
    "industrial",
    "pasture",
    "permanentcrop",
    "residential",
    "river",
    "sealake",
]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Sample images directory not found")
def test_all_eurosat_sample_images():
    """
    Test that every sample satellite image in the EuroSAT dataset
    can be processed cleanly by the CV pipeline.
    """
    sample_files = list(DATA_DIR.glob("*.jpg"))
    assert len(sample_files) >= 10, f"Expected at least 10 sample images, found {len(sample_files)}"

    for img_path in sample_files:
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        res = get_cv_analysis(img_bytes)

        # 1. Pipeline status
        assert res["status"] == "success", f"Failed on image {img_path.name}: {res.get('error')}"

        # 2. Metadata validation
        meta = res["image_metadata"]
        assert meta["width"] == 64
        assert meta["height"] == 64
        assert meta["channels"] == 3

        # 3. Segments and detections present
        assert len(res["segments"]) > 0
        assert len(res["labels"]) > 0

        # 4. Highlighted output valid base64 PNG
        assert res["highlighted_image"].startswith("data:image/png;base64,")

        # 5. JSON serializability
        serialized = json.dumps(res)
        assert len(serialized) > 0


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Sample images directory not found")
def test_query_grounding_on_eurosat_water():
    river_img = DATA_DIR / "river_01.jpg"
    if not river_img.exists():
        pytest.skip("river_01.jpg not found")

    with open(river_img, "rb") as f:
        img_bytes = f.read()

    res = get_cv_analysis(img_bytes, query="Find rivers and water")
    assert res["status"] == "success"
    assert len(res["grounded_regions"]) > 0
    assert any("water" in g["matched_class"] for g in res["grounded_regions"])


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Sample images directory not found")
def test_query_grounding_on_eurosat_residential():
    res_img = DATA_DIR / "residential_01.jpg"
    if not res_img.exists():
        pytest.skip("residential_01.jpg not found")

    with open(res_img, "rb") as f:
        img_bytes = f.read()

    res = get_cv_analysis(img_bytes, query="Show residential houses and living areas")
    assert res["status"] == "success"
    assert len(res["grounded_regions"]) > 0
    assert any("residential" in g["matched_class"] for g in res["grounded_regions"])


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Sample images directory not found")
def test_query_grounding_on_eurosat_forest():
    forest_img = DATA_DIR / "forest_01.jpg"
    if not forest_img.exists():
        pytest.skip("forest_01.jpg not found")

    with open(forest_img, "rb") as f:
        img_bytes = f.read()

    res = get_cv_analysis(img_bytes, query="Locate forests and trees")
    assert res["status"] == "success"
    assert len(res["grounded_regions"]) > 0
    assert any("forest" in g["matched_class"] for g in res["grounded_regions"])
