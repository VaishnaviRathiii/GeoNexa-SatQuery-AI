"""
Unit tests for Region Identification and Query Grounding Module.
"""

import pytest
import numpy as np
from app.computer_vision.region_identifier import RegionIdentifier
from app.computer_vision.segmentation import SatelliteSegmenter
from app.computer_vision.preprocessing import SatellitePreprocessor
from app.computer_vision.schemas import SegmentedRegion, GroundedRegion


@pytest.fixture
def identifier():
    return RegionIdentifier(min_region_area=10)


@pytest.fixture
def synthetic_scene():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:40, 10:40] = [15, 60, 180]   # Water
    img[50:90, 50:90] = [25, 140, 25]   # Forest
    return img


def test_extract_regions_from_map(identifier, synthetic_scene):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_scene)
    seg = SatelliteSegmenter()
    unified_map, class_dict = seg.create_unified_landcover_map(synthetic_scene, indices)

    regions = identifier.extract_regions_from_map(synthetic_scene, unified_map, class_dict, indices)
    assert isinstance(regions, list)
    assert len(regions) > 0

    for r in regions:
        assert isinstance(r, SegmentedRegion)
        assert r.area_pixels >= 10
        assert 0.0 <= r.area_percentage <= 100.0
        assert 0.0 <= r.confidence <= 1.0
        assert len(r.centroid) == 2
        assert len(r.mean_color_rgb) == 3
        assert "exg" in r.spectral_indices


def test_query_grounding_water(identifier, synthetic_scene):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_scene)
    seg = SatelliteSegmenter()
    unified_map, class_dict = seg.create_unified_landcover_map(synthetic_scene, indices)
    regions = identifier.extract_regions_from_map(synthetic_scene, unified_map, class_dict, indices)

    grounded = identifier.ground_query("Find water bodies and lakes", regions)
    assert isinstance(grounded, list)
    assert len(grounded) >= 1
    assert any(g.matched_class == "water" for g in grounded)
    assert grounded[0].relevance_score > 0.4
    assert "Grounded" in grounded[0].explanation


def test_query_grounding_forest(identifier, synthetic_scene):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_scene)
    seg = SatelliteSegmenter()
    unified_map, class_dict = seg.create_unified_landcover_map(synthetic_scene, indices)
    regions = identifier.extract_regions_from_map(synthetic_scene, unified_map, class_dict, indices)

    grounded = identifier.ground_query("Show forests and woodland", regions)
    assert len(grounded) >= 1
    assert any("forest" in g.matched_class for g in grounded)


def test_query_grounding_empty_query(identifier, synthetic_scene):
    grounded = identifier.ground_query("", [])
    assert grounded == []


def test_generate_summary(identifier, synthetic_scene):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_scene)
    seg = SatelliteSegmenter()
    unified_map, class_dict = seg.create_unified_landcover_map(synthetic_scene, indices)
    regions = identifier.extract_regions_from_map(synthetic_scene, unified_map, class_dict, indices)

    summary = identifier.generate_summary(regions, [], query="Find water")
    assert isinstance(summary, str)
    assert len(summary) > 10
    assert "Scene breakdown" in summary
