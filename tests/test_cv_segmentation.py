"""
Unit tests for Satellite Image Segmentation Module.
"""

import pytest
import numpy as np
from app.computer_vision.segmentation import SatelliteSegmenter
from app.computer_vision.preprocessing import SatellitePreprocessor


@pytest.fixture
def segmenter():
    return SatelliteSegmenter(default_k=4, slic_n_segments=16)


@pytest.fixture
def synthetic_multizone_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Zone 1: Water (top-left)
    img[0:50, 0:50] = [15, 60, 180]
    # Zone 2: Forest (top-right)
    img[0:50, 50:100] = [25, 130, 25]
    # Zone 3: Urban/Industrial (bottom-left)
    img[50:100, 0:50] = [210, 210, 210]
    # Zone 4: Soil/Cropland (bottom-right)
    img[50:100, 50:100] = [170, 120, 60]
    return img


def test_segment_kmeans(segmenter, synthetic_multizone_image):
    labels, centers = segmenter.segment_kmeans(synthetic_multizone_image, k=4)
    assert labels.shape == (100, 100)
    assert len(np.unique(labels)) <= 4
    assert centers.shape[0] == 4


def test_segment_slic(segmenter, synthetic_multizone_image):
    labels = segmenter.segment_slic(synthetic_multizone_image, n_segments=16)
    assert labels.shape == (100, 100)
    assert len(np.unique(labels)) > 1


def test_segment_semantic_rules(segmenter, synthetic_multizone_image):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_multizone_image)
    masks = segmenter.segment_semantic_rules(synthetic_multizone_image, indices)

    assert "water" in masks
    assert "forest" in masks
    assert "vegetation_crop" in masks
    assert "industrial" in masks
    assert "bare_soil" in masks

    # Verify water mask is active in top-left
    assert np.any(masks["water"][0:50, 0:50] > 0)
    # Verify forest mask is active in top-right
    assert np.any(masks["forest"][0:50, 50:100] > 0)


def test_create_unified_landcover_map(segmenter, synthetic_multizone_image):
    pre = SatellitePreprocessor()
    indices = pre.compute_spectral_indices(synthetic_multizone_image)
    unified_map, class_dict = segmenter.create_unified_landcover_map(synthetic_multizone_image, indices)

    assert unified_map.shape == (100, 100)
    assert len(class_dict) > 0
    assert not np.any(unified_map == -1)  # All pixels assigned
