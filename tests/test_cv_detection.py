"""
Unit tests for Satellite Feature and Object Detection Module.
"""

import pytest
import numpy as np
import cv2
from app.computer_vision.detection import SatelliteFeatureDetector
from app.computer_vision.preprocessing import SatellitePreprocessor
from app.computer_vision.schemas import BoundingBox, DetectedFeature


@pytest.fixture
def detector():
    return SatelliteFeatureDetector(min_region_area=8, nms_iou_threshold=0.45)


@pytest.fixture
def synthetic_scene():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # 1. Forest patch
    img[10:50, 10:50] = [20, 140, 20]
    # 2. Water lake
    img[60:110, 60:110] = [10, 80, 200]
    # 3. Bright industrial roof
    img[10:40, 70:110] = [220, 220, 220]
    # 4. Highway road line
    cv2.line(img, (0, 64), (127, 64), (180, 180, 180), 3)
    return img


def test_detect_linear_features(detector, synthetic_scene):
    detections = detector.detect_linear_features(synthetic_scene)
    assert isinstance(detections, list)
    # Should detect the road line
    assert len(detections) >= 1
    assert any(d.label == "highway_road" for d in detections)
    for d in detections:
        assert isinstance(d.bbox, BoundingBox)
        assert d.confidence > 0.0


def test_detect_water_bodies(detector, synthetic_scene):
    preprocessor = SatellitePreprocessor()
    indices = preprocessor.compute_spectral_indices(synthetic_scene)
    water_dets = detector.detect_water_bodies(synthetic_scene, indices)
    assert len(water_dets) >= 1
    assert any(d.label in ["sealake", "river"] for d in water_dets)


def test_detect_vegetation_parcels(detector, synthetic_scene):
    preprocessor = SatellitePreprocessor()
    indices = preprocessor.compute_spectral_indices(synthetic_scene)
    veg_dets = detector.detect_vegetation_parcels(synthetic_scene, indices)
    assert len(veg_dets) >= 1
    assert any(d.label in ["forest", "vegetation_crop", "pasture"] for d in veg_dets)


def test_detect_builtup_structures(detector, synthetic_scene):
    built_dets = detector.detect_builtup_structures(synthetic_scene)
    assert isinstance(built_dets, list)


def test_detect_all_with_nms(detector, synthetic_scene):
    all_dets = detector.detect_all(synthetic_scene)
    assert len(all_dets) > 0
    # Detections should be sorted by confidence descending
    confidences = [d.confidence for d in all_dets]
    assert confidences == sorted(confidences, reverse=True)


def test_nms_boxes():
    from app.computer_vision.utils import nms_boxes
    b1 = BoundingBox(ymin=10, xmin=10, ymax=50, xmax=50)
    b2 = BoundingBox(ymin=12, xmin=12, ymax=48, xmax=48)  # High overlap with b1
    b3 = BoundingBox(ymin=80, xmin=80, ymax=110, xmax=110) # Disjoint

    keep = nms_boxes([b1, b2, b3], [0.9, 0.7, 0.8], iou_threshold=0.5)
    assert 0 in keep  # Highest confidence b1 kept
    assert 1 not in keep # b2 suppressed
    assert 2 in keep  # b3 kept
