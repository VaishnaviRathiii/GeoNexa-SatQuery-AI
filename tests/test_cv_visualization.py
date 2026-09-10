"""
Unit tests for Satellite Image Visualization Module.
"""

import pytest
import numpy as np
from app.computer_vision.visualization import SatelliteVisualizer
from app.computer_vision.schemas import (
    SegmentedRegion,
    DetectedFeature,
    BoundingBox,
    GroundedRegion,
)
from app.computer_vision.utils import decode_base64_to_image


@pytest.fixture
def visualizer():
    return SatelliteVisualizer()


@pytest.fixture
def sample_image():
    return np.full((128, 128, 3), 100, dtype=np.uint8)


@pytest.fixture
def sample_region():
    return SegmentedRegion(
        region_id=1,
        label="water",
        class_name="water",
        confidence=0.92,
        area_pixels=400,
        area_percentage=25.0,
        centroid=(30.0, 30.0),
        bbox=BoundingBox(ymin=10, xmin=10, ymax=50, xmax=50),
        mean_color_rgb=[20, 80, 200],
        spectral_indices={"exg": 0.0, "ndwi": 0.3},
        contour_points=[[10, 10], [50, 10], [50, 50], [10, 50]],
    )


@pytest.fixture
def sample_detection():
    return DetectedFeature(
        feature_id="det_1",
        feature_type="water_body",
        label="water",
        confidence=0.88,
        bbox=BoundingBox(ymin=10, xmin=10, ymax=50, xmax=50),
        properties={"area_pixels": 400},
    )


def test_draw_segmentation_overlay(visualizer, sample_image, sample_region):
    overlay = visualizer.draw_segmentation_overlay(sample_image, [sample_region])
    assert overlay.shape == sample_image.shape
    assert overlay.dtype == np.uint8
    # Overlay should modify the pixel values in the region
    assert not np.array_equal(overlay, sample_image)


def test_draw_bounding_boxes(visualizer, sample_image, sample_detection):
    annotated = visualizer.draw_bounding_boxes(sample_image, [sample_detection])
    assert annotated.shape == sample_image.shape
    assert not np.array_equal(annotated, sample_image)


def test_draw_query_spotlight(visualizer, sample_image, sample_region):
    grounded = GroundedRegion(
        target_query="water",
        matched_class="water",
        relevance_score=0.9,
        region=sample_region,
        explanation="Matched water body",
    )
    spotlight = visualizer.draw_query_spotlight(sample_image, [grounded], query="Find water")
    assert spotlight.ndim == 3
    assert spotlight.shape[2] == 3


def test_create_side_by_side(visualizer, sample_image):
    annotated = sample_image.copy()
    annotated[10:30, 10:30] = [255, 0, 0]
    side_by_side = visualizer.create_side_by_side(sample_image, annotated)
    assert side_by_side.ndim == 3
    # Width should be greater than original (2 images + separator)
    assert side_by_side.shape[1] > sample_image.shape[1] * 2


def test_render_full_visualization_base64(visualizer, sample_image, sample_region, sample_detection):
    img_out, b64_str = visualizer.render_full_visualization(
        sample_image,
        [sample_region],
        [sample_detection],
    )
    assert isinstance(b64_str, str)
    assert b64_str.startswith("data:image/png;base64,")

    # Verify decoding back into numpy
    decoded = decode_base64_to_image(b64_str)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == img_out.shape
