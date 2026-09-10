"""
Unit tests for Satellite Preprocessing Module.
"""

import io
import pytest
import numpy as np
from PIL import Image
from app.computer_vision.preprocessing import SatellitePreprocessor


@pytest.fixture
def preprocessor():
    return SatellitePreprocessor(target_size=(128, 128))


@pytest.fixture
def sample_rgb_array():
    # 64x64 synthetic satellite RGB image
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:30, 10:30] = [34, 139, 34]    # Green patch (forest)
    arr[35:55, 35:55] = [30, 144, 255]   # Blue patch (water)
    arr[40:60, 10:25] = [200, 200, 200]  # Bright patch (concrete)
    return arr


@pytest.fixture
def sample_image_bytes(sample_rgb_array):
    pil_img = Image.fromarray(sample_rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def test_decode_valid_bytes(preprocessor, sample_image_bytes):
    decoded = preprocessor.decode_image(sample_image_bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (64, 64, 3)
    assert decoded.dtype == np.uint8


def test_decode_pil_image(preprocessor, sample_rgb_array):
    pil_img = Image.fromarray(sample_rgb_array)
    decoded = preprocessor.decode_image(pil_img)
    assert decoded.shape == (64, 64, 3)


def test_decode_numpy_array(preprocessor, sample_rgb_array):
    decoded = preprocessor.decode_image(sample_rgb_array)
    assert np.array_equal(decoded, sample_rgb_array)


def test_decode_grayscale_numpy(preprocessor):
    gray = np.full((32, 32), 128, dtype=np.uint8)
    decoded = preprocessor.decode_image(gray)
    assert decoded.shape == (32, 32, 3)


def test_decode_empty_bytes_raises_error(preprocessor):
    with pytest.raises(ValueError, match="empty"):
        preprocessor.decode_image(b"")


def test_decode_corrupt_bytes_raises_error(preprocessor):
    with pytest.raises(ValueError, match="Failed to decode"):
        preprocessor.decode_image(b"NOT_A_VALID_IMAGE_BYTES_BUFFER")


def test_decode_invalid_type_raises_error(preprocessor):
    with pytest.raises(TypeError):
        preprocessor.decode_image(12345)


def test_validate_image_too_small(preprocessor):
    small = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocessor.validate_image(small)


def test_resize_standard(preprocessor, sample_rgb_array):
    resized, meta = preprocessor.resize_image(sample_rgb_array, target_size=(128, 128))
    assert resized.shape == (128, 128, 3)
    assert meta["target_size"] == (128, 128)
    assert meta["original_size"] == (64, 64)


def test_resize_keep_aspect_ratio(preprocessor):
    rect_img = np.zeros((64, 128, 3), dtype=np.uint8)
    resized, meta = preprocessor.resize_image(rect_img, target_size=(128, 128), keep_aspect_ratio=True)
    assert resized.shape == (128, 128, 3)
    assert meta["scale"] == 1.0
    assert meta["pad_top"] == 32


def test_normalization_minmax(preprocessor, sample_rgb_array):
    norm = preprocessor.normalize_image(sample_rgb_array, method="minmax")
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_normalization_standard(preprocessor, sample_rgb_array):
    norm = preprocessor.normalize_image(sample_rgb_array, method="standard")
    assert norm.dtype == np.float32
    assert abs(norm.mean()) < 0.1


def test_normalization_imagenet(preprocessor, sample_rgb_array):
    norm = preprocessor.normalize_image(sample_rgb_array, method="imagenet")
    assert norm.dtype == np.float32
    assert norm.shape == sample_rgb_array.shape


def test_normalization_sentinel2_approx(preprocessor, sample_rgb_array):
    norm = preprocessor.normalize_image(sample_rgb_array, method="sentinel2_approx")
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_enhance_contrast_clahe(preprocessor, sample_rgb_array):
    enhanced = preprocessor.enhance_contrast(sample_rgb_array)
    assert enhanced.shape == sample_rgb_array.shape
    assert enhanced.dtype == np.uint8


def test_compute_spectral_indices(preprocessor, sample_rgb_array):
    indices = preprocessor.compute_spectral_indices(sample_rgb_array)
    assert "exg" in indices
    assert "vari" in indices
    assert "gli" in indices
    assert "ndwi_rgb" in indices
    assert "ndti" in indices
    assert "brightness" in indices
    assert "texture" in indices

    # Check shapes
    h, w = sample_rgb_array.shape[:2]
    for k, v in indices.items():
        assert v.shape == (h, w)


def test_extract_metadata(preprocessor, sample_rgb_array):
    meta = preprocessor.extract_metadata(sample_rgb_array)
    assert meta.width == 64
    assert meta.height == 64
    assert meta.channels == 3
    assert len(meta.mean_rgb) == 3
    assert len(meta.std_rgb) == 3
    assert isinstance(meta.estimated_scene_type, str)


def test_preprocess_end_to_end(preprocessor, sample_image_bytes):
    processed, meta, indices = preprocessor.preprocess(
        sample_image_bytes,
        resize=True,
        target_size=(100, 100),
        enhance=True,
    )
    assert processed.shape == (100, 100, 3)
    assert meta.width == 64
    assert meta.height == 64
    assert "exg" in indices
