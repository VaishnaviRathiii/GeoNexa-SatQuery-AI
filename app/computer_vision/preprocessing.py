"""
Satellite Image Preprocessing Module.

Handles decoding, validation, color conversions, resizing, normalization,
contrast enhancement, and spectral index extraction for remote sensing imagery.
"""

import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Union, Optional
from app.computer_vision.schemas import ImageMetadata


class SatellitePreprocessor:
    """
    Production-ready satellite image preprocessor.
    Ensures safe byte decoding, spatial consistency, color accuracy,
    and extraction of remote sensing indices.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        default_normalization: str = "minmax",
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
    ):
        self.target_size = target_size
        self.default_normalization = default_normalization
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit

    def decode_image(self, image_input: Union[bytes, str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Safely decode input into an RGB uint8 numpy array of shape (H, W, 3).

        Supports:
            - Raw bytes (PNG, JPEG, GeoTIFF, WebP, etc.)
            - File path string
            - PIL Image object
            - Numpy array (RGB, BGR, or Grayscale)
        """
        if image_input is None:
            raise ValueError("Image input cannot be None.")

        # 1. If input is raw bytes
        if isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise ValueError("Image bytes buffer is empty (0 bytes).")
            try:
                # Use PIL first as it safely handles a wide variety of formats
                pil_img = Image.open(io.BytesIO(image_input))
                pil_img = pil_img.convert("RGB")
                img_array = np.array(pil_img, dtype=np.uint8)
            except Exception as pil_err:
                # Fallback to OpenCV
                np_buf = np.frombuffer(image_input, dtype=np.uint8)
                bgr_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if bgr_img is None:
                    raise ValueError(f"Failed to decode image bytes: {pil_err}")
                img_array = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 2. If input is file path string
        elif isinstance(image_input, str):
            try:
                pil_img = Image.open(image_input).convert("RGB")
                img_array = np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Could not open image file '{image_input}': {e}")

        # 3. If input is PIL Image
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_array = np.array(pil_img, dtype=np.uint8)

        # 4. If input is numpy array
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Numpy image array is empty.")

            # Handle float arrays [0, 1]
            if np.issubdtype(image_input.dtype, np.floating):
                if image_input.max() <= 1.0:
                    img_array = (image_input * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_array = image_input.clip(0, 255).astype(np.uint8)
            else:
                img_array = image_input.astype(np.uint8)

            # Handle grayscale or RGBA
            if img_array.ndim == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.ndim == 3:
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif img_array.shape[2] == 1:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] != 3:
                    raise ValueError(f"Unsupported channel dimension: {img_array.shape[2]}")
            else:
                raise ValueError(f"Unsupported array dimension: {img_array.ndim}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        self.validate_image(img_array)
        return img_array

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that the image is a valid 3-channel RGB image with non-zero dimensions.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        height, width, _ = image.shape
        if height < 4 or width < 4:
            raise ValueError(f"Image dimensions too small ({width}x{height}). Minimum is 4x4.")

        return True

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        keep_aspect_ratio: bool = False,
        padding_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize image to target size using high-quality interpolation.

        Args:
            image: Input RGB image.
            target_size: (width, height) tuple. Defaults to self.target_size.
            keep_aspect_ratio: If True, scales uniformly and applies symmetric padding.
            padding_color: Background color for padded borders.

        Returns:
            Tuple of (resized_image, transform_metadata_dict).
        """
        target_size = target_size or self.target_size
        target_w, target_h = target_size
        orig_h, orig_w = image.shape[:2]

        if orig_w == target_w and orig_h == target_h:
            return image.copy(), {
                "original_size": (orig_w, orig_h),
                "target_size": (target_w, target_h),
                "scale": 1.0,
                "pad_top": 0,
                "pad_left": 0,
            }

        # Choose optimal interpolation method
        if target_w < orig_w and target_h < orig_h:
            interpolation = cv2.INTER_AREA  # Best for downscaling satellite images
        else:
            interpolation = cv2.INTER_CUBIC  # High-fidelity upscaling

        if not keep_aspect_ratio:
            resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
            return resized, {
                "original_size": (orig_w, orig_h),
                "target_size": (target_w, target_h),
                "scale_x": target_w / orig_w,
                "scale_y": target_h / orig_h,
                "pad_top": 0,
                "pad_left": 0,
            }

        # Uniform scale with letterboxing/padding
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = scaled_img

        return canvas, {
            "original_size": (orig_w, orig_h),
            "target_size": (target_w, target_h),
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
        }

    def normalize_image(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalize pixel values for downstream neural network or classical feature extractors.

        Methods:
            - 'minmax': [0.0, 1.0] float32
            - 'standard': Zero-mean, unit-variance float32
            - 'imagenet': ImageNet mean/std normalization
            - 'sentinel2_approx': Remote sensing surface reflectance scaled float32
        """
        method = method or self.default_normalization
        img_float = image.astype(np.float32)

        if method == "minmax":
            return img_float / 255.0

        elif method == "standard":
            mean = np.mean(img_float, axis=(0, 1), keepdims=True)
            std = np.std(img_float, axis=(0, 1), keepdims=True) + 1e-7
            return (img_float - mean) / std

        elif method == "imagenet":
            # ImageNet standard: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            scaled = img_float / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return (scaled - mean) / std

        elif method == "sentinel2_approx":
            # Sentinel-2 RGB approximate BOA (Bottom-of-Atmosphere) reflectance [0.0, 1.0]
            # Clips atmospheric haze and scales dynamic range
            p2 = np.percentile(img_float, 2, axis=(0, 1), keepdims=True)
            p98 = np.percentile(img_float, 98, axis=(0, 1), keepdims=True)
            clipped = np.clip((img_float - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)
            return clipped

        else:
            raise ValueError(f"Unknown normalization method: '{method}'")

    def enhance_contrast(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
        in LAB color space. Enhances local terrain contrast while preserving colors.
        """
        clip_limit = clip_limit or self.clahe_clip_limit
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l)

        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    def compute_spectral_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute remote sensing visual spectral indices from RGB channels:
            - ExG (Excess Green Index): 2*G - R - B (Vegetation indicator)
            - VARI (Visible Atmospherically Resistant Index): (G - R) / (G + R - B)
            - GLI (Green Leaf Index): (2*G - R - B) / (2*G + R + B)
            - NDWI_RGB (Visible Water Index): Water surface detection
            - NDTI (Normalized Difference Tillage/Soil Index): (R - G) / (R + G)
            - Brightness: V channel in HSV
            - Texture: Local standard deviation (Built-up / structural density)
        """
        img_f = image.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        eps = 1e-7

        # 1. Excess Green Index (ExG)
        exg = (2.0 * g - r - b)

        # 2. Visible Atmospherically Resistant Index (VARI)
        vari = (g - r) / (g + r - b + eps)
        vari = np.clip(vari, -1.0, 1.0)

        # 3. Green Leaf Index (GLI)
        gli = (2.0 * g - r - b) / (2.0 * g + r + b + eps)
        gli = np.clip(gli, -1.0, 1.0)

        # 4. Visible Water Index (NDWI_RGB)
        # Water reflects more in blue/green and absorbs strongly in red
        water_ratio = (b - r) / (b + r + eps)
        green_red_ratio = (g - r) / (g + r + eps)
        ndwi_rgb = (water_ratio + green_red_ratio) / 2.0
        ndwi_rgb = np.clip(ndwi_rgb, -1.0, 1.0)

        # 5. Normalized Difference Tillage / Soil Index (NDTI)
        ndti = (r - g) / (r + g + eps)
        ndti = np.clip(ndti, -1.0, 1.0)

        # 6. Brightness (Value channel)
        brightness = np.max(img_f, axis=2)

        # 7. Texture / Local Standard Deviation (Built-up indicator)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mean_filter = cv2.blur(gray, (5, 5))
        sq_mean_filter = cv2.blur(gray * gray, (5, 5))
        texture_var = np.sqrt(np.maximum(0.0, sq_mean_filter - mean_filter * mean_filter))

        return {
            "exg": exg,
            "vari": vari,
            "gli": gli,
            "ndwi_rgb": ndwi_rgb,
            "ndti": ndti,
            "brightness": brightness,
            "texture": texture_var,
        }

    def extract_metadata(self, image: np.ndarray) -> ImageMetadata:
        """
        Extract basic statistical metadata and estimated scene type from an RGB image.
        """
        h, w, c = image.shape
        mean_rgb = np.mean(image, axis=(0, 1)).tolist()
        std_rgb = np.std(image, axis=(0, 1)).tolist()

        indices = self.compute_spectral_indices(image)
        mean_exg = float(np.mean(indices["exg"]))
        mean_ndwi = float(np.mean(indices["ndwi_rgb"]))
        mean_ndti = float(np.mean(indices["ndti"]))
        mean_texture = float(np.mean(indices["texture"]))
        mean_brightness = float(np.mean(indices["brightness"]))

        # Heuristic initial scene type estimation
        if mean_ndwi > 0.15 and mean_brightness < 0.45:
            estimated_scene = "water_body"
        elif mean_exg > 0.08:
            estimated_scene = "dense_vegetation_or_forest"
        elif mean_texture > 0.12 and mean_brightness > 0.35:
            estimated_scene = "urban_or_built_up"
        elif mean_ndti > 0.05:
            estimated_scene = "agricultural_or_bare_soil"
        else:
            estimated_scene = "mixed_terrain"

        return ImageMetadata(
            width=w,
            height=h,
            channels=c,
            format="RGB",
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            estimated_scene_type=estimated_scene,
        )

    def preprocess(
        self,
        image_input: Union[bytes, str, Image.Image, np.ndarray],
        resize: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        enhance: bool = False,
    ) -> Tuple[np.ndarray, ImageMetadata, Dict[str, np.ndarray]]:
        """
        Complete preprocessing pipeline:
        Decode -> Validate -> (Enhance) -> (Resize) -> Extract Indices & Metadata.

        Returns:
            Tuple of (preprocessed_rgb, metadata, spectral_indices).
        """
        raw_rgb = self.decode_image(image_input)
        metadata = self.extract_metadata(raw_rgb)

        processed = raw_rgb
        if enhance and self.enable_clahe:
            processed = self.enhance_contrast(processed)

        if resize:
            processed, _ = self.resize_image(processed, target_size=target_size)

        indices = self.compute_spectral_indices(processed)
        return processed, metadata, indices
