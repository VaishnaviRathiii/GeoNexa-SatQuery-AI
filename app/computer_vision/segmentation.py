"""
Satellite Image Segmentation Module.

Implements multi-strategy image segmentation for remote sensing imagery:
K-Means clustering, SLIC superpixel decomposition, Multi-Otsu, and
spectral rule-based land-cover segmentation.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from skimage.segmentation import slic


class SatelliteSegmenter:
    """
    Modular segmentation engine tailored for remote sensing satellite imagery.
    """

    def __init__(
        self,
        default_k: int = 5,
        slic_n_segments: int = 48,
        slic_compactness: float = 12.0,
    ):
        self.default_k = default_k
        self.slic_n_segments = slic_n_segments
        self.slic_compactness = slic_compactness

    def segment_kmeans(
        self,
        image: np.ndarray,
        k: Optional[int] = None,
        include_spatial: bool = True,
        include_spectral: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment satellite image using multi-feature K-Means clustering.
        """
        k = k or self.default_k
        h, w = image.shape[:2]

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        feature_list = [lab[:, :, 0:1], lab[:, :, 1:2], lab[:, :, 2:3]]

        if include_spatial:
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            spatial_weight = 0.20
            feature_list.append((x_coords / float(w) * spatial_weight)[:, :, np.newaxis])
            feature_list.append((y_coords / float(h) * spatial_weight)[:, :, np.newaxis])

        if include_spectral:
            img_f = image.astype(np.float32) / 255.0
            r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
            exg = ((2.0 * g - r - b) + 1.0) / 2.0
            ndwi = ((b - r) / (b + r + 1e-7) + 1.0) / 2.0
            feature_list.append(exg[:, :, np.newaxis])
            feature_list.append(ndwi[:, :, np.newaxis])

        feature_matrix = np.concatenate(feature_list, axis=2).reshape(-1, len(feature_list))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        flags = cv2.KMEANS_PP_CENTERS

        _, labels, centers = cv2.kmeans(
            feature_matrix.astype(np.float32),
            k,
            None,
            criteria,
            5,
            flags,
        )

        label_map = labels.reshape(h, w).astype(np.int32)
        return label_map, centers

    def segment_slic(
        self,
        image: np.ndarray,
        n_segments: Optional[int] = None,
        compactness: Optional[float] = None,
    ) -> np.ndarray:
        """
        Segment image into compact superpixels using the SLIC algorithm.
        """
        n_segments = n_segments or self.slic_n_segments
        compactness = compactness or self.slic_compactness

        try:
            segments = slic(
                image,
                n_segments=n_segments,
                compactness=compactness,
                start_label=0,
                channel_axis=2,
            )
            return segments.astype(np.int32)
        except Exception:
            labels, _ = self.segment_kmeans(image, k=min(12, n_segments))
            return labels

    def segment_semantic_rules(
        self,
        image: np.ndarray,
        spectral_indices: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Produce dedicated semantic binary segmentation masks for distinct
        satellite land-cover classes.
        """
        h, w = image.shape[:2]
        img_f = image.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

        exg = spectral_indices.get("exg", np.zeros((h, w), dtype=np.float32))
        ndwi = spectral_indices.get("ndwi_rgb", np.zeros((h, w), dtype=np.float32))
        ndti = spectral_indices.get("ndti", np.zeros((h, w), dtype=np.float32))
        brightness = spectral_indices.get("brightness", np.zeros((h, w), dtype=np.float32))
        texture = spectral_indices.get("texture", np.zeros((h, w), dtype=np.float32))

        # 1. Water mask (strong blue dominance, or high NDWI with negative ExG)
        water_mask = (
            ((b > g * 1.35) & (ndwi > 0.15))
            | ((ndwi > 0.30) & (exg < 0.01))
            | ((b > r * 1.3) & (b > g * 1.25) & (brightness < 0.85))
        )

        # 2. Forest (dense green canopy / strong green dominance)
        forest_mask = (
            (
                (exg > 0.06)
                | ((g > r * 1.30) & (g >= b * 0.85))
                | ((g > r * 1.20) & (brightness < 0.45) & (exg > 0.01))
            )
            & (~water_mask)
        )

        # 3. Crop / Agricultural vegetation (lighter green / active crops / pasture)
        crop_mask = (
            ((exg > 0.02) | ((g > r * 1.10) & (g > b * 0.85)))
            & (~forest_mask)
            & (~water_mask)
        )

        # 4. Urban / Residential (high texture variation, moderate brightness)
        residential_mask = (
            (texture > 0.035) & (brightness > 0.22) & (brightness < 0.58)
            & (~water_mask) & (~forest_mask)
        )

        # 5. Industrial (large bright roofs, low vegetation)
        industrial_mask = (
            ((brightness > 0.55) & (texture > 0.05)) | (brightness > 0.68)
        ) & (~water_mask) & (~forest_mask) & (~crop_mask)

        # 6. Bare soil / tilled earth (earth tones: red > green, positive NDTI)
        soil_mask = (
            ((ndti > 0.02) | (r > g * 1.05))
            & (~water_mask) & (~forest_mask) & (~crop_mask) & (~residential_mask) & (~industrial_mask)
        )

        # Cleanup masks with morphological open/close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        masks = {
            "water": water_mask,
            "forest": forest_mask,
            "vegetation_crop": crop_mask,
            "residential": residential_mask,
            "industrial": industrial_mask,
            "bare_soil": soil_mask,
        }

        cleaned_masks = {}
        for class_name, raw_m in masks.items():
            u8_m = raw_m.astype(np.uint8) * 255
            cleaned = cv2.morphologyEx(u8_m, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            cleaned_masks[class_name] = cleaned

        return cleaned_masks

    def create_unified_landcover_map(
        self,
        image: np.ndarray,
        spectral_indices: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Merge semantic rule masks with K-Means clustering into a unified
        land-cover segmentation map where each region has a clear label.
        """
        h, w = image.shape[:2]
        semantic_masks = self.segment_semantic_rules(image, spectral_indices)

        unified_map = np.full((h, w), -1, dtype=np.int32)
        class_dict: Dict[int, str] = {}

        class_priorities = [
            ("water", 1),
            ("forest", 2),
            ("industrial", 3),
            ("residential", 4),
            ("vegetation_crop", 5),
            ("bare_soil", 6),
        ]

        for class_name, class_id in class_priorities:
            mask = semantic_masks[class_name] > 0
            assign_mask = mask & (unified_map == -1)
            unified_map[assign_mask] = class_id
            class_dict[class_id] = class_name

        # For remaining unclassified pixels, cluster via K-Means
        unassigned = unified_map == -1
        if np.any(unassigned):
            kmeans_map, _ = self.segment_kmeans(image, k=3)
            next_id = 7
            for k_val in np.unique(kmeans_map):
                k_mask = (kmeans_map == k_val) & unassigned
                if np.any(k_mask):
                    unified_map[k_mask] = next_id
                    class_dict[next_id] = f"terrain_cluster_{k_val + 1}"
                    next_id += 1

        return unified_map, class_dict
