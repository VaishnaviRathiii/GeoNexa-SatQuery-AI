"""
Satellite Feature and Object Detection Module.

Implements real, lightweight, reproducible feature and object detection
for remote sensing satellite imagery.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from app.computer_vision.schemas import BoundingBox, DetectedFeature
from app.computer_vision.utils import nms_boxes


class SatelliteFeatureDetector:
    """
    Multi-feature satellite detector combining morphological analysis,
    spectral indices, keypoint density clustering, and linear edge extraction.
    """

    def __init__(
        self,
        min_region_area: int = 16,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 25,
    ):
        self.min_region_area = min_region_area
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def detect_linear_features(
        self,
        image: np.ndarray,
        min_line_length: int = 20,
        max_line_gap: int = 6,
    ) -> List[DetectedFeature]:
        """
        Detect linear infrastructure (highways, main roads, long canals/rivers)
        using Canny edge detection and Probabilistic Hough Transform.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=min(min_line_length, max(8, min(h, w) // 4)),
            maxLineGap=max_line_gap,
        )

        detections = []
        if lines is None or len(lines) == 0:
            return detections

        # Group lines or extract prominent linear corridors
        for idx, line in enumerate(lines[:10]):
            x1, y1, x2, y2 = line[0]
            length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            angle = float(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Pad bounding box around the line
            pad = 4
            xmin = max(0, min(x1, x2) - pad)
            xmax = min(w, max(x1, x2) + pad)
            ymin = max(0, min(y1, y2) - pad)
            ymax = min(h, max(y1, y2) + pad)

            bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
            confidence = min(0.95, 0.50 + (length / max(h, w)) * 0.45)

            detections.append(
                DetectedFeature(
                    feature_id=f"linear_{idx + 1}",
                    feature_type="linear_road",
                    label="highway_road",
                    confidence=confidence,
                    bbox=bbox,
                    properties={
                        "length_pixels": round(length, 1),
                        "orientation_deg": round(angle, 1),
                        "endpoints": [[int(x1), int(y1)], [int(x2), int(y2)]],
                    },
                )
            )

        return detections

    def detect_builtup_structures(
        self,
        image: np.ndarray,
        spectral_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[DetectedFeature]:
        """
        Detect residential clusters and industrial structures using
        corner/texture density and high-reflectance roof signatures.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect Harris corners / keypoints for urban density
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corner_norm = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        corner_mask = (corner_norm > 140).astype(np.uint8)

        # Dilate to cluster nearby structural corners
        kernel_size = max(3, min(h, w) // 16)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        clustered = cv2.dilate(corner_mask, kernel, iterations=2)
        clustered = cv2.morphologyEx(clustered, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Clip bounds
            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(w, x + bw)
            ymax = min(h, y + bh)

            patch_img = image[ymin:ymax, xmin:xmax]
            patch_brightness = float(np.mean(patch_img))

            # Determine whether residential or industrial
            # Industrial tends to have higher uniform brightness and larger contiguous roofs
            if patch_brightness > 145 and (bw * bh) > (h * w * 0.08):
                label = "industrial"
                f_type = "built_cluster"
                conf = min(0.92, 0.60 + (area / (h * w)) * 0.3)
            else:
                label = "residential"
                f_type = "built_cluster"
                conf = min(0.88, 0.55 + (area / (h * w)) * 0.3)

            bbox = BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
            detections.append(
                DetectedFeature(
                    feature_id=f"built_{idx + 1}",
                    feature_type=f_type,
                    label=label,
                    confidence=conf,
                    bbox=bbox,
                    properties={
                        "area_pixels": int(area),
                        "mean_patch_brightness": round(patch_brightness, 1),
                    },
                )
            )

        return detections

    def detect_water_bodies(
        self,
        image: np.ndarray,
        spectral_indices: Dict[str, np.ndarray],
    ) -> List[DetectedFeature]:
        """
        Detect water bodies (lakes, rivers, ponds, sea) using spectral NDWI
        and low brightness absorption characteristics.
        """
        h, w = image.shape[:2]
        ndwi = spectral_indices.get("ndwi_rgb")
        brightness = spectral_indices.get("brightness")

        if ndwi is None or brightness is None:
            return []

        # Water: high NDWI_RGB and moderate-to-low brightness
        water_mask = ((ndwi > 0.04) & (brightness < 0.65)).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

            # Elongated water is typically a river/canal; broad is sea/lake
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                label = "river"
            else:
                label = "sealake"

            mean_ndwi_val = float(np.mean(ndwi[y : y + bh, x : x + bw]))
            confidence = min(0.96, 0.65 + max(0.0, mean_ndwi_val) * 0.4)

            bbox = BoundingBox(ymin=y, xmin=x, ymax=min(h, y + bh), xmax=min(w, x + bw))
            detections.append(
                DetectedFeature(
                    feature_id=f"water_{idx + 1}",
                    feature_type="water_body",
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    properties={
                        "area_pixels": int(area),
                        "aspect_ratio": round(aspect_ratio, 2),
                        "mean_ndwi": round(mean_ndwi_val, 4),
                    },
                )
            )

        return detections

    def detect_vegetation_parcels(
        self,
        image: np.ndarray,
        spectral_indices: Dict[str, np.ndarray],
    ) -> List[DetectedFeature]:
        """
        Detect vegetation parcels, agricultural fields, and forest stands.
        """
        h, w = image.shape[:2]
        exg = spectral_indices.get("exg")
        if exg is None:
            return []

        # Vegetation threshold
        veg_mask = (exg > 0.05).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            patch_exg = float(np.mean(exg[y : y + bh, x : x + bw]))

            # High ExG + darker green -> forest; lighter green -> crop / pasture
            patch_rgb = image[y : y + bh, x : x + bw]
            mean_intensity = float(np.mean(patch_rgb))

            if patch_exg > 0.12 and mean_intensity < 90:
                label = "forest"
            elif patch_exg > 0.08:
                label = "vegetation_crop"
            else:
                label = "pasture"

            confidence = min(0.94, 0.60 + max(0.0, patch_exg) * 0.4)
            bbox = BoundingBox(ymin=y, xmin=x, ymax=min(h, y + bh), xmax=min(w, x + bw))

            detections.append(
                DetectedFeature(
                    feature_id=f"veg_{idx + 1}",
                    feature_type="vegetation_parcel",
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    properties={
                        "area_pixels": int(area),
                        "mean_exg": round(patch_exg, 4),
                    },
                )
            )

        return detections

    def detect_all(
        self,
        image: np.ndarray,
        spectral_indices: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[DetectedFeature]:
        """
        Run all detectors on the satellite image, apply NMS, and return
        the most salient detected features.
        """
        from app.computer_vision.preprocessing import SatellitePreprocessor

        if spectral_indices is None:
            preprocessor = SatellitePreprocessor()
            spectral_indices = preprocessor.compute_spectral_indices(image)

        all_detections: List[DetectedFeature] = []

        # 1. Detect water bodies
        all_detections.extend(self.detect_water_bodies(image, spectral_indices))

        # 2. Detect vegetation parcels & forests
        all_detections.extend(self.detect_vegetation_parcels(image, spectral_indices))

        # 3. Detect built-up & industrial clusters
        all_detections.extend(self.detect_builtup_structures(image, spectral_indices))

        # 4. Detect linear features (roads / highways / canals)
        all_detections.extend(self.detect_linear_features(image))

        if not all_detections:
            return []

        # Apply Non-Maximum Suppression
        boxes = [d.bbox for d in all_detections]
        scores = [d.confidence for d in all_detections]
        keep_indices = nms_boxes(boxes, scores, iou_threshold=self.nms_iou_threshold)

        filtered = [all_detections[i] for i in keep_indices]
        # Sort by confidence descending
        filtered.sort(key=lambda d: d.confidence, reverse=True)

        return filtered[: self.max_detections]
