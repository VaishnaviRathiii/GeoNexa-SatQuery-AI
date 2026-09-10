"""
Region Identification and Query Grounding Module.

Extracts spatial, geometric, and spectral properties from segmented regions,
classifies land-cover types, and performs query-specific visual grounding.
"""

import re
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from app.computer_vision.schemas import (
    BoundingBox,
    DetectedFeature,
    SegmentedRegion,
    GroundedRegion,
)


# Semantic mapping of query terms / synonyms to target land-cover classes
QUERY_CLASS_SYNONYMS: Dict[str, Set[str]] = {
    "water": {
        "water", "waterbody", "waterbodies", "river", "rivers", "lake", "lakes",
        "sea", "ocean", "pond", "reservoir", "stream", "canal", "wetland", "sealake"
    },
    "forest": {
        "forest", "forests", "forestry", "trees", "woodland", "jungle", "canopy",
        "timberland", "dense vegetation", "wood"
    },
    "vegetation_crop": {
        "crop", "crops", "agriculture", "agricultural", "farm", "farmland", "farming",
        "pasture", "grass", "grassland", "field", "fields", "vegetation", "annualcrop",
        "permanentcrop", "herbaceous", "herbaceousvegetation", "greenery"
    },
    "residential": {
        "residential", "house", "houses", "housing", "homes", "suburb", "suburban",
        "neighborhood", "settlement", "village", "town", "urban", "living area", "living areas",
        "buildings", "properties"
    },
    "industrial": {
        "industrial", "industry", "factory", "factories", "plant", "warehouse",
        "commercial", "complex", "manufacturing", "heavy industry", "storage"
    },
    "highway_road": {
        "road", "roads", "highway", "highways", "freeway", "motorway", "street",
        "pathway", "transport", "corridor", "infrastructure", "pavement", "linear"
    },
    "bare_soil": {
        "soil", "bare soil", "ground", "bare ground", "dirt", "sand", "fallow",
        "quarry", "cleared land", "earth"
    },
}


class RegionIdentifier:
    """
    Identifies, describes, and grounds visual regions in remote sensing scenes.
    """

    def __init__(self, min_region_area: int = 8):
        self.min_region_area = min_region_area

    def extract_regions_from_map(
        self,
        image: np.ndarray,
        label_map: np.ndarray,
        class_dict: Dict[int, str],
        spectral_indices: Dict[str, np.ndarray],
    ) -> List[SegmentedRegion]:
        """
        Extract individual contiguous segmented regions with full geometric,
        spectral, and spatial attributes.
        """
        h, w = image.shape[:2]
        total_pixels = float(h * w)
        regions: List[SegmentedRegion] = []
        region_counter = 1

        exg = spectral_indices.get("exg", np.zeros((h, w), dtype=np.float32))
        ndwi = spectral_indices.get("ndwi_rgb", np.zeros((h, w), dtype=np.float32))
        ndti = spectral_indices.get("ndti", np.zeros((h, w), dtype=np.float32))
        brightness = spectral_indices.get("brightness", np.zeros((h, w), dtype=np.float32))
        texture = spectral_indices.get("texture", np.zeros((h, w), dtype=np.float32))

        unique_labels = np.unique(label_map)

        for lbl in unique_labels:
            if lbl < 0:
                continue

            class_name = class_dict.get(int(lbl), f"cluster_{lbl}")
            binary_mask = (label_map == lbl).astype(np.uint8) * 255

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_region_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                area_pct = (area / total_pixels) * 100.0

                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = float(m["m10"] / m["m00"])
                    cy = float(m["m01"] / m["m00"])
                else:
                    cx = float(x + bw / 2.0)
                    cy = float(y + bh / 2.0)

                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                contour_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

                roi_mask = binary_mask[y : y + bh, x : x + bw] > 0
                roi_rgb = image[y : y + bh, x : x + bw]

                if np.any(roi_mask):
                    mean_r = int(np.mean(roi_rgb[:, :, 0][roi_mask]))
                    mean_g = int(np.mean(roi_rgb[:, :, 1][roi_mask]))
                    mean_b = int(np.mean(roi_rgb[:, :, 2][roi_mask]))

                    reg_exg = float(np.mean(exg[y : y + bh, x : x + bw][roi_mask]))
                    reg_ndwi = float(np.mean(ndwi[y : y + bh, x : x + bw][roi_mask]))
                    reg_ndti = float(np.mean(ndti[y : y + bh, x : x + bw][roi_mask]))
                    reg_bright = float(np.mean(brightness[y : y + bh, x : x + bw][roi_mask]))
                    reg_tex = float(np.mean(texture[y : y + bh, x : x + bw][roi_mask]))
                else:
                    mean_r, mean_g, mean_b = int(np.mean(roi_rgb[:, :, 0])), int(np.mean(roi_rgb[:, :, 1])), int(np.mean(roi_rgb[:, :, 2]))
                    reg_exg, reg_ndwi, reg_ndti, reg_bright, reg_tex = 0.0, 0.0, 0.0, 0.5, 0.1

                confidence = self._compute_region_confidence(class_name, reg_exg, reg_ndwi, reg_ndti, reg_bright, reg_tex)
                bbox = BoundingBox(ymin=y, xmin=x, ymax=min(h, y + bh), xmax=min(w, x + bw))

                region = SegmentedRegion(
                    region_id=region_counter,
                    label=class_name,
                    class_name=class_name,
                    confidence=confidence,
                    area_pixels=int(area),
                    area_percentage=area_pct,
                    centroid=(cx, cy),
                    bbox=bbox,
                    mean_color_rgb=[mean_r, mean_g, mean_b],
                    spectral_indices={
                        "exg": reg_exg,
                        "ndwi": reg_ndwi,
                        "ndti": reg_ndti,
                        "brightness": reg_bright,
                        "texture": reg_tex,
                    },
                    contour_points=contour_pts,
                )

                regions.append(region)
                region_counter += 1

        regions.sort(key=lambda r: r.area_pixels, reverse=True)
        return regions

    def _compute_region_confidence(
        self,
        class_name: str,
        exg: float,
        ndwi: float,
        ndti: float,
        brightness: float,
        texture: float,
    ) -> float:
        """Compute confidence score for a labeled region."""
        base_conf = 0.75

        if "water" in class_name or "river" in class_name or "sealake" in class_name:
            if ndwi > 0.05:
                return min(0.96, base_conf + 0.20)
            return 0.80

        if "forest" in class_name:
            if exg > 0.04 and brightness < 0.45:
                return min(0.95, base_conf + 0.18)
            return 0.80

        if "vegetation" in class_name or "crop" in class_name or "pasture" in class_name:
            if exg > 0.02:
                return min(0.92, base_conf + 0.15)
            return 0.78

        if "residential" in class_name:
            if texture > 0.035:
                return min(0.90, base_conf + 0.14)
            return 0.76

        if "industrial" in class_name:
            if brightness > 0.55:
                return min(0.92, base_conf + 0.16)
            return 0.76

        if "soil" in class_name:
            if ndti > 0.02:
                return min(0.90, base_conf + 0.14)
            return 0.74

        return base_conf

    def ground_query(
        self,
        query: str,
        regions: List[SegmentedRegion],
        detections: Optional[List[DetectedFeature]] = None,
    ) -> List[GroundedRegion]:
        """
        Ground a textual query against segmented and detected visual regions.
        """
        if not query or not query.strip():
            return []

        clean_query = query.lower().strip()
        words = set(re.findall(r"\b[a-z_]+\b", clean_query))

        # Identify which target classes match the query words
        matched_target_classes: Dict[str, float] = {}

        for target_class, synonyms in QUERY_CLASS_SYNONYMS.items():
            intersection = words.intersection(synonyms)
            if intersection:
                matched_target_classes[target_class] = 1.0
            else:
                for syn in synonyms:
                    if syn in clean_query:
                        matched_target_classes[target_class] = 0.85
                        break

        grounded: List[GroundedRegion] = []
        seen_bboxes = set()

        # 1. Ground from segmented regions
        for region in regions:
            reg_class = region.class_name.lower()
            rel_score = 0.0
            matched_cls = reg_class

            for target_cls, base_score in matched_target_classes.items():
                if target_cls in reg_class or reg_class in target_cls:
                    rel_score = max(rel_score, base_score * region.confidence)
                    matched_cls = target_cls

                elif "cluster" in reg_class or reg_class == "unknown":
                    sp = region.spectral_indices
                    if target_cls == "water" and sp.get("ndwi", 0) > 0.04:
                        rel_score = max(rel_score, 0.75 * region.confidence)
                        matched_cls = "water"
                    elif target_cls == "forest" and sp.get("exg", 0) > 0.01 and sp.get("brightness", 1.0) < 0.45:
                        rel_score = max(rel_score, 0.75 * region.confidence)
                        matched_cls = "forest"
                    elif target_cls == "vegetation_crop" and sp.get("exg", 0) > 0.01:
                        rel_score = max(rel_score, 0.75 * region.confidence)
                        matched_cls = "vegetation_crop"
                    elif target_cls == "residential" and sp.get("texture", 0) > 0.03:
                        rel_score = max(rel_score, 0.75 * region.confidence)
                        matched_cls = "residential"
                    elif target_cls == "industrial" and sp.get("brightness", 0) > 0.50:
                        rel_score = max(rel_score, 0.75 * region.confidence)
                        matched_cls = "industrial"

            if rel_score == 0.0:
                for w in words:
                    if w in reg_class:
                        rel_score = 0.70 * region.confidence
                        break

            if rel_score >= 0.35:
                explanation = (
                    f"Grounded '{region.label}' region ({region.area_percentage}% of scene) "
                    f"matching query '{query}' with {round(rel_score * 100, 1)}% confidence."
                )

                bbox_tuple = (region.bbox.ymin, region.bbox.xmin, region.bbox.ymax, region.bbox.xmax)
                seen_bboxes.add(bbox_tuple)

                grounded.append(
                    GroundedRegion(
                        target_query=query,
                        matched_class=matched_cls,
                        relevance_score=rel_score,
                        region=region,
                        explanation=explanation,
                    )
                )

        # 2. Ground from detected features (e.g., linear roads, builtup clusters)
        if detections:
            for det in detections:
                det_label = det.label.lower()
                rel_score = 0.0
                matched_cls = det_label

                for target_cls, base_score in matched_target_classes.items():
                    if target_cls in det_label or det_label in target_cls:
                        rel_score = max(rel_score, base_score * det.confidence)
                        matched_cls = target_cls

                if rel_score >= 0.35:
                    bbox_tuple = (det.bbox.ymin, det.bbox.xmin, det.bbox.ymax, det.bbox.xmax)
                    if bbox_tuple not in seen_bboxes:
                        synth_region = SegmentedRegion(
                            region_id=len(grounded) + 100,
                            label=det.label,
                            class_name=matched_cls,
                            confidence=det.confidence,
                            area_pixels=det.bbox.area,
                            area_percentage=round((det.bbox.area / 4096.0) * 100.0, 2),
                            centroid=(float(det.bbox.xmin + det.bbox.width / 2.0), float(det.bbox.ymin + det.bbox.height / 2.0)),
                            bbox=det.bbox,
                            mean_color_rgb=[150, 150, 150],
                            spectral_indices={},
                            contour_points=None,
                        )

                        grounded.append(
                            GroundedRegion(
                                target_query=query,
                                matched_class=matched_cls,
                                relevance_score=rel_score,
                                region=synth_region,
                                explanation=f"Grounded detected '{det.label}' feature ({det.feature_type}) matching query '{query}'.",
                            )
                        )
                        seen_bboxes.add(bbox_tuple)

        grounded.sort(key=lambda g: (g.relevance_score, g.region.area_pixels), reverse=True)
        return grounded

    def generate_summary(
        self,
        regions: List[SegmentedRegion],
        detections: List[DetectedFeature],
        grounded: Optional[List[GroundedRegion]] = None,
        query: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive, human-readable natural language summary
        of the Computer Vision scene analysis.
        """
        if not regions and not detections:
            return "No prominent features or regions detected in satellite scene."

        class_areas: Dict[str, float] = {}
        for r in regions:
            c = r.class_name
            class_areas[c] = class_areas.get(c, 0.0) + r.area_percentage

        sorted_classes = sorted(class_areas.items(), key=lambda x: x[1], reverse=True)
        class_breakdown = ", ".join([f"{name}: {pct:.1f}%" for name, pct in sorted_classes[:4]])

        summary_parts = [f"Scene breakdown ({class_breakdown})."]

        if detections:
            feat_types = [d.label for d in detections[:3]]
            summary_parts.append(f"Identified {len(detections)} key visual features including {', '.join(set(feat_types))}.")

        if query and grounded:
            summary_parts.append(
                f"Query grounding for '{query}': highlighted {len(grounded)} matching region(s)."
            )
        elif query and not grounded:
            summary_parts.append(
                f"Query '{query}' did not match specific high-confidence visual regions."
            )

        return " ".join(summary_parts)
