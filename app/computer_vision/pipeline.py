"""
End-to-End Computer Vision Pipeline Module.

Orchestrates preprocessing, feature detection, segmentation, region identification,
query grounding, and visualization into a single unified pipeline.
"""

from typing import Union, Optional, Tuple, Dict, Any, List
from PIL import Image
import numpy as np

from app.computer_vision.schemas import (
    CVAnalysisResult,
    ImageMetadata,
)
from app.computer_vision.preprocessing import SatellitePreprocessor
from app.computer_vision.detection import SatelliteFeatureDetector
from app.computer_vision.segmentation import SatelliteSegmenter
from app.computer_vision.region_identifier import RegionIdentifier
from app.computer_vision.visualization import SatelliteVisualizer


class ComputerVisionPipeline:
    """
    Main Computer Vision pipeline for GeoNexa satellite image query analysis.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        min_region_area: int = 12,
        nms_iou_threshold: float = 0.45,
    ):
        self.preprocessor = SatellitePreprocessor(target_size=target_size)
        self.detector = SatelliteFeatureDetector(
            min_region_area=min_region_area,
            nms_iou_threshold=nms_iou_threshold,
        )
        self.segmenter = SatelliteSegmenter()
        self.region_identifier = RegionIdentifier(min_region_area=min_region_area)
        self.visualizer = SatelliteVisualizer()

    def process(
        self,
        image_input: Union[bytes, str, Image.Image, np.ndarray],
        query: Optional[str] = None,
        resize: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        enhance: bool = False,
    ) -> CVAnalysisResult:
        """
        Execute full end-to-end CV analysis on satellite image.

        Args:
            image_input: Raw image bytes, file path, PIL Image, or numpy array.
            query: Optional textual query for visual grounding (e.g. "Find water bodies").
            resize: Whether to standardize image dimensions.
            target_size: Optional custom (width, height) resolution.
            enhance: Whether to apply CLAHE contrast enhancement.

        Returns:
            CVAnalysisResult with structured data, detections, masks, and highlighted image.
        """
        try:
            # 1. Preprocessing & Spectral Index Extraction
            processed_img, metadata, spectral_indices = self.preprocessor.preprocess(
                image_input=image_input,
                resize=resize,
                target_size=target_size,
                enhance=enhance,
            )

            # 2. Land-cover Segmentation
            unified_map, class_dict = self.segmenter.create_unified_landcover_map(
                processed_img,
                spectral_indices,
            )

            # 3. Region Extraction & Classification
            regions = self.region_identifier.extract_regions_from_map(
                image=processed_img,
                label_map=unified_map,
                class_dict=class_dict,
                spectral_indices=spectral_indices,
            )

            # 4. Feature & Object Detection
            detections = self.detector.detect_all(
                processed_img,
                spectral_indices=spectral_indices,
            )

            # 5. Query Grounding (if query provided)
            grounded_regions = []
            if query and query.strip():
                grounded_regions = self.region_identifier.ground_query(
                    query=query,
                    regions=regions,
                    detections=detections,
                )

            # 6. Generate Summary
            summary = self.region_identifier.generate_summary(
                regions=regions,
                detections=detections,
                grounded=grounded_regions,
                query=query,
            )

            # 7. Render Highlighting & Visual Output
            annotated_img, b64_highlighted = self.visualizer.render_full_visualization(
                original_image=processed_img,
                regions=regions,
                detections=detections,
                grounded_regions=grounded_regions if (query and grounded_regions) else None,
                query=query,
            )

            # 8. Format structured backward-compatible lists
            detected_regions_list = [
                f"{r.label} (Area: {r.area_percentage}%, Conf: {int(r.confidence * 100)}%)"
                for r in regions[:6]
            ]
            all_labels = sorted(list(set([r.class_name for r in regions] + [d.label for d in detections])))

            return CVAnalysisResult(
                status="success",
                image_metadata=metadata.to_dict(),
                detected_features=[d.to_dict() for d in detections],
                detections=[d.bbox.to_dict() for d in detections],
                segments=[r.to_dict() for r in regions],
                grounded_regions=[g.to_dict() for g in grounded_regions],
                highlighted_image=b64_highlighted,
                summary=summary,
                detected_regions=detected_regions_list,
                labels=all_labels,
                query=query,
                error=None,
            )

        except Exception as err:
            # Graceful error response so backend never crashes on invalid inputs
            return CVAnalysisResult(
                status="error",
                image_metadata={},
                detected_features=[],
                detections=[],
                segments=[],
                grounded_regions=[],
                highlighted_image="",
                summary="Error occurred during Computer Vision processing.",
                detected_regions=[],
                labels=[],
                query=query,
                error=str(err),
            )


# Global singleton pipeline instance
_DEFAULT_PIPELINE: Optional[ComputerVisionPipeline] = None


def get_default_pipeline() -> ComputerVisionPipeline:
    """Get or create singleton instance of ComputerVisionPipeline."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = ComputerVisionPipeline()
    return _DEFAULT_PIPELINE
