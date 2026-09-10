"""
Computer Vision module (M2's area - Shubham).

Main entrypoint integrated with GeoNexa backend architecture.
Provides real satellite-image preprocessing, feature/object detection,
segmentation, region identification, query-specific visual grounding,
and visual highlighting.
"""

from typing import Optional, Union, Dict, Any
from app.computer_vision.pipeline import (
    ComputerVisionPipeline,
    get_default_pipeline,
)


def get_cv_analysis(
    image_bytes: bytes,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform full Computer Vision analysis on satellite image bytes.

    Args:
        image_bytes: Raw bytes of the uploaded satellite image.
        query: Optional text question or target feature query
               (e.g. "Find water bodies", "Show residential areas").

    Returns:
        dict containing:
            - status: "success" | "error"
            - image_metadata: resolution, color stats, scene type
            - detected_features: list of detected objects/features with bounding boxes
            - segments: segmented land-cover regions with spatial and spectral properties
            - grounded_regions: query-matched regions with relevance scores
            - highlighted_image: Base64 data URI of the visual output
            - summary: natural language description of scene features
            - detected_regions: backward-compatible summary list
            - labels: list of all detected land-cover labels
    """
    pipeline = get_default_pipeline()
    result = pipeline.process(image_bytes, query=query)
    return result.to_dict()


def analyze_image_file(
    image_path: str,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to analyze a local image file path.
    """
    pipeline = get_default_pipeline()
    result = pipeline.process(image_path, query=query)
    return result.to_dict()


def get_pipeline() -> ComputerVisionPipeline:
    """
    Get the active ComputerVisionPipeline instance.
    """
    return get_default_pipeline()
