"""
GeoNexa Computer Vision Subsystem (M2 - Shubham).

Provides satellite image preprocessing, feature/object detection,
segmentation, region identification, query-specific visual grounding,
and visual highlighting.
"""

from app.computer_vision.cv_module import get_cv_analysis
from app.computer_vision.pipeline import ComputerVisionPipeline, get_default_pipeline
from app.computer_vision.preprocessing import SatellitePreprocessor
from app.computer_vision.detection import SatelliteFeatureDetector
from app.computer_vision.segmentation import SatelliteSegmenter
from app.computer_vision.region_identifier import RegionIdentifier
from app.computer_vision.visualization import SatelliteVisualizer
from app.computer_vision.schemas import (
    BoundingBox,
    DetectedFeature,
    SegmentedRegion,
    GroundedRegion,
    CVAnalysisResult,
)

__all__ = [
    "get_cv_analysis",
    "ComputerVisionPipeline",
    "get_default_pipeline",
    "SatellitePreprocessor",
    "SatelliteFeatureDetector",
    "SatelliteSegmenter",
    "RegionIdentifier",
    "SatelliteVisualizer",
    "BoundingBox",
    "DetectedFeature",
    "SegmentedRegion",
    "GroundedRegion",
    "CVAnalysisResult",
]
