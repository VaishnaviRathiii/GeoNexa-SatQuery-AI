"""
Data structures and schemas for GeoNexa Computer Vision subsystem.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class BoundingBox:
    """Bounding box coordinates [ymin, xmin, ymax, xmax] and [x, y, w, h]."""
    ymin: int
    xmin: int
    ymax: int
    xmax: int

    @property
    def x(self) -> int:
        return self.xmin

    @property
    def y(self) -> int:
        return self.ymin

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection-over-Union with another bounding box."""
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area) / float(union_area)

    def to_dict(self) -> Dict[str, int]:
        return {
            "ymin": int(self.ymin),
            "xmin": int(self.xmin),
            "ymax": int(self.ymax),
            "xmax": int(self.xmax),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> "BoundingBox":
        return cls(ymin=int(y), xmin=int(x), ymax=int(y + h), xmax=int(x + w))


@dataclass
class DetectedFeature:
    """Detected feature or object in satellite imagery."""
    feature_id: str
    feature_type: str  # linear_road, built_cluster, water_body, vegetation_parcel, keypoint_cluster
    label: str
    confidence: float
    bbox: BoundingBox
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "properties": self.properties,
        }


@dataclass
class SegmentedRegion:
    """Segmented region / land-cover zone in satellite imagery."""
    region_id: int
    label: str
    class_name: str
    confidence: float
    area_pixels: int
    area_percentage: float
    centroid: Tuple[float, float]  # (x, y)
    bbox: BoundingBox
    mean_color_rgb: List[int]
    spectral_indices: Dict[str, float] = field(default_factory=dict)
    contour_points: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": int(self.region_id),
            "label": self.label,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "area_pixels": int(self.area_pixels),
            "area_percentage": round(float(self.area_percentage), 2),
            "centroid": [round(float(c), 2) for c in self.centroid],
            "bbox": self.bbox.to_dict(),
            "mean_color_rgb": [int(c) for c in self.mean_color_rgb],
            "spectral_indices": {k: round(float(v), 4) for k, v in self.spectral_indices.items()},
            "contour_points": self.contour_points,
        }


@dataclass
class GroundedRegion:
    """Region grounded against a textual search query."""
    target_query: str
    matched_class: str
    relevance_score: float
    region: SegmentedRegion
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_query": self.target_query,
            "matched_class": self.matched_class,
            "relevance_score": round(float(self.relevance_score), 4),
            "region": self.region.to_dict(),
            "explanation": self.explanation,
        }


@dataclass
class ImageMetadata:
    """Metadata extracted during satellite preprocessing."""
    width: int
    height: int
    channels: int
    format: str
    mean_rgb: List[float]
    std_rgb: List[float]
    estimated_scene_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "channels": int(self.channels),
            "format": self.format,
            "mean_rgb": [round(float(v), 2) for v in self.mean_rgb],
            "std_rgb": [round(float(v), 2) for v in self.std_rgb],
            "estimated_scene_type": self.estimated_scene_type,
        }


@dataclass
class CVAnalysisResult:
    """Complete Computer Vision analysis response structure."""
    status: str  # success, error, warning
    image_metadata: Dict[str, Any]
    detected_features: List[Dict[str, Any]]
    detections: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    grounded_regions: List[Dict[str, Any]]
    highlighted_image: str  # Base64 data URI
    mask_image: Optional[str] = None
    summary: str = ""
    detected_regions: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    query: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "image_metadata": self.image_metadata,
            "detected_features": self.detected_features,
            "detections": self.detections,
            "segments": self.segments,
            "grounded_regions": self.grounded_regions,
            "highlighted_image": self.highlighted_image,
            "mask_image": self.mask_image,
            "summary": self.summary,
            "detected_regions": self.detected_regions,
            "labels": self.labels,
            "query": self.query,
            "error": self.error,
        }
