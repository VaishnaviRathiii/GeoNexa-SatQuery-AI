"""
Satellite Image Visualization and Highlighting Module.

Generates rich visual outputs: segmentation mask overlays, bounding boxes,
label badges, query spotlights, side-by-side comparisons, and base64 exports.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
from app.computer_vision.schemas import (
    SegmentedRegion,
    DetectedFeature,
    GroundedRegion,
)
from app.computer_vision.utils import (
    get_class_color,
    encode_image_to_base64,
    CLUSTER_PALETTE,
)


class SatelliteVisualizer:
    """
    Renders high-quality visual annotations for satellite scenes.
    """

    def __init__(
        self,
        default_alpha: float = 0.40,
        box_thickness: int = 2,
        font_scale: float = 0.45,
    ):
        self.default_alpha = default_alpha
        self.box_thickness = box_thickness
        self.font_scale = font_scale

    def draw_segmentation_overlay(
        self,
        image: np.ndarray,
        regions: List[SegmentedRegion],
        alpha: Optional[float] = None,
        draw_contours: bool = True,
    ) -> np.ndarray:
        """
        Overlay colored segmentation masks and crisp boundary contours
        onto the satellite image.
        """
        alpha = alpha if alpha is not None else self.default_alpha
        h, w = image.shape[:2]
        overlay = image.copy()
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for reg in regions:
            color = get_class_color(reg.class_name)
            if reg.contour_points and len(reg.contour_points) >= 3:
                pts = np.array(reg.contour_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_canvas, [pts], color)
                if draw_contours:
                    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
            else:
                # Fallback to bounding box fill
                b = reg.bbox
                cv2.rectangle(mask_canvas, (b.xmin, b.ymin), (b.xmax, b.ymax), color, -1)

        # Blend mask canvas with original image
        blended = cv2.addWeighted(overlay, 1.0 - alpha, mask_canvas, alpha, 0)
        return blended

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        detections: List[Union[DetectedFeature, SegmentedRegion]],
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw crisp bounding boxes with label tags and confidence badges.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for item in detections:
            bbox = item.bbox
            label = item.label
            conf = getattr(item, "confidence", 1.0)
            color = get_class_color(label)

            # Draw outer border with dark outline for contrast
            cv2.rectangle(
                annotated,
                (bbox.xmin, bbox.ymin),
                (bbox.xmax, bbox.ymax),
                (0, 0, 0),
                self.box_thickness + 2,
            )
            cv2.rectangle(
                annotated,
                (bbox.xmin, bbox.ymin),
                (bbox.xmax, bbox.ymax),
                color,
                self.box_thickness,
            )

            if show_labels:
                text = label
                if show_confidence and conf < 1.0:
                    text = f"{label} {int(conf * 100)}%"

                # Calculate text size for badge background
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(text, font, self.font_scale, 1)

                badge_ymin = max(0, bbox.ymin - th - 6)
                badge_ymax = max(th + 6, bbox.ymin)
                badge_xmin = bbox.xmin
                badge_xmax = min(w, bbox.xmin + tw + 6)

                # Filled badge background
                cv2.rectangle(annotated, (badge_xmin, badge_ymin), (badge_xmax, badge_ymax), color, -1)
                cv2.rectangle(annotated, (badge_xmin, badge_ymin), (badge_xmax, badge_ymax), (0, 0, 0), 1)

                # Text in white or black for high readability
                brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    text,
                    (badge_xmin + 3, badge_ymax - 4),
                    font,
                    self.font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def draw_query_spotlight(
        self,
        image: np.ndarray,
        grounded_regions: List[GroundedRegion],
        query: str,
    ) -> np.ndarray:
        """
        Create a query spotlight visualization:
        Dims non-relevant regions and highlights grounded regions with vibrant overlays.
        """
        h, w = image.shape[:2]
        # Dim background by 40%
        dimmed = (image.astype(np.float32) * 0.60).astype(np.uint8)
        highlighted = dimmed.copy()

        for g in grounded_regions:
            reg = g.region
            color = get_class_color(reg.class_name)

            if reg.contour_points and len(reg.contour_points) >= 3:
                pts = np.array(reg.contour_points, dtype=np.int32).reshape((-1, 1, 2))
                # Restore original bright pixels for region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # Blend region
                highlighted[mask > 0] = image[mask > 0]
                # Draw thick vibrant border
                cv2.polylines(highlighted, [pts], isClosed=True, color=color, thickness=2)
            else:
                b = reg.bbox
                highlighted[b.ymin : b.ymax, b.xmin : b.xmax] = image[b.ymin : b.ymax, b.xmin : b.xmax]
                cv2.rectangle(highlighted, (b.xmin, b.ymin), (b.xmax, b.ymax), color, 2)

            # Add badge
            badge_text = f"{reg.label} ({int(g.relevance_score * 100)}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                highlighted,
                badge_text,
                (reg.bbox.xmin + 2, max(14, reg.bbox.ymin - 3)),
                font,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Render top banner with query text
        banner_h = 24
        banner = np.full((banner_h, w, 3), (25, 25, 25), dtype=np.uint8)
        cv2.putText(
            banner,
            f"Query: {query[:45]}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        final_composite = np.vstack([banner, highlighted])
        return final_composite

    def create_side_by_side(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        title_left: str = "Original",
        title_right: str = "CV Analysis",
    ) -> np.ndarray:
        """
        Create a side-by-side composite comparison image.
        """
        h_orig, w_orig = original.shape[:2]
        h_ann, w_ann = annotated.shape[:2]

        target_h = max(h_orig, h_ann)
        # Resize if heights mismatch
        if h_orig != target_h:
            orig_scaled = cv2.resize(original, (int(w_orig * target_h / h_orig), target_h))
        else:
            orig_scaled = original

        if h_ann != target_h:
            ann_scaled = cv2.resize(annotated, (int(w_ann * target_h / h_ann), target_h))
        else:
            ann_scaled = annotated

        # Add header bars
        header_h = 20
        header_left = np.full((header_h, orig_scaled.shape[1], 3), (35, 35, 35), dtype=np.uint8)
        header_right = np.full((header_h, ann_scaled.shape[1], 3), (35, 35, 35), dtype=np.uint8)

        cv2.putText(header_left, title_left, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header_right, title_right, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        left_panel = np.vstack([header_left, orig_scaled])
        right_panel = np.vstack([header_right, ann_scaled])

        # Combine side by side with thin vertical separator line
        separator = np.full((left_panel.shape[0], 2, 3), (128, 128, 128), dtype=np.uint8)
        combined = np.hstack([left_panel, separator, right_panel])
        return combined

    def render_full_visualization(
        self,
        original_image: np.ndarray,
        regions: List[SegmentedRegion],
        detections: List[DetectedFeature],
        grounded_regions: Optional[List[GroundedRegion]] = None,
        query: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate primary highlighted visual output and return as (numpy_array, base64_data_uri).
        """
        if query and grounded_regions:
            # Query-specific highlight takes precedence
            highlighted = self.draw_query_spotlight(original_image, grounded_regions, query)
        else:
            # Standard multi-layer segmentation + detection highlight
            seg_overlay = self.draw_segmentation_overlay(original_image, regions[:8], alpha=0.35)
            highlighted = self.draw_bounding_boxes(seg_overlay, detections[:10])

        b64_uri = encode_image_to_base64(highlighted, format="PNG")
        return highlighted, b64_uri
