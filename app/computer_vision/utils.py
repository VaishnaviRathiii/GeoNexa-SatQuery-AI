"""
Utility helpers for GeoNexa Computer Vision subsystem.
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Optional
from app.computer_vision.schemas import BoundingBox


# Distinct, visually appealing RGB colors for land-cover classes
CLASS_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "water": (30, 144, 255),          # Dodger Blue
    "river": (0, 191, 255),           # Deep Sky Blue
    "sealake": (25, 25, 112),         # Midnight Blue
    "forest": (34, 139, 34),          # Forest Green
    "vegetation_crop": (124, 252, 0), # Lawn Green
    "pasture": (154, 205, 50),        # Yellow Green
    "residential": (255, 140, 0),     # Dark Orange
    "industrial": (186, 85, 211),     # Medium Orchid
    "highway_road": (255, 215, 0),    # Gold
    "bare_soil": (210, 105, 30),      # Chocolate
    "unknown": (169, 169, 169),       # Dark Gray
}

# Distinct colors for segment cluster IDs
CLUSTER_PALETTE: List[Tuple[int, int, int]] = [
    (30, 144, 255),   # Blue
    (34, 139, 34),    # Green
    (255, 140, 0),    # Orange
    (186, 85, 211),   # Purple
    (255, 215, 0),    # Yellow
    (0, 206, 209),    # Dark Turquoise
    (255, 99, 71),    # Tomato
    (147, 112, 219),  # Medium Purple
    (60, 179, 113),   # Medium Sea Green
    (210, 105, 30),   # Chocolate
]


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Get standard RGB color for a given class name."""
    norm_name = class_name.lower().strip()
    return CLASS_COLOR_MAP.get(norm_name, CLASS_COLOR_MAP["unknown"])


def encode_image_to_base64(image_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encode an RGB numpy array into a base64 data URI string."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_rgb)}")

    # Ensure uint8
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            img_to_encode = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_to_encode = image_rgb.clip(0, 255).astype(np.uint8)
    else:
        img_to_encode = image_rgb

    # Convert to PIL and write to buffer
    pil_img = Image.fromarray(img_to_encode)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{encoded}"


def decode_base64_to_image(data_uri: str) -> np.ndarray:
    """Decode a base64 string or data URI into an RGB numpy array."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_uri)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def nms_boxes(
    boxes: List[BoundingBox],
    scores: List[float],
    iou_threshold: float = 0.4,
) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) on a list of BoundingBoxes.
    Returns the indices of kept boxes.
    """
    if not boxes or not scores:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(int(idx))
        if len(order) == 1:
            break

        current_box = boxes[idx]
        remaining_indices = order[1:]
        ious = np.array([current_box.iou(boxes[i]) for i in remaining_indices])

        # Keep only indices where IoU is less than threshold
        keep_mask = ious <= iou_threshold
        order = remaining_indices[keep_mask]

    return keep
