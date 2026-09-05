"""
Computer Vision module (M2's area - Shubham).

This is a MOCK implementation. Replace the body of get_cv_analysis()
with Shubham's real preprocessing/detection/segmentation output once
ready, keeping the same input/output shape:

    input:  image (bytes)
    output: dict with detected regions / labels
"""


def get_cv_analysis(image_bytes: bytes) -> dict:
    """
    Mock computer vision analysis.
    Real version (M2) will preprocess the image and return detected
    features/regions, e.g. bounding boxes or segmentation masks.
    """
    return {
        "detected_regions": ["mock_region_1", "mock_region_2"],
        "labels": ["vegetation", "water_body"],
    }