"""
Geospatial module (M3's area - Pranavi).

This is a compatible interface implementation.
"""


def get_geo_info(image_bytes: bytes) -> dict:
    """
    Mock geospatial info extractor.
    """
    return {
        "coordinates": {"lat": 0.0, "lon": 0.0},
        "crs": "MOCK_CRS",
        "note": "Mock geospatial data - real metadata pending M3's module",
    }
