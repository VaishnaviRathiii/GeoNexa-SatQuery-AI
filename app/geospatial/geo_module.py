"""
Geospatial module (M3's area - Pranavi).

This is a MOCK implementation. Replace the body of get_geo_info()
with Pranavi's real GeoTIFF/metadata/coordinate extraction once ready,
keeping the same input/output shape:

    input:  image (bytes)
    output: dict with geographic metadata
"""


def get_geo_info(image_bytes: bytes) -> dict:
    """
    Mock geospatial info extractor.
    Real version (M3) will read GeoTIFF metadata, coordinates and CRS
    from the uploaded image, where available.
    """
    return {
        "coordinates": {"lat": 0.0, "lon": 0.0},
        "crs": "MOCK_CRS",
        "note": "Mock geospatial data - real metadata pending M3's module",
    }