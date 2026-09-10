# GeoNexa-SatQuery-AI

AI-Powered Satellite Imagery Query & Visual Grounding System.

## Architecture

GeoNexa integrates Computer Vision (M2), AI/VLM (M1), Geospatial (M3), Backend API (M4), Frontend (M5), and Data Testing (M6).

- **Computer Vision Subsystem (`app/computer_vision/`)**:
  - Satellite image preprocessing, decoding, normalization, and spectral index calculation (ExG, VARI, NDWI, NDTI, texture).
  - Object and feature detection (highways, roads, residential/industrial built-up clusters, water bodies, agricultural parcels).
  - Multi-feature segmentation (K-Means, SLIC superpixels, semantic land-cover masks).
  - Region identification with bounding boxes, spatial centroids, and polygon boundaries.
  - Query-specific visual grounding and query spotlight highlighting.
  - Base64 annotated visual output generation.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest -v
```

### 3. Run Computer Vision Demo
```bash
python demo_cv.py
```

### 4. Run Backend Server
```bash
uvicorn app.backend.main:app --reload
```

For full documentation on the Computer Vision pipeline, see [docs/COMPUTER_VISION.md](docs/COMPUTER_VISION.md).
