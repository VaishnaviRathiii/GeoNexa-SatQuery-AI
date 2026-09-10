# GeoNexa Computer Vision Subsystem (M2 - Shubham)

## 1. Overview & Architecture

The **GeoNexa Computer Vision Subsystem** provides an end-to-end, lightweight, production-grade vision pipeline tailored for remote sensing and satellite imagery. It performs satellite-image preprocessing, multi-scale feature detection, land-cover segmentation, spatial/geometric region identification, query-specific visual grounding, and annotated visual highlighting.

```
                      +-----------------------------+
                      |   Satellite Image (Bytes)   |
                      |   + Optional Text Query     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Satellite Preprocessor   |
                      |  - Safe Byte Decoding       |
                      |  - Validation & Metadata    |
                      |  - CLAHE Contrast Boost     |
                      |  - Spectral Indices (ExG,   |
                      |    NDWI, NDTI, Texture)     |
                      +--------------+--------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
+-------------------------------+         +-------------------------------+
|     Satellite Segmenter       |         |   Satellite Feature Detector  |
|  - Semantic Land-Cover Rules  |         |  - Linear Roads/Corridors     |
|  - Multi-feature K-Means      |         |  - Built-up Clusters          |
|  - SLIC Superpixels           |         |  - Water Bodies & Parcels     |
|  - Unified Priority Mapping   |         |  - Non-Maximum Suppression    |
+---------------+---------------+         +---------------+---------------+
                |                                         |
                +--------------------+--------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Region Identifier      |
                      |  - Polygon & Spatial Props  |
                      |  - Spectral Confidence      |
                      |  - Query Grounding Engine   |
                      |  - Natural Language Summary |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Satellite Visualizer    |
                      |  - Semi-transparent Masks   |
                      |  - Bounding Boxes & Badges  |
                      |  - Query Spotlight Banner   |
                      |  - Base64 Data URI Export   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      CVAnalysisResult       |
                      |   (JSON-Serializable Dict)  |
                      +-----------------------------+
```

---

## 2. Preprocessing Pipeline (`app/computer_vision/preprocessing.py`)

- **Safe Decoding**: Safely handles image bytes from various formats (JPEG, PNG, GeoTIFF, WebP) with Pillow and OpenCV fallback. Gracefully catches corrupt/empty inputs.
- **Validation**: Strict verification of non-empty dimensions ($H, W \ge 4$), 3 RGB channels, and non-null pixel buffers.
- **Resizing**: Standardized resizing using `cv2.INTER_AREA` for downscaling and `cv2.INTER_CUBIC` for upscaling, with optional uniform letterbox padding.
- **Normalization Options**:
  - `minmax`: Pixel values scaled to $[0.0, 1.0]$.
  - `standard`: Zero-mean, unit-variance $(\mathbf{x} - \mu) / \sigma$.
  - `imagenet`: Normalized with ImageNet standard mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
  - `sentinel2_approx`: Bottom-of-Atmosphere (BOA) surface reflectance approximation with 2%–98% percentile haze clipping.
- **Contrast Enhancement (CLAHE)**: Contrast Limited Adaptive Histogram Equalization applied to the Luminance (L) channel in LAB color space to reveal subtle terrain textures.
- **Spectral Index Extraction**:
  - **ExG (Excess Green Index)**: $2G - R - B$ (sensitive to active green vegetation and crops).
  - **VARI (Visible Atmospherically Resistant Index)**: $\frac{G - R}{G + R - B + \epsilon}$ (reduces atmospheric scattering variations).
  - **GLI (Green Leaf Index)**: $\frac{2G - R - B}{2G + R + B + \epsilon}$ (canopy leaf index).
  - **NDWI_RGB (Visible Water Index)**: $\frac{1}{2}\left[\frac{B - R}{B + R + \epsilon} + \frac{G - R}{G + R + \epsilon}\right]$ (differentiates water bodies from shadow and soil).
  - **NDTI (Normalized Difference Tillage Index)**: $\frac{R - G}{R + G + \epsilon}$ (identifies dry soil, sand, and tilled land).
  - **Texture & Brightness**: Standard deviation of local neighborhood gradients and Value channel.

---

## 3. Object & Feature Detection (`app/computer_vision/detection.py`)

- **Linear Infrastructure Detection**: Canny edge filtering + Probabilistic Hough Transform (`HoughLinesP`) to detect transport corridors, highways, and waterways.
- **Built-up & Industrial Cluster Detection**: Harris corner / keypoint density clustering coupled with morphological dilation and texture variance analysis.
- **Water Body Extraction**: Connected component analysis on NDWI-sensitive binary masks with aspect-ratio categorization (river vs sea/lake).
- **Vegetation Parcels**: Morphological contour detection on ExG-filtered masks with distinction between dark canopy (forest) and bright field (crops/pasture).
- **Non-Maximum Suppression (NMS)**: Fast IoU suppression to eliminate redundant bounding boxes.

---

## 4. Image Segmentation (`app/computer_vision/segmentation.py`)

- **Multi-feature K-Means Clustering**: Clusters pixels using a 7-dimensional feature space (LAB color + normalized spatial $(x, y)$ + ExG + NDWI) to preserve spatial and spectral cohesion.
- **SLIC Superpixels**: Compact superpixel grouping via `skimage.segmentation.slic`.
- **Semantic Rule-Based Masks**: Satellite-specific spectral thresholds producing masks for:
  - `water` (sea, lake, river)
  - `forest` (dense dark canopy)
  - `vegetation_crop` (agricultural cropland and pasture)
  - `residential` (urban settlements and housing)
  - `industrial` (bright large commercial/industrial roofs)
  - `bare_soil` (tilled soil, sand, fallow ground)
- **Unified Land-Cover Map**: Priority-based assignment merging rule-based semantic classes with K-Means residual clusters.

---

## 5. Region Identification & Spatial Properties (`app/computer_vision/region_identifier.py`)

For every segmented region, the system computes:
- `region_id`: Unique integer identifier.
- `label` & `class_name`: Assigned land-cover class.
- `area_pixels` & `area_percentage`: Pixel count and percentage of total scene area.
- `centroid`: Center of mass coordinate $(x, y)$.
- `bbox`: Enclosing `BoundingBox` (`ymin`, `xmin`, `ymax`, `xmax`, `x`, `y`, `width`, `height`).
- `mean_color_rgb`: Average RGB color.
- `spectral_indices`: Regional average of ExG, NDWI, NDTI, brightness, and texture.
- `contour_points`: Douglas-Peucker simplified polygon boundary coordinates for GIS/frontend rendering.
- `confidence`: Spectral purity score $(0.0 - 1.0)$.

---

## 6. Query-Specific Visual Grounding

When the user supplies a textual query (e.g., *"Find rivers and water bodies"*, *"Highlight residential neighborhoods"*, *"Show forests"*):
1. **Query Parsing**: Tokenizes and matches query terms against semantic class synonym dictionaries.
2. **Relevance Scoring**: Scores each region by multiplying semantic match affinity by region confidence and area relevance.
3. **Explanation Generation**: Produces a natural language explanation for why the region was grounded.
4. **Visual Spotlight**: Dims non-matching background regions by 40% while illuminating and highlighting matching regions with high-contrast borders and query header badges.

---

## 7. Visual Highlighting & Visualization (`app/computer_vision/visualization.py`)

- **Mask Overlay**: Semi-transparent colored overlays with distinct class palettes (Dodger Blue for water, Forest Green for forest, Lawn Green for crops, Dark Orange for residential, Medium Orchid for industrial, Gold for highways).
- **Bounding Boxes & Badges**: Crisp bounding boxes with dark background label chips and confidence percentages.
- **Base64 PNG Export**: Encoded directly into standard data URI format (`data:image/png;base64,...`) for web/mobile/frontend display.
- **Side-by-Side Comparison**: Helper to generate side-by-side composite views of original image vs annotated output.

---

## 8. API Contract & Backend Integration (`app/computer_vision/cv_module.py`)

The primary backend integration point is:

```python
def get_cv_analysis(image_bytes: bytes, query: Optional[str] = None) -> dict:
```

### Output Schema:
```json
{
  "status": "success",
  "image_metadata": {
    "width": 64,
    "height": 64,
    "channels": 3,
    "format": "RGB",
    "mean_rgb": [72.7, 85.9, 90.6],
    "std_rgb": [26.7, 17.1, 12.5],
    "estimated_scene_type": "water_body"
  },
  "detected_features": [
    {
      "feature_id": "water_1",
      "feature_type": "water_body",
      "label": "river",
      "confidence": 0.88,
      "bbox": {"ymin": 10, "xmin": 5, "ymax": 55, "xmax": 58, "x": 5, "y": 10, "width": 53, "height": 45},
      "properties": {"area_pixels": 612, "aspect_ratio": 1.18, "mean_ndwi": 0.221}
    }
  ],
  "detections": [
    {"ymin": 10, "xmin": 5, "ymax": 55, "xmax": 58, "x": 5, "y": 10, "width": 53, "height": 45}
  ],
  "segments": [
    {
      "region_id": 1,
      "label": "water",
      "class_name": "water",
      "confidence": 0.94,
      "area_pixels": 612,
      "area_percentage": 14.94,
      "centroid": [31.5, 32.2],
      "bbox": {"ymin": 10, "xmin": 5, "ymax": 55, "xmax": 58, "x": 5, "y": 10, "width": 53, "height": 45},
      "mean_color_rgb": [55, 78, 92],
      "spectral_indices": {"exg": 0.021, "ndwi": 0.215, "ndti": -0.175, "brightness": 0.362, "texture": 0.023},
      "contour_points": [[5, 10], [58, 10], [58, 55], [5, 55]]
    }
  ],
  "grounded_regions": [
    {
      "target_query": "Find rivers and water bodies",
      "matched_class": "water",
      "relevance_score": 0.94,
      "region": { ... },
      "explanation": "Grounded 'water' region (14.9% of scene) matching query 'Find rivers and water bodies' with 94.0% confidence."
    }
  ],
  "highlighted_image": "data:image/png;base64,iVBORw0KGgo...",
  "summary": "Scene breakdown (forest: 53.8%, vegetation_crop: 15.2%, water: 14.9%). Identified 20 key visual features. Query grounding for 'Find rivers and water bodies': highlighted matching regions.",
  "detected_regions": ["water (Area: 14.94%, Conf: 94%)", "forest (Area: 53.81%, Conf: 91%)"],
  "labels": ["bare_soil", "forest", "pasture", "residential", "river", "water"],
  "query": "Find rivers and water bodies",
  "error": null
}
```

---

## 9. How to Run

### Run Unit and Integration Tests:
```bash
pytest -v
```

### Run Demonstration Script:
```bash
python demo_cv.py
```
This runs real EuroSAT satellite images across 6 scenarios and exports highlighted visual images to `outputs/cv_demo/`.

### Run Backend Server:
```bash
uvicorn app.backend.main:app --reload
```

---

## 10. Limitations & Future Roadmap

- **Current Implementation**: Lightweight, CPU-friendly, ultra-fast (<50ms per image), deterministic, and zero-GPU dependency. Ideal for hackathons, testing, and edge devices.
- **Future Enhancements (Post-Milestone)**:
  - Integration of a fine-tuned Segment Anything Model (SAM) or Mask2Former for fine-grained sub-meter satellite segmentations.
  - CLIP/SigLIP vision-language embeddings for zero-shot open-vocabulary visual grounding.
  - Multi-band 13-channel Sentinel-2 GeoTIFF processing (NIR, SWIR, RedEdge).
