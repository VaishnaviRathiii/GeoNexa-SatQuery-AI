"""
GeoNexa Computer Vision Subsystem Demonstration Script.

Runs the full end-to-end satellite image analysis pipeline on sample EuroSAT
images, executes query-specific grounding, and exports highlighted visual outputs.

Usage:
    python demo_cv.py
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image

from app.computer_vision.cv_module import get_cv_analysis
from app.computer_vision.utils import decode_base64_to_image

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cv_demo"


def main():
    print("=" * 65)
    print("GeoNexa Computer Vision Subsystem (M2 - Shubham) Demonstration")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    demo_cases = [
        ("river_01.jpg", "Find rivers and water bodies"),
        ("forest_01.jpg", "Locate forests and dense vegetation"),
        ("residential_01.jpg", "Highlight residential housing and neighborhoods"),
        ("industrial_01.jpg", "Identify industrial complexes and factories"),
        ("highway_01.jpg", "Detect highway transport corridors and roads"),
        ("annualcrop_01.jpg", "Analyze agricultural cropland and fields"),
    ]

    for img_name, query in demo_cases:
        img_path = DATA_DIR / img_name
        if not img_path.exists():
            print(f"Skipping {img_name} (file not found)")
            continue

        print(f"\n[DEMO] Processing: {img_name}")
        print(f"       Query: '{query}'")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Run CV analysis with query grounding
        result = get_cv_analysis(img_bytes, query=query)

        print(f"       Status: {result['status']}")
        print(f"       Summary: {result['summary']}")
        print(f"       Labels: {', '.join(result['labels'][:5])}")
        print(f"       Detected Features: {len(result['detected_features'])}")
        print(f"       Grounded Regions: {len(result['grounded_regions'])}")

        # Save output visual artifact
        if result.get("highlighted_image"):
            highlighted_np = decode_base64_to_image(result["highlighted_image"])
            out_filename = OUTPUT_DIR / f"highlighted_{img_path.stem}.png"
            Image.fromarray(highlighted_np).save(out_filename)
            print(f"       Saved Visual Artifact: {out_filename.relative_to(PROJECT_ROOT)}")

        # Save json result
        json_filename = OUTPUT_DIR / f"result_{img_path.stem}.json"
        with open(json_filename, "w", encoding="utf-8") as jf:
            # Save lightweight json without giant base64 for inspection
            json_copy = dict(result)
            json_copy["highlighted_image"] = f"<Base64 PNG Image - {len(result['highlighted_image'])} chars>"
            json.dump(json_copy, jf, indent=2)

    print("\n" + "=" * 65)
    print(f"Demonstration completed successfully. Outputs saved in: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
