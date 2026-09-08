from pathlib import Path
import shutil
import csv

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

# Actual extracted EuroSAT location
DATASET_DIR = DATA_DIR / "downloads" / "EuroSAT_RGB" / "EuroSAT_RGB"

SAMPLE_DIR = DATA_DIR / "sample_images"
METADATA_FILE = DATA_DIR / "image_metadata.csv"

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

SAMPLES_PER_CLASS = 2

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

metadata_rows = []

for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    images = sorted(class_dir.glob("*.jpg"))

    if len(images) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Not enough images in {class_name}. "
            f"Found {len(images)}, required {SAMPLES_PER_CLASS}."
        )

    selected_images = images[:SAMPLES_PER_CLASS]

    for index, source_image in enumerate(selected_images, start=1):

        destination_name = (
            f"{class_name.lower()}_{index:02d}.jpg"
        )

        destination = SAMPLE_DIR / destination_name

        shutil.copy2(source_image, destination)

        metadata_rows.append({
            "image_name": destination_name,
            "dataset_id": "DS003",
            "scene_type": class_name,
            "location": "EuroSAT dataset",
            "acquisition_date": "Dataset metadata",
            "resolution": "64x64 pixels",
            "image_type": "RGB",
            "source": "EuroSAT",
            "license_or_access": "See EuroSAT dataset terms",
            "notes": f"Sample {index} from {class_name} class",
        })


fieldnames = [
    "image_name",
    "dataset_id",
    "scene_type",
    "location",
    "acquisition_date",
    "resolution",
    "image_type",
    "source",
    "license_or_access",
    "notes",
]

with METADATA_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(metadata_rows)


print("======================================")
print("EuroSAT sample setup complete")
print("======================================")
print(f"Classes: {len(CLASSES)}")
print(f"Images per class: {SAMPLES_PER_CLASS}")
print(f"Total sample images: {len(metadata_rows)}")
print(f"Sample directory: {SAMPLE_DIR}")
print(f"Metadata file: {METADATA_FILE}")