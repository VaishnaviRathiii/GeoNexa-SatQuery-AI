from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_FILE = DATA_DIR / "test_questions.csv"
RESULTS_FILE = DATA_DIR / "evaluation_results.csv"
IMAGE_DIR = DATA_DIR / "sample_images"


def test_data_directory_exists():
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()


def test_sample_images_directory_exists():
    assert IMAGE_DIR.exists()
    assert IMAGE_DIR.is_dir()


def test_questions_file_exists():
    assert QUESTIONS_FILE.exists()


def test_questions_file_is_valid():
    with QUESTIONS_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) > 0

    required_columns = {
        "test_id",
        "image_name",
        "query",
        "expected_result",
        "category",
    }

    assert required_columns.issubset(reader.fieldnames)


def test_question_ids_are_unique():
    with QUESTIONS_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        ids = [row["test_id"] for row in reader]

    assert len(ids) == len(set(ids))


def test_evaluation_results_file_exists():
    assert RESULTS_FILE.exists()


def test_evaluation_results_headers():
    with RESULTS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "test_id",
            "image_name",
            "query",
            "expected_result",
            "actual_result",
            "status",
            "response_time_seconds",
            "notes",
        }

        assert required_columns.issubset(reader.fieldnames)