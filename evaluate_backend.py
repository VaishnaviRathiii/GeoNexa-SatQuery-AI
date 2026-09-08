import csv
import os
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
TEST_FILE = "data/test_questions.csv"
IMAGE_DIR = "data/sample_images"
RESULT_FILE = "data/evaluation_results.csv"


def evaluate_tests():
    results = []

    with open(TEST_FILE, "r", encoding="utf-8", newline="") as f:
        tests = list(csv.DictReader(f))

    print(f"Loaded {len(tests)} test cases.")
    print("-" * 70)

    for test in tests:
        test_id = test["test_id"]
        image_name = test["image_name"]
        question = test["query"]
        expected = test["expected_result"]

        image_path = os.path.join(IMAGE_DIR, image_name)

        print(f"Running {test_id}: {image_name}")

        if not os.path.exists(image_path):
            results.append({
                "test_id": test_id,
                "image_name": image_name,
                "query": question,
                "expected_result": expected,
                "actual_result": "",
                "status": "ERROR",
                "response_time_seconds": "",
                "notes": "Image file not found."
            })
            continue

        start = time.perf_counter()

        try:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    f"{BASE_URL}/analyze",
                    files={
                        "image": (
                            image_name,
                            image_file,
                            "image/jpeg"
                        )
                    },
                    data={
                        "question": question
                    },
                    timeout=120
                )

            elapsed = round(time.perf_counter() - start, 3)

            if response.status_code != 200:
                results.append({
                    "test_id": test_id,
                    "image_name": image_name,
                    "query": question,
                    "expected_result": expected,
                    "actual_result": "",
                    "status": "ERROR",
                    "response_time_seconds": elapsed,
                    "notes": f"HTTP {response.status_code}: {response.text[:200]}"
                })
                print(f"  ERROR - HTTP {response.status_code}")
                continue

            data = response.json()
            actual = data.get("answer", "")

            # Basic evaluation:
            # PASS only when the expected class appears in the answer.
            if expected.lower() in actual.lower():
                status = "PASS"
                notes = "Expected class found in backend AI answer."
            else:
                status = "FAIL"
                notes = "Expected class not found in backend AI answer."

            results.append({
                "test_id": test_id,
                "image_name": image_name,
                "query": question,
                "expected_result": expected,
                "actual_result": actual,
                "status": status,
                "response_time_seconds": elapsed,
                "notes": notes
            })

            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            print(f"  Status:   {status}")
            print(f"  Time:     {elapsed}s")

        except requests.RequestException as e:
            elapsed = round(time.perf_counter() - start, 3)

            results.append({
                "test_id": test_id,
                "image_name": image_name,
                "query": question,
                "expected_result": expected,
                "actual_result": "",
                "status": "ERROR",
                "response_time_seconds": elapsed,
                "notes": f"Request failed: {e}"
            })

            print(f"  ERROR: {e}")

        print("-" * 70)

    # Rewrite evaluation results with the latest complete run.
    fieldnames = [
        "test_id",
        "image_name",
        "query",
        "expected_result",
        "actual_result",
        "status",
        "response_time_seconds",
        "notes"
    ]

    with open(RESULT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    passed = sum(r["status"] == "PASS" for r in results)
    failed = sum(r["status"] == "FAIL" for r in results)
    errors = sum(r["status"] == "ERROR" for r in results)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total : {len(results)}")
    print(f"PASS  : {passed}")
    print(f"FAIL  : {failed}")
    print(f"ERROR : {errors}")
    print("========================================")


if __name__ == "__main__":
    evaluate_tests()