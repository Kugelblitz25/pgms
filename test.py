import argparse
import json
import time
from typing import Any

from src.inference import Inference


def load_test_case(file_path: str) -> list[dict[str, Any]]:
    """Loads a test case from a JSON file."""
    with open(file_path) as f:
        test_case = json.load(f)
    return test_case


def run_single_test_case(test_case: dict[str, Any], test_name: str) -> float:
    """Runs a single test case and compares the output with the expected output."""
    print(f"--- Running test: {test_name} ---")
    input_data = test_case["Input"]
    expected_output = test_case["Output"]

    start_time = time.perf_counter()

    inference = Inference(input_data)
    inference.triangulate_and_get_cliques()
    inference.get_junction_tree()
    inference.assign_potentials_to_cliques()
    z_value = inference.get_z_value()
    marginals = inference.compute_marginals()
    top_k = inference.compute_top_k()

    end_time = time.perf_counter()
    print(f"Inference time: {end_time - start_time:.4f} seconds")
    assert (
        abs(z_value - expected_output["Z_value"]) < 1e-6
    ), f"Z_value mismatch: Got {z_value}, Expected {expected_output['Z_value']}"
    print(f"✔ Z_value matches: {z_value}")
    assert len(marginals) == len(
        expected_output["Marginals"]
    ), "Marginals length mismatch."
    for i, marginal in enumerate(marginals):
        assert (
            abs(marginal[0] - expected_output["Marginals"][i][0]) < 1e-6
        ), f"Marginal value mismatch at index {i}"
    print("✔ Marginals match.")
    assert len(top_k) == len(
        expected_output["Top_k_assignments"]
    ), "Top_k_assignments length mismatch."
    for i, item in enumerate(top_k):
        assert (
            item["assignment"] == expected_output["Top_k_assignments"][i]["assignment"]
        ), f"Top_k assignment mismatch at index {i}"
        assert (
            abs(
                item["probability"]
                - expected_output["Top_k_assignments"][i]["probability"]
            )
            < 1e-6
        ), f"Top_k probability mismatch at index {i}"
    print("✔ Top K assignments match.")
    print(f"--- Test {test_name} passed ---")

    return end_time - start_time


def main():
    parser = argparse.ArgumentParser(
        description="Run inference test cases from a JSON file."
    )
    parser.add_argument("file_path", type=str, help="Path to the JSON test case file.")
    args = parser.parse_args()

    try:
        test_data = load_test_case(args.file_path)
    except FileNotFoundError:
        print(f"Error: Test case file not found at {args.file_path}")
        return

    total_time = 0.0
    for i, test_case in enumerate(test_data):
        total_time += run_single_test_case(
            test_case, f"{args.file_path} - Test Case {i+1}"
        )

    print(f"Total inference time: {total_time:.4f} seconds")


if __name__ == "__main__":
    main()
