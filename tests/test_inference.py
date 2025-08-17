import json
from typing import Any

import pytest

from src.inference import Inference


def load_test_cases():
    with open("testcases/TestCases.json") as f:
        test_cases = json.load(f)
    return test_cases


@pytest.mark.parametrize("test_case", load_test_cases())
def test_inference_with_test_cases(test_case: dict[str, Any]):
    input_data = test_case["Input"]
    expected_output = test_case["Output"]

    inference = Inference(input_data)

    inference.triangulate_and_get_cliques()
    assert (
        len(inference.maximal_cliques) > 0
    ), "Triangulation should produce maximal cliques."

    inference.get_junction_tree()
    assert inference.junction_tree is not None, "Junction tree should be created."
    assert (
        len(inference.junction_tree.cliques) > 0
    ), "Junction tree should have cliques."

    inference.assign_potentials_to_cliques()

    z_value = inference.get_z_value()
    assert z_value == expected_output["Z_value"]

    marginals = inference.compute_marginals()
    assert len(marginals) == len(expected_output["Marginals"])
    for i, marginal in enumerate(marginals):
        assert marginal == pytest.approx(expected_output["Marginals"][i])  # type: ignore

    top_k = inference.compute_top_k()
    assert len(top_k) == len(expected_output["Top_k_assignments"])
    for i, item in enumerate(top_k):
        assert (
            item["assignment"] == expected_output["Top_k_assignments"][i]["assignment"]
        )
        assert item["probability"] == pytest.approx(
            expected_output["Top_k_assignments"][i]["probability"]
        )  # type: ignore
