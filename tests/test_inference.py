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
        assert item["probability"] == pytest.approx( # type: ignore
            expected_output["Top_k_assignments"][i]["probability"]
        )  


def test_inference_no_k_value():
    data: dict[str, Any] = {
        "VariablesCount": 2,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [1, 2, 3, 4]
            }
        ]
    }
    
    inference = Inference(data)
    assert inference.k == 1


def test_compute_marginals_node_not_in_clique():
    data: dict[str, Any] = {
        "VariablesCount": 3,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [1, 2, 3, 4]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()

    with pytest.raises(ValueError, match="disconnected"):
        inference.get_junction_tree()


def test_triangulation_with_complete_graph():
    data: dict[str, Any] = {
        "VariablesCount": 3,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1, 2],
                "potentials": [1, 2, 3, 4, 5, 6, 7, 8]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()
    
    assert len(inference.maximal_cliques) >= 1


def test_multiple_potentials_same_variables():
    data: dict[str, Any] = {
        "VariablesCount": 2,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [1, 2, 3, 4]
            },
            {
                "cliques": [0, 1],
                "potentials": [2, 3, 4, 5]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()
    inference.get_junction_tree()
    inference.assign_potentials_to_cliques()
    
    z_value = inference.get_z_value()
    assert z_value > 0


def test_disconnected_components():
    data: dict[str, Any] = {
        "VariablesCount": 4,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [1, 2, 3, 4]
            },
            {
                "cliques": [2, 3],
                "potentials": [1, 2, 3, 4]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()
    with pytest.raises(ValueError, match="disconnected"):
        inference.get_junction_tree()


def test_large_k_value():
    data: dict[str, Any] = {
        "VariablesCount": 2,
        "k value (in top k)": 100,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [1, 2, 3, 4]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()
    inference.get_junction_tree()
    inference.assign_potentials_to_cliques()
    inference.get_z_value()
    
    top_k = inference.compute_top_k()
    assert len(top_k) <= 4


def test_zero_potentials():
    data: dict[str, Any] = {
        "VariablesCount": 2,
        "Cliques and Potentials": [
            {
                "cliques": [0, 1],
                "potentials": [0, 0, 1, 2]
            }
        ]
    }
    
    inference = Inference(data)
    inference.triangulate_and_get_cliques()
    inference.get_junction_tree()
    inference.assign_potentials_to_cliques()
    
    z_value = inference.get_z_value()
    assert z_value == 3