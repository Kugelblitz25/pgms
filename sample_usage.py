from typing import Any

from src.inference import Inference


def sample_usage() -> None:
    # Create a sample graph
    # 0 -- 1
    # |    |
    # 3 -- 2

    data: dict[str, Any] = {
        "VariablesCount": 4,
        "Cliques and Potentials": [
            {"cliques": [0, 1], "potentials": [1, 2, 3, 4]},
            {"cliques": [1, 2], "potentials": [5, 6, 7, 8]},
            {"cliques": [2, 3], "potentials": [9, 10, 11, 12]},
            {"cliques": [3, 0], "potentials": [13, 14, 15, 16]},
        ],
        "k value (in top k)": 2,
    }

    inference = Inference(data)

    print("Original Graph:")
    print(inference.graph)

    inference.triangulate_and_get_cliques()
    print("\nMaximal Cliques after triangulation:")
    for clique in inference.maximal_cliques:
        print([var.id for var in clique.nodes])

    inference.get_junction_tree()
    print("\nJunction Tree:")
    print(inference.junction_tree)

    inference.assign_potentials_to_cliques()

    z = inference.get_z_value()
    print(f"\nPartition function (Z): {z}")

    marginals = inference.compute_marginals()
    print("\nMarginal probabilities:")
    for i, m in enumerate(marginals):
        print(f"  Variable {i}: {m}")

    top_k = inference.compute_top_k()
    print("\nTop-k assignments:")
    for item in top_k:
        print(
            f"  Assignment: {item['assignment']}, Probability: {item['probability']:.4f}"
        )


if __name__ == "__main__":
    sample_usage()
