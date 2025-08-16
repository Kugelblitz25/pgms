# Probabilistic Graphical Model Inference

This project implements algorithms for inference in probabilistic graphical models (PGMs). Specifically, it uses the junction tree algorithm to perform tasks like calculating partition functions (Z-value), computing marginals, and finding the most probable assignments (top-k).

## Features

- **Undirected Graphs**: Representation of undirected graphical models.
- **Triangulation**: Converts a graph to a chordal graph using Maximum Cardinality Search (MCS).
- **Junction Tree**: Constructs a junction tree from the maximal cliques of a triangulated graph.
- **Inference**:
  - Calculate the partition function (Z-value).
  - Compute marginal probabilities for each variable.
  - Find the top-k most probable assignments.

## File Structure

- `variables.py`: Defines `Variable`, `Assignment`, and `Potential` classes to represent the components of a graphical model.
- `graphs.py`: Defines `UGraph` for the graph structure and `Clique`.
- `junction_tree.py`: Implements the `JTree` class, including construction and message passing.
- `test.py`: Script to run test cases from the `testcases` folder.
- `sample_usage.py`: A script demonstrating the usage of the implemented modules.
- `testcases/`: Directory containing sample test cases in JSON format.

## Usage

To run the inference on the provided test cases, you can use `test.py`.

```bash
python test.py
```
