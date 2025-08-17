from typing import Any

from .graphs import Clique, UGraph
from .junction_tree import JTree
from .variables import Potential, Variable


class Inference:
    def __init__(self, data: dict[str, Any]) -> None:
        """
        Initialize the Inference class with the input data.
        """
        num_nodes = data["VariablesCount"]
        self.nodes = [Variable(i) for i in range(num_nodes)]
        self.graph = UGraph(self.nodes)
        self.potentials: set[Potential] = set()
        self.maximal_cliques: list[Clique] = []
        self.z: int = 0
        cliques = data["Cliques and Potentials"]
        self.k = data.get("k value (in top k)", 1)
        for clique_data in cliques:
            nodes: list[Variable] = [self.nodes[id] for id in clique_data["cliques"]]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    self.graph.add_edge(nodes[i], nodes[j])
            self.potentials.add(Potential.from_vars(nodes, clique_data["potentials"]))

    def triangulate_and_get_cliques(self) -> None:
        """
        Triangulate the undirected graph and extract the maximal cliques.
        """
        MCS_order = self.graph.get_MCS_order()
        while MCS_order:
            cur_node = MCS_order.pop()
            left_neighbours = list(cur_node.neighbours & set(MCS_order))
            for i in range(len(left_neighbours) - 1):
                for j in range(i + 1, len(left_neighbours)):
                    if left_neighbours[i].is_neighbour(left_neighbours[j]):
                        continue
                    self.graph.add_edge(left_neighbours[i], left_neighbours[j])

        MCS_order = self.graph.get_MCS_order()

        while MCS_order:
            node = MCS_order.pop()
            candidate_clique = {node} | {
                neighbour for neighbour in node.neighbours if neighbour in MCS_order
            }

            if not any(
                clique.includes(candidate_clique) for clique in self.maximal_cliques
            ):
                self.maximal_cliques.append(Clique(list(candidate_clique)))

    def get_junction_tree(self) -> None:
        """
        Construct the junction tree from the maximal cliques.
        """
        self.junction_tree = JTree(self.maximal_cliques)

    def assign_potentials_to_cliques(self) -> None:
        """
        Assign potentials to the cliques in the junction tree.
        """
        for potential in self.potentials:
            for clique in self.junction_tree.cliques:
                if potential.variables.issubset(clique.nodes):
                    clique.add_potential(potential)
                    break

    def get_z_value(self) -> int:
        """
        Compute the partition function (Z value) of the graphical model.
        """
        root = self.junction_tree.cliques[0]
        self.junction_tree.message_pass(root)

        tot_potential = root.potential
        for neighbour in root.neighbours:
            tot_potential *= neighbour.messages[root]
        self.z = tot_potential.marginalize(root.nodes).potentials[0]
        return self.z

    def compute_marginals(self) -> list[list[float]]:
        """
        Compute the marginal probabilities for all variables in the graphical model.
        """
        marginals: list[list[float]] = []
        for node in self.nodes:
            root = None
            for clique in self.junction_tree.cliques:
                if clique.includes({node}):
                    root = clique
                    break

            if not root:
                raise ValueError(f"Node {node.id} not found in any clique.")

            self.junction_tree.message_pass(root)
            tot_potential = root.potential
            for neighbour in root.neighbours:
                tot_potential *= neighbour.messages[root]
            rest_nodes = list(set(root.nodes) - {node})
            marginalized = tot_potential.marginalize(rest_nodes)
            potentials = marginalized.potentials
            marginals.append(
                [
                    potentials[0] / self.z,
                    potentials[1] / self.z,
                ]
            )
        self.marginals = marginals
        return marginals

    def compute_top_k(self) -> list[dict[str, Any]]:
        """
        Compute the top-k most probable assignments in the graphical model.
        """
        top_k_pots = self.junction_tree.get_top_k(self.k)
        print(top_k_pots)

        results: list[dict[str, Any]] = []
        for assignment, pot in zip(top_k_pots.assignments, top_k_pots.potentials):
            results.append(
                {
                    "assignment": [assignment[node] for node in self.nodes],
                    "probability": pot / self.z,
                }
            )
        return results
