from collections import defaultdict
from heapq import heappop, heappush

from .graphs import Clique
from .variables import Assignment, Potential


class JTree:
    """Represents a junction tree."""

    def __init__(self, cliques: list[Clique]) -> None:
        """Initializes a junction tree."""
        self.cliques = cliques
        clique_graph = self.get_graph()
        self.MST(clique_graph)
        self.infered: dict[Assignment, Potential] = {}

    def get_graph(self) -> dict[Clique, list[tuple[int, Clique]]]:
        """Creates a clique graph with weights as shared variables count."""
        clique_graph: dict[Clique, list[tuple[int, Clique]]] = defaultdict(list)

        for i in range(len(self.cliques) - 1):
            for j in range(i + 1, len(self.cliques)):
                clique1 = self.cliques[i]
                clique2 = self.cliques[j]
                shared_vars = clique1.shared(clique2)
                if shared_vars:
                    weight = len(shared_vars) * -1
                    clique_graph[clique1].append((weight, clique2))
                    clique_graph[clique2].append((weight, clique1))
        return clique_graph

    def MST(self, clique_graph: dict[Clique, list[tuple[int, Clique]]]) -> None:
        """Constructs the Maximum Spanning Tree of the clique graph."""
        start_clique = self.cliques[0]
        visited = set([start_clique])
        heap: list[tuple[int, Clique, Clique]] = []

        for weight, neighbour in clique_graph[start_clique]:
            heappush(heap, (weight, start_clique, neighbour))

        while heap and len(visited) < len(self.cliques):
            weight, c1, c2 = heappop(heap)
            visited.add(c2)
            c1.add_neighbour(c2, weight)
            c2.add_neighbour(c1, weight)

            for next_weight, neighbour in clique_graph[c2]:
                if neighbour not in visited:
                    heappush(heap, (next_weight, c2, neighbour))

    def message_pass(self, root: Clique, parent: Clique | None = None) -> None:
        """Performs message passing from leaves to root."""
        for child in root.neighbours:
            if child == parent:
                continue
            if root in child.messages:
                continue
            self.message_pass(child, root)
            separator = root.shared(child)
            msg = child.potential
            for neighbor in child.neighbours:
                if neighbor == root:
                    continue
                msg = msg * neighbor.messages[child]
            msg = msg.marginalize(list(set(child.nodes) - separator))
            child.messages[root] = msg

    def inference(self, vals: Assignment) -> Potential:
        """Performs inference given evidence."""

        def pass_evidence(root: Clique, parent: Clique | None = None) -> Potential:
            var_vals = Assignment({k: vals[k] for k in root.nodes if k in vals})
            tot_potential = root.potential.get_potential(var_vals)
            assert (
                tot_potential is not None
            ), "get_potential should not return None here"
            for child in root.neighbours:
                if child == parent:
                    continue
                tot_potential *= pass_evidence(child, root)
            return tot_potential

        return pass_evidence(self.cliques[0])

    def get_top_k(self, k: int) -> Potential:
        """Gets the top k most probable assignments."""

        def bfs(root: Clique, parent: Clique | None = None) -> Potential:
            top_k_pots = root.potential
            for child in root.neighbours:
                if child == parent:
                    continue
                separator = root.shared(child)
                rest_vars = list(set(child.nodes) - separator)
                top_k_pots *= bfs(child, root).max_marginalize(rest_vars, k)
            return top_k_pots

        return bfs(self.cliques[0]).top(k)

    def __str__(self) -> str:
        return ("\n" + "=" * 10 + "\n").join(map(str, self.cliques))
