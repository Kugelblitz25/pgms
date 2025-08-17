from .variables import Potential, Variable


class UGraph:
    """Represents an undirected graph."""

    def __init__(self, nodes: list[Variable]) -> None:
        """Initializes an undirected graph."""
        self.nodes = nodes

    def add_edge(self, node1: Variable, node2: Variable) -> None:
        """Adds an edge between two nodes."""
        node1.add_neighbour(node2)
        node2.add_neighbour(node1)

    def get_MCS_order(self) -> list[Variable]:
        """Gets the Maximum Cardinality Search order of the nodes."""
        ordered: list[Variable] = []
        top_node = max(self.nodes, key=lambda node: node.degree)
        rem_nodes = set(self.nodes)
        ordered.append(top_node)
        ordered_set = {top_node}
        rem_nodes.remove(top_node)
        while rem_nodes:
            top_node = sorted(
                rem_nodes,
                key=lambda node: len(node.neighbours & ordered_set),
            )[-1]
            ordered.append(top_node)
            ordered_set.add(top_node)
            rem_nodes.remove(top_node)
        return ordered

    def __str__(self) -> str:
        return "\n".join(map(str, self.nodes))


class Clique(UGraph):
    """Represents a clique in the graph."""

    def __init__(self, nodes: list[Variable]) -> None:
        """Initializes a clique."""
        super().__init__(nodes)
        self.neighbours: dict["Clique", int] = dict()
        self.messages: dict["Clique", Potential] = dict()
        self.id = ",".join([f"x{node.id}" for node in self.nodes])
        self.potential = Potential.from_vars(self.nodes)

    def add_potential(self, potentials: Potential) -> None:
        """Adds a potential to the clique."""
        self.potential *= potentials

    def includes(self, nodes: set[Variable]) -> bool:
        """Checks if the clique includes a set of nodes."""
        return nodes <= set(self.nodes)

    def shared(self, clique: "Clique") -> set[Variable]:
        """Returns the shared variables with another clique."""
        return set(self.nodes) & set(clique.nodes)

    def add_neighbour(self, clique: "Clique", weight: int) -> None:
        """Adds a neighbour clique."""
        self.neighbours[clique] = weight

    def __str__(self) -> str:
        output: list[str] = []
        for neighbour, weight in self.neighbours.items():
            output.append(f"-{weight}->({neighbour.id})")
        return f"({self.id})\n" + "\n".join(output)

    def __lt__(self, other: "Clique") -> bool:
        return self.id < other.id
