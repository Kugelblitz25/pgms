from collections import defaultdict
from itertools import product
from typing import Optional


class Variable:
    """Represents a variable in the graphical model."""

    def __init__(self, id: int, values: list[int] = [0, 1]) -> None:
        """Initializes a Variable."""
        self.id = id
        self.neighbours: set["Variable"] = set()
        self.values: list[int] = values
        self.marginal: Potential = Potential.from_vars([self])

    def add_neighbour(self, node: "Variable") -> None:
        """Adds a neighbour to the variable."""
        self.neighbours.add(node)

    def is_neighbour(self, node: "Variable") -> bool:
        """Checks if a variable is a neighbour."""
        return node in self.neighbours

    @property
    def degree(self) -> int:
        """Returns the degree of the variable."""
        return len(self.neighbours)

    def __lt__(self, other: "Variable") -> bool:
        return self.id < other.id

    def __str__(self) -> str:
        return f"{self.id} -> {[neighbour.id for neighbour in self.neighbours]}"


class Assignment(dict[Variable, int]):
    """Represents an assignment of values to a set of variables."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Initializes an Assignment."""
        super().__init__(*args, **kwargs)
        self.variables = set(self.keys())
        self.id = ",".join(f"(x{var.id}:{self[var]})" for var in self.variables)

    def __setitem__(self, key: Variable, value: int) -> None:
        raise TypeError("Assignment is immutable")

    def __delitem__(self, key: Variable) -> None:
        raise TypeError("Assignment is immutable")

    def update(self, *args, **kwargs) -> None:  # type: ignore
        raise TypeError("Assignment is immutable")

    def copy(self) -> "Assignment":
        return Assignment(super().copy())

    def __le__(self, other: "Assignment") -> bool:
        return set(self.items()) <= set(other.items())

    def __lt__(self, other: "Assignment") -> bool:
        return self.__le__(other)

    def __str__(self) -> str:
        return self.id


class Potential:
    """Represents a potential function over a set of variables."""

    def __init__(
        self, assignments: list[Assignment], potentials: Optional[list[int]] = None
    ) -> None:
        """Initializes a Potential."""
        self.assignments = assignments
        self.potentials = potentials or [1] * len(self.assignments)
        assert len(self.assignments) == len(
            self.potentials
        ), "Potentials and Assignments do not match."
        self.variables = set(self.assignments[0].variables)
        for assignment in self.assignments:
            assert self.variables == set(
                assignment.variables
            ), "Assignments do not contain same variables"
        assert all(
            isinstance(pt, int) for pt in self.potentials
        ), "All potentials must be integers."
        self.id2pot: dict[str, int] = {
            combo.id: potential
            for combo, potential in zip(self.assignments, self.potentials)
        }

    @staticmethod
    def from_vars(
        vars: list[Variable], potentials: Optional[list[int]] = None
    ) -> "Potential":
        """Creates a Potential from a list of variables."""
        combos = product(*[var.values for var in vars])
        assignments = [
            Assignment({var: val for var, val in zip(vars, combo)}) for combo in combos
        ]
        return Potential(assignments, potentials)

    def get_potential(self, vals: Assignment) -> Optional["Potential"]:
        """Gets a new potential conditioned on some values."""
        if not vals.variables <= self.variables:
            raise ValueError("Unkown variables.")
        assignments: list[Assignment] = []
        potentials: list[int] = []
        for assignment, potential in zip(self.assignments, self.potentials):
            if vals <= assignment:
                assignments.append(assignment)
                potentials.append(potential)

        if not assignments:
            return None

        return Potential(assignments, potentials)

    def sum(self) -> int:
        """Sums the potentials."""
        return sum(self.potentials)

    def marginalize(self, nodes: list[Variable]) -> "Potential":
        """Marginalizes out a set of variables by summing."""
        if not set(nodes) <= self.variables:
            raise ValueError("Marginalizing over unkown variables.")

        rem_vars = list(self.variables - set(nodes))
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)

        for i, assignment in enumerate(self.assignments):
            key = tuple(assignment[var] for var in rem_vars)
            groups[key].append(i)

        assignments: list[Assignment] = []
        potentials: list[int] = []

        for key, idxs in groups.items():
            assignments.append(
                Assignment({var: val for var, val in zip(rem_vars, key)})
            )
            potentials.append(sum([self.potentials[i] for i in idxs]))

        return Potential(assignments, potentials)

    def max_marginalize(self, nodes: list[Variable], k: int = 1) -> "Potential":
        """Marginalizes out a set of variables by taking the max."""
        if not set(nodes) <= self.variables:
            raise ValueError("Marginalizing over unkown variables.")

        rem_vars = list(self.variables - set(nodes))
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)

        for i, assignment in enumerate(self.assignments):
            key = tuple(assignment[var] for var in rem_vars)
            groups[key].append(i)

        assignments: list[Assignment] = []
        potentials: list[int] = []

        for idxs in groups.values():
            top_k_idxs = sorted(idxs, key=lambda idx: -self.potentials[idx])[:k]
            assignments.extend([self.assignments[i] for i in top_k_idxs])
            potentials.extend([self.potentials[i] for i in top_k_idxs])

        return Potential(assignments, potentials)

    def is_part(self, nodes: list[Variable]) -> bool:
        """Checks if the potential's variables are a subset of the given nodes."""
        return self.variables <= set(nodes)

    def top(self, k: int = 1) -> "Potential":
        """Returns the top k assignments with the highest potentials."""
        ids = sorted(range(len(self.potentials)), key=lambda cb: -self.potentials[cb])
        top_k_ids = ids[: min(k, len(self.assignments))]
        potentials = [self.potentials[id] for id in top_k_ids]
        assignments = [self.assignments[id] for id in top_k_ids]
        return Potential(assignments, potentials)

    def __mul__(self, other: "Potential") -> "Potential":
        """Multiplies two potentials."""
        shared_vars = list(self.variables & other.variables)

        assignments: list[Assignment] = []
        potentials: list[int] = []

        if not shared_vars:
            for ass1, pot1 in zip(self.assignments, self.potentials):
                for ass2, pot2 in zip(other.assignments, other.potentials):
                    assignments.append(Assignment({**ass1, **ass2}))
                    potentials.append(pot1 * pot2)

            return Potential(assignments, potentials)

        self_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        other_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)

        for i, assignment in enumerate(self.assignments):
            key = tuple(assignment[var] for var in shared_vars)
            self_groups[key].append(i)

        for i, assignment in enumerate(other.assignments):
            key = tuple(assignment[var] for var in shared_vars)
            other_groups[key].append(i)

        for key, idxs in self_groups.items():
            if key not in other_groups:
                continue
            combs = product(idxs, other_groups[key])
            for idx1, idx2 in combs:
                new_assignment = Assignment(
                    {**self.assignments[idx1], **other.assignments[idx2]}
                )
                assignments.append(new_assignment)
                potentials.append(self.potentials[idx1] * other.potentials[idx2])

        return Potential(assignments, potentials)

    def __lt__(self, other: "Potential") -> bool:
        return self.top().potentials[0] < other.top().potentials[0]

    def __str__(self) -> str:
        header = f"{' | '.join([f'x{vars.id}' for vars in self.variables])} | Potential"
        rows = [header, "-" * len(header)]
        for combo, val in zip(self.assignments, self.potentials):
            rows.append(f"{' | '.join(map(str, combo.values()))} | {val}")
        return "\n".join(rows)
