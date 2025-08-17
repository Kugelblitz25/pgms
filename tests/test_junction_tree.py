import pytest  

from src.graphs import Clique
from src.junction_tree import JTree
from src.variables import Potential, Variable, Assignment


@pytest.fixture
def sample_cliques():
    v = [Variable(i) for i in range(5)]
    c1 = Clique([v[0], v[1], v[2]])
    c2 = Clique([v[1], v[2], v[3]])
    c3 = Clique([v[2], v[3], v[4]])
    return [c1, c2, c3]


def test_junction_tree_creation(sample_cliques: list[Clique]):
    jt = JTree(sample_cliques)
    assert len(jt.cliques) == 3


def test_message_passing(sample_cliques: list[Clique]):
    jt = JTree(sample_cliques)

    # Assign potentials
    for clique in jt.cliques:
        variables = sorted(list(clique.nodes), key=lambda v: v.id)
        clique.potential = Potential.from_vars(variables)

    root = jt.cliques[0]
    jt.message_pass(root)
    assert all(len(c.messages) > 0 for c in root.neighbours)


def test_top_k_assignments(sample_cliques: list[Clique]):
    jt = JTree(sample_cliques)

    # Get top k assignments
    k = 3
    potential = jt.get_top_k(k)

    # Check if we got correct number of assignments
    assert len(potential.assignments) == k

    # Check if assignments are in descending order of probability
    for i in range(len(potential.assignments) - 1):
        assert potential.potentials[i] >= potential.potentials[i + 1]


def test_single_clique_tree():
    var = Variable(0)
    clique = Clique([var])
    jt = JTree([clique])
    
    assert len(jt.cliques) == 1
    jt.message_pass(clique)


def test_inference_with_evidence():
    var1, var2 = Variable(0), Variable(1)
    clique = Clique([var1, var2])
    clique.potential = Potential.from_vars([var1, var2], [1, 2, 3, 4])
    
    jt = JTree([clique])
    vals = Assignment({var1: 0})
    
    result = jt.inference(vals)
    assert result.sum() == 3


def test_get_top_k_with_k_greater_than_assignments():
    var = Variable(0)
    clique = Clique([var])
    clique.potential = Potential.from_vars([var], [1, 2])
    
    jt = JTree([clique])
    result = jt.get_top_k(5)
    
    assert len(result.assignments) == 2
