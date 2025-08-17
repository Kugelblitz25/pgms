import pytest

from src.graphs import Clique, UGraph
from src.variables import Variable


@pytest.fixture
def sample_graph():
    nodes = [Variable(i) for i in range(4)]
    graph = UGraph(nodes)
    graph.add_edge(nodes[0], nodes[1])
    graph.add_edge(nodes[1], nodes[2])
    graph.add_edge(nodes[2], nodes[3])
    graph.add_edge(nodes[3], nodes[0])
    return graph


def test_graph_creation(sample_graph: UGraph):
    assert len(sample_graph.nodes) == 4


def test_add_edge(sample_graph: UGraph):
    nodes = list(sample_graph.nodes)
    assert nodes[1] in nodes[0].neighbours
    assert nodes[0] in nodes[1].neighbours


def test_get_mcs_order(sample_graph: UGraph):
    mcs_order = sample_graph.get_MCS_order()
    assert len(mcs_order) == 4


def test_clique_creation():
    nodes = [Variable(i) for i in range(3)]
    clique = Clique(nodes)
    assert len(clique.nodes) == 3


def test_empty_graph():
    with pytest.raises(AssertionError, match="at least one node"):
        _ = UGraph([])


def test_single_node_graph():
    var = Variable(0)
    graph = UGraph([var])
    mcs_order = graph.get_MCS_order()
    assert mcs_order == [var]


def test_clique_shared_variables():
    var1, var2, var3 = Variable(0), Variable(1), Variable(2)
    clique1 = Clique([var1, var2])
    clique2 = Clique([var2, var3])

    shared = clique1.shared(clique2)
    assert shared == {var2}


def test_clique_includes():
    var1, var2, var3 = Variable(0), Variable(1), Variable(2)
    clique = Clique([var1, var2, var3])

    assert clique.includes({var1, var2})
    assert not clique.includes({var1, Variable(4)})


def test_clique_comparison():
    var1, var2 = Variable(0), Variable(1)
    clique1 = Clique([var1])
    clique2 = Clique([var2])

    assert (clique1 < clique2) or (clique2 < clique1)
