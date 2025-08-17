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
