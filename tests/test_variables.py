from itertools import product

from src.variables import Assignment, Potential, Variable


def test_variable_creation():
    var = Variable(0)
    assert var.id == 0
    assert var.neighbours == set()


def test_potential_creation():
    var1 = Variable(0)
    var2 = Variable(1)
    assignments = [
        Assignment(dict(zip([var1, var2], values)))
        for values in product([0, 1], [0, 1])
    ]
    pot = Potential(assignments, [1, 2, 3, 4])
    assert len(pot.variables) == 2
    assert len(pot.potentials) == 4


def test_potential_multiplication():
    var1 = Variable(0)
    var2 = Variable(1)
    ass1 = [Assignment({var1: 0}), Assignment({var1: 1})]
    ass2 = [Assignment({var2: 0}), Assignment({var2: 1})]
    pot1 = Potential(ass1, [2, 3])
    pot2 = Potential(ass2, [4, 5])
    pot3 = pot1 * pot2
    assert len(pot3.variables) == 2
    assert pot3.potentials == [8, 10, 12, 15]


def test_potential_marginalization():
    var1 = Variable(0)
    var2 = Variable(1)
    assignments = [
        Assignment(dict(zip([var1, var2], values)))
        for values in product([0, 1], [0, 1])
    ]
    pot = Potential(assignments, [1, 2, 3, 4])
    marginal_pot = pot.marginalize([var1])
    assert len(marginal_pot.variables) == 1
    assert marginal_pot.potentials == [4, 6]


def test_potential_max_marginalization():
    var1 = Variable(0)
    var2 = Variable(1)
    assignments = [
        Assignment(dict(zip([var1, var2], values)))
        for values in product([0, 1], [0, 1])
    ]
    pot = Potential(assignments, [1, 2, 3, 4])
    max_marginal = pot.max_marginalize([var2])
    assert len(max_marginal.variables) == 2
    assert max_marginal.potentials == [2, 4]
