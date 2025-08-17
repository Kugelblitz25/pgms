from itertools import product

import pytest

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


def test_variable_with_custom_values():
    var = Variable(0, [0, 1, 2])
    assert var.values == [0, 1, 2]


def test_assignment_immutability():
    var = Variable(0)
    assignment = Assignment({var: 1})

    with pytest.raises(TypeError):
        assignment[var] = 2

    with pytest.raises(TypeError):
        del assignment[var]

    with pytest.raises(TypeError):
        assignment.update({var: 2})  # type: ignore


def test_assignment_comparison():
    var1, var2 = Variable(0), Variable(1)
    ass1 = Assignment({var1: 0, var2: 1})
    ass2 = Assignment({var1: 0})

    assert ass2 <= ass1
    assert not ass1 <= ass2


def test_potential_error_cases():
    var1, var2 = Variable(0), Variable(1)

    assignments = [Assignment({var1: 0}), Assignment({var2: 0})]
    with pytest.raises(AssertionError, match="same variables"):
        Potential(assignments, [1, 1])

    assignments = [Assignment({var1: 0}), Assignment({var1: 1})]
    with pytest.raises(AssertionError, match="integers"):
        Potential(assignments, [1.5, 2.5])  # type: ignore


def test_potential_get_potential_unknown_vars():
    var1 = Variable(0)
    var2 = Variable(1)
    pot = Potential.from_vars([var1])
    vals = Assignment({var2: 0})

    with pytest.raises(ValueError, match="Unkown variables"):
        pot.get_potential(vals)


def test_potential_get_potential_no_match():
    var = Variable(0)
    pot = Potential([Assignment({var: 0})], [5])
    vals = Assignment({var: 1})

    result = pot.get_potential(vals)
    assert result is None


def test_potential_marginalize_unknown_vars():
    var1 = Variable(0)
    var2 = Variable(1)
    pot = Potential.from_vars([var1])

    with pytest.raises(ValueError, match="unkown variables"):
        pot.marginalize([var2])


def test_potential_max_marginalize_unknown_vars():
    var1 = Variable(0)
    var2 = Variable(1)
    pot = Potential.from_vars([var1])

    with pytest.raises(ValueError, match="unkown variables"):
        pot.max_marginalize([var2])


def test_potential_multiplication_no_shared_vars():
    var1, var2 = Variable(0), Variable(1)
    pot1 = Potential.from_vars([var1], [2, 3])
    pot2 = Potential.from_vars([var2], [4, 5])

    result = pot1 * pot2
    assert len(result.assignments) == 4
    assert result.potentials == [8, 10, 12, 15]


def test_potential_multiplication_with_shared_vars():
    var1, var2 = Variable(0), Variable(1)
    pot1 = Potential.from_vars([var1, var2], [1, 2, 3, 4])
    pot2 = Potential.from_vars([var2], [2, 3])

    result = pot1 * pot2
    expected_potentials = [2, 6, 6, 12]
    assert result.potentials == expected_potentials


def test_potential_top_k_more_than_available():
    var = Variable(0)
    pot = Potential.from_vars([var], [1, 2])

    result = pot.top(5)
    assert len(result.assignments) == 2
    assert result.potentials == [2, 1]


def test_potential_comparison():
    var = Variable(0)
    pot1 = Potential.from_vars([var], [1, 2])
    pot2 = Potential.from_vars([var], [3, 4])

    assert pot1 < pot2


def test_potential_is_part():
    var1, var2, var3 = Variable(0), Variable(1), Variable(2)
    pot = Potential.from_vars([var1, var2])

    assert pot.is_part([var1, var2, var3])
    assert not pot.is_part([var1])
