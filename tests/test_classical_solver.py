import pytest

from traffic_opt.quantum.classical_solver import solve_qubo
from traffic_opt.quantum.qubo import QUBO, QUBOTerm


def test_known_qubo_returns_minimum_assignment_and_energy() -> None:
    qubo = QUBO(
        variables=("x", "y"),
        linear_terms=(
            QUBOTerm(("x",), -3),
            QUBOTerm(("y",), -2),
        ),
        quadratic_terms=(QUBOTerm(("x", "y"), 4),),
        constant=1,
    )

    result = solve_qubo(qubo)

    assert dict(result.assignment) == {"x": 1, "y": 0}
    assert result.selected_variables == ("x",)
    assert result.objective_value == -2
    assert result.number_of_variables == 2
    assert result.solver_name == "exhaustive_classical"


def test_equal_energy_solutions_use_deterministic_zero_first_tie_breaking() -> None:
    qubo = QUBO(
        variables=("x", "y"),
        linear_terms=(),
        quadratic_terms=(),
    )

    result = solve_qubo(qubo)

    assert dict(result.assignment) == {"x": 0, "y": 0}
    assert result.selected_variables == ()
    assert result.objective_value == 0


def test_empty_qubo_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty QUBO"):
        solve_qubo(QUBO(variables=(), linear_terms=(), quadratic_terms=()))


def test_single_variable_qubo() -> None:
    result = solve_qubo(
        QUBO(
            variables=("phase_000",),
            linear_terms=(QUBOTerm(("phase_000",), -5),),
            quadratic_terms=(),
        )
    )

    assert result.assignment == {"phase_000": 1}
    assert result.objective_value == -5


def test_multiple_variable_qubo_and_complete_assignment() -> None:
    qubo = QUBO(
        variables=("a", "b", "c"),
        linear_terms=(
            QUBOTerm(("a",), -1),
            QUBOTerm(("b",), -4),
            QUBOTerm(("c",), -2),
        ),
        quadratic_terms=(
            QUBOTerm(("a", "b"), 10),
            QUBOTerm(("a", "c"), 10),
            QUBOTerm(("b", "c"), 10),
        ),
    )

    result = solve_qubo(qubo)

    assert set(result.assignment) == {"a", "b", "c"}
    assert result.selected_variables == ("b",)
    assert result.objective_value == -4


def test_solver_result_is_reproducible_and_assignment_is_read_only() -> None:
    qubo = QUBO(
        variables=("x", "y"),
        linear_terms=(QUBOTerm(("x",), -1),),
        quadratic_terms=(),
    )

    first = solve_qubo(qubo)
    second = solve_qubo(qubo)

    assert first == second
    with pytest.raises(TypeError):
        first.assignment["x"] = 0