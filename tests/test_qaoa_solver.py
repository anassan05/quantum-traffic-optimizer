import pytest

from traffic_opt.quantum.classical_solver import solve_qubo
from traffic_opt.quantum.qaoa_solver import qubo_to_ising, solve_qaoa
from traffic_opt.quantum.qubo import QUBO, QUBOTerm


def single_qubo() -> QUBO:
    return QUBO(
        variables=("x",),
        linear_terms=(QUBOTerm(("x",), -2),),
        quadratic_terms=(),
        constant=1,
    )


def multi_qubo() -> QUBO:
    return QUBO(
        variables=("x", "y"),
        linear_terms=(QUBOTerm(("x",), -3), QUBOTerm(("y",), -1)),
        quadratic_terms=(QUBOTerm(("x", "y"), 4),),
        constant=2,
    )


def test_qaoa_initialization_and_ising_mapping() -> None:
    ising = qubo_to_ising(multi_qubo())

    assert ising.offset == 1.0
    assert ising.linear_terms == (("x", 0.5), ("y", -0.5))
    assert ising.quadratic_terms == (("x", "y", 1.0),)


def test_single_variable_qaoa_result_is_valid() -> None:
    result = solve_qaoa(single_qubo(), shots=64, seed=7)

    assert set(result.assignment) == {"x"}
    assert result.objective_value == single_qubo().evaluate(result.assignment)
    assert result.number_of_variables == 1
    assert result.shots == 64
    assert result.qaoa_depth == 1
    assert result.solver_name == "qaoa_aer"


def test_multi_variable_result_contains_all_variables_and_valid_energy() -> None:
    qubo = multi_qubo()
    result = solve_qaoa(qubo, qaoa_depth=2, shots=128, seed=11)

    assert set(result.assignment) == set(qubo.variables)
    assert result.objective_value == qubo.evaluate(result.assignment)
    assert result.qaoa_depth == 2


def test_seeded_execution_is_reproducible() -> None:
    first = solve_qaoa(multi_qubo(), shots=128, seed=19)
    second = solve_qaoa(multi_qubo(), shots=128, seed=19)

    assert first == second


def test_qaoa_can_report_match_with_classical_optimum() -> None:
    qubo = multi_qubo()
    classical = solve_qubo(qubo)
    result = solve_qaoa(qubo, shots=256, seed=3, classical_result=classical)

    assert result.matches_classical_optimum is True
    assert result.objective_value == classical.objective_value


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"qaoa_depth": 0}, "depth"),
        ({"shots": 0}, "shots"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_qaoa(single_qubo(), **kwargs)


def test_empty_qubo_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty QUBO"):
        solve_qaoa(QUBO(variables=(), linear_terms=(), quadratic_terms=()))