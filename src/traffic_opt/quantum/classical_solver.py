"""Deterministic exhaustive classical solver for small QUBOs."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from .qubo import QUBO


@dataclass(frozen=True, slots=True)
class ClassicalSolveResult:
    """Immutable result returned by :func:`solve_qubo`.

    ``assignment`` contains every QUBO variable. ``selected_variables`` is a
    deterministic tuple of the variables assigned one, which is convenient
    for phase-selection callers.
    """

    assignment: Mapping[str, int]
    selected_variables: tuple[str, ...]
    objective_value: float
    number_of_variables: int
    solver_name: str = "exhaustive_classical"

    def __post_init__(self) -> None:
        assignment = MappingProxyType(dict(self.assignment))
        object.__setattr__(self, "assignment", assignment)
        if self.number_of_variables != len(assignment):
            raise ValueError("result variable count does not match assignment")
        if tuple(
            variable for variable, value in assignment.items() if value == 1
        ) != self.selected_variables:
            raise ValueError("selected variables do not match assignment")
        if not isfinite(self.objective_value):
            raise ValueError("objective value must be finite")


def solve_qubo(qubo: QUBO) -> ClassicalSolveResult:
    r"""Find the exact minimum-energy binary assignment by enumeration.

    Assignments are enumerated in the QUBO's declared variable order. Because
    ``product((0, 1), repeat=n)`` visits zero before one, equal-energy
    solutions are resolved lexicographically and reproducibly.

    The runtime is $O(2^n \cdot (n + m))$, where $n$ is the number of
    variables and $m$ is the number of QUBO terms. This is intended only for
    the small formulations used before quantum backends are introduced.
    """

    _validate_qubo(qubo)
    variables = qubo.variables
    best_assignment: dict[str, int] | None = None
    best_energy: float | None = None
    for values in product((0, 1), repeat=len(variables)):
        assignment = dict(zip(variables, values))
        energy = qubo.evaluate(assignment)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_assignment = assignment

    if best_assignment is None or best_energy is None:
        raise RuntimeError("failed to enumerate QUBO assignments")
    selected = tuple(variable for variable in variables if best_assignment[variable])
    return ClassicalSolveResult(
        assignment=best_assignment,
        selected_variables=selected,
        objective_value=best_energy,
        number_of_variables=len(variables),
    )


def _validate_qubo(qubo: QUBO) -> None:
    if not isinstance(qubo, QUBO):
        raise TypeError("solver expects a QUBO instance")
    if not qubo.variables:
        raise ValueError("cannot solve an empty QUBO")
    if len(set(qubo.variables)) != len(qubo.variables):
        raise ValueError("QUBO variables must be unique")
    if any(not variable for variable in qubo.variables):
        raise ValueError("QUBO variable names must be non-empty")
    variable_set = set(qubo.variables)
    for term in (*qubo.linear_terms, *qubo.quadratic_terms):
        if not isfinite(term.coefficient):
            raise ValueError("QUBO coefficients must be finite")
        if any(variable not in variable_set for variable in term.variables):
            raise ValueError("QUBO term references an unknown variable")
    if not isfinite(qubo.constant):
        raise ValueError("QUBO constant must be finite")