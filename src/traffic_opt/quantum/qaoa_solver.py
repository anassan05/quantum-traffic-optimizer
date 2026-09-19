"""Small QAOA solver backed by Qiskit Aer.

This module is an experimental local-simulator integration. It does not claim
quantum advantage and does not modify traffic signals or simulation state.
"""

from dataclasses import dataclass
from itertools import product
from math import isfinite, pi
from types import MappingProxyType
from typing import Mapping

from .classical_solver import ClassicalSolveResult
from .qubo import QUBO


@dataclass(frozen=True, slots=True)
class IsingHamiltonian:
    """Diagonal Ising Hamiltonian equivalent to a binary QUBO."""

    linear_terms: tuple[tuple[str, float], ...]
    quadratic_terms: tuple[tuple[str, str, float], ...]
    offset: float


@dataclass(frozen=True, slots=True)
class QAOASolveResult:
    """Result of sampling a QAOA circuit and selecting the best sample."""

    assignment: Mapping[str, int]
    selected_variables: tuple[str, ...]
    objective_value: float
    number_of_variables: int
    shots: int
    qaoa_depth: int
    solver_name: str = "qaoa_aer"
    matches_classical_optimum: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment", MappingProxyType(dict(self.assignment)))
        if set(self.assignment) != set(self.selected_variables) | {
            variable for variable, value in self.assignment.items() if value == 0
        }:
            raise ValueError("assignment and selected variables do not agree")
        if self.number_of_variables != len(self.assignment):
            raise ValueError("result variable count does not match assignment")


def qubo_to_ising(qubo: QUBO) -> IsingHamiltonian:
    r"""Convert a QUBO into coefficients for $H=c+\sum h_iZ_i+\sum J_{ij}Z_iZ_j$."""

    _validate_qubo(qubo)
    linear = {variable: 0.0 for variable in qubo.variables}
    quadratic: dict[tuple[str, str], float] = {}
    offset = qubo.constant
    for term in qubo.linear_terms:
        variable = term.variables[0]
        coefficient = term.coefficient
        offset += coefficient / 2
        linear[variable] -= coefficient / 2
    for term in qubo.quadratic_terms:
        left, right = term.variables
        coefficient = term.coefficient
        offset += coefficient / 4
        linear[left] -= coefficient / 4
        linear[right] -= coefficient / 4
        quadratic[(left, right)] = quadratic.get((left, right), 0.0) + coefficient / 4
    return IsingHamiltonian(
        tuple((variable, linear[variable]) for variable in qubo.variables),
        tuple(
            (left, right, quadratic[(left, right)])
            for left, right in sorted(quadratic)
        ),
        offset,
    )


def solve_qaoa(
    qubo: QUBO,
    *,
    qaoa_depth: int = 1,
    shots: int = 256,
    seed: int | None = 0,
    classical_result: ClassicalSolveResult | None = None,
) -> QAOASolveResult:
    """Build, tune, and sample a small QAOA circuit using Qiskit Aer.

    A deterministic coarse angle grid is used to select QAOA parameters. The
    returned assignment is always chosen from Aer's measured samples and its
    energy is recomputed with the existing QUBO evaluator.
    """

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    _validate_qubo(qubo)
    _validate_parameters(qaoa_depth, shots, seed)
    ising = qubo_to_ising(qubo)
    angles = _select_angles(qubo, ising, qaoa_depth, seed)
    circuit = _build_qaoa_circuit(qubo, ising, qaoa_depth, angles, measure=True)
    simulator = AerSimulator(seed_simulator=seed)
    compiled = transpile(circuit, simulator, optimization_level=0)
    counts = simulator.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    sampled_assignments = [_assignment_from_bitstring(qubo, bitstring) for bitstring in counts]
    best_assignment = min(
        sampled_assignments,
        key=lambda assignment: (qubo.evaluate(assignment), tuple(assignment.values())),
    )
    objective = qubo.evaluate(best_assignment)
    matches = None
    if classical_result is not None:
        if set(classical_result.assignment) != set(qubo.variables):
            raise ValueError("classical result variables do not match the QUBO")
        matches = objective == classical_result.objective_value
    return QAOASolveResult(
        assignment=best_assignment,
        selected_variables=tuple(
            variable for variable in qubo.variables if best_assignment[variable]
        ),
        objective_value=objective,
        number_of_variables=len(qubo.variables),
        shots=shots,
        qaoa_depth=qaoa_depth,
        matches_classical_optimum=matches,
    )


def _select_angles(
    qubo: QUBO,
    ising: IsingHamiltonian,
    depth: int,
    seed: int | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Select angles by a deterministic coarse Aer statevector search."""

    from qiskit_aer import AerSimulator

    del seed
    grid = (0.0, pi / 4, pi / 2, 3 * pi / 4)
    candidates = product(grid, repeat=2 * depth)
    best: tuple[float, tuple[float, ...]] | None = None
    simulator = AerSimulator(method="statevector")
    for values in candidates:
        gammas = tuple(values[:depth])
        betas = tuple(values[depth:])
        circuit = _build_qaoa_circuit(
            qubo, ising, depth, (gammas, betas), measure=False
        )
        circuit.save_statevector()
        statevector = simulator.run(circuit).result().data(0)["statevector"]
        probabilities = statevector.probabilities()
        expectation = 0.0
        for index, probability in enumerate(probabilities):
            assignment = _assignment_from_bitstring(qubo, format(index, f"0{len(qubo.variables)}b"))
            expectation += probability * qubo.evaluate(assignment)
        if best is None or expectation < best[0]:
            best = (expectation, values)
    if best is None:
        raise RuntimeError("failed to select QAOA angles")
    return tuple(best[1][:depth]), tuple(best[1][depth:])


def _build_qaoa_circuit(
    qubo: QUBO,
    ising: IsingHamiltonian,
    depth: int,
    angles: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    measure: bool,
) -> "QuantumCircuit":
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(len(qubo.variables), len(qubo.variables) if measure else 0)
    circuit.h(range(len(qubo.variables)))
    indices = {variable: index for index, variable in enumerate(qubo.variables)}
    gammas, betas = angles
    for layer in range(depth):
        gamma = gammas[layer]
        beta = betas[layer]
        for variable, coefficient in ising.linear_terms:
            circuit.rz(2 * gamma * coefficient, indices[variable])
        for left, right, coefficient in ising.quadratic_terms:
            circuit.rzz(2 * gamma * coefficient, indices[left], indices[right])
        for index in range(len(qubo.variables)):
            circuit.rx(2 * beta, index)
    if measure:
        circuit.measure(range(len(qubo.variables)), range(len(qubo.variables)))
    return circuit


def _assignment_from_bitstring(qubo: QUBO, bitstring: str) -> dict[str, int]:
    bits = bitstring.replace(" ", "")
    if len(bits) != len(qubo.variables) or any(bit not in "01" for bit in bits):
        raise ValueError("Aer returned an invalid measurement bitstring")
    return {
        variable: int(bit)
        for variable, bit in zip(qubo.variables, reversed(bits))
    }


def _validate_parameters(depth: int, shots: int, seed: int | None) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("QAOA depth must be a positive integer")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")


def _validate_qubo(qubo: QUBO) -> None:
    if not isinstance(qubo, QUBO):
        raise TypeError("QAOA solver expects a QUBO instance")
    if not qubo.variables:
        raise ValueError("cannot solve an empty QUBO")
    if len(set(qubo.variables)) != len(qubo.variables):
        raise ValueError("QUBO variables must be unique")
    variables = set(qubo.variables)
    for term in (*qubo.linear_terms, *qubo.quadratic_terms):
        if not isfinite(term.coefficient):
            raise ValueError("QUBO coefficients must be finite")
        if any(variable not in variables for variable in term.variables):
            raise ValueError("QUBO term references an unknown variable")
    if not isfinite(qubo.constant):
        raise ValueError("QUBO constant must be finite")