"""Deterministic QUBO formulation for selecting one signal phase.

This module only builds a mathematical objective. It does not solve the QUBO,
invoke Qiskit, or make claims about quantum advantage.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
)
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.simulation.state import SimulationState


@dataclass(frozen=True, slots=True)
class QUBOWeights:
    """Non-negative weights for the signal-phase QUBO objective."""

    queue_length: float = 1.0
    waiting_time: float = 1.0
    waiting_vehicles: float = 1.0
    road_occupancy: float = 1.0
    phase_switching: float = 1.0
    fairness: float = 1.0
    one_hot_penalty: float = 100.0
    conflict_penalty: float = 100.0

    def __post_init__(self) -> None:
        for name, value in self.__dataclass_fields__.items():
            weight = getattr(self, name)
            if weight < 0:
                raise ValueError(f"{name} weight cannot be negative")
        if self.one_hot_penalty == 0:
            raise ValueError("one-hot penalty must be greater than zero")
        if self.conflict_penalty == 0:
            raise ValueError("conflict penalty must be greater than zero")


@dataclass(frozen=True, slots=True)
class QUBOTerm:
    """One linear or quadratic term in a QUBO objective."""

    variables: tuple[str, ...]
    coefficient: float

    def __post_init__(self) -> None:
        if len(self.variables) not in (1, 2):
            raise ValueError("a QUBO term must contain one or two variables")
        if tuple(sorted(self.variables)) != self.variables:
            raise ValueError("QUBO term variables must be sorted")
        if len(self.variables) == 2 and self.variables[0] == self.variables[1]:
            raise ValueError("quadratic QUBO term variables must be distinct")


@dataclass(frozen=True, slots=True)
class QUBO:
    r"""Immutable, ordered representation of a binary quadratic objective.

    The objective is

    $$E(x) = c + \sum_i a_i x_i + \sum_{i<j} b_{ij}x_ix_j,$$

    where each variable is binary. Terms are tuples rather than mutable
    dictionaries so serialization and comparisons remain deterministic.
    """

    variables: tuple[str, ...]
    linear_terms: tuple[QUBOTerm, ...]
    quadratic_terms: tuple[QUBOTerm, ...]
    constant: float = 0.0

    def evaluate(self, assignment: Mapping[str, int]) -> float:
        """Evaluate the objective for a complete binary assignment."""

        if set(assignment) != set(self.variables):
            raise ValueError("assignment must contain exactly the QUBO variables")
        if any(value not in (0, 1) for value in assignment.values()):
            raise ValueError("QUBO assignments must be binary")
        energy = self.constant
        energy += sum(
            term.coefficient * assignment[term.variables[0]]
            for term in self.linear_terms
        )
        energy += sum(
            term.coefficient
            * assignment[term.variables[0]]
            * assignment[term.variables[1]]
            for term in self.quadratic_terms
        )
        return energy


_CONFLICTING_MOVEMENTS = {
    frozenset({"north_south", "east_west"}),
    frozenset({"northbound", "southbound"}),
    frozenset({"eastbound", "westbound"}),
}


def build_signal_phase_qubo(
    state: SimulationState,
    phases: Sequence[SignalPhaseConfig],
    *,
    current_phase_index: int = 0,
    weights: QUBOWeights | None = None,
    movement_road_map: Mapping[str, tuple[str, ...]] | None = None,
    road_capacities: Mapping[str, int] | None = None,
) -> QUBO:
    """Build a QUBO selecting exactly one safe candidate signal phase.

    Demand terms are negative because the objective is minimized: selecting a
    phase serving a high-demand approach lowers energy. The one-hot penalty
    discourages selecting zero or multiple phases. Cross-phase conflicting
    movement pairs receive an additional quadratic penalty. Candidate phases
    are individually validated with the existing domain safety rules.
    """

    candidates = tuple(phases)
    _validate_inputs(state, candidates, current_phase_index, road_capacities)
    weights = weights or QUBOWeights()
    variables = tuple(f"phase_{index:03d}" for index in range(len(candidates)))
    components = tuple(
        _phase_components(state, phase, movement_road_map, road_capacities)
        for phase in candidates
    )
    demand = tuple(sum(component) for component in components)
    maximum_demand = max(demand, default=0.0)

    linear: list[QUBOTerm] = []
    quadratic: list[QUBOTerm] = []
    constant = weights.one_hot_penalty
    for index, variable in enumerate(variables):
        phase = candidates[index]
        coefficient = -weights.one_hot_penalty
        queue, waiting_time, waiting_vehicles, occupancy = components[index]
        coefficient -= weights.queue_length * queue
        coefficient -= weights.waiting_time * waiting_time
        coefficient -= weights.waiting_vehicles * waiting_vehicles
        coefficient -= weights.road_occupancy * occupancy
        coefficient += weights.phase_switching * (
            0 if index == current_phase_index else 1
        )
        coefficient += weights.fairness * (maximum_demand - demand[index])
        linear.append(QUBOTerm((variable,), coefficient))

    for left in range(len(variables)):
        for right in range(left + 1, len(variables)):
            coefficient = 2 * weights.one_hot_penalty
            if _phases_conflict(candidates[left], candidates[right]):
                coefficient += weights.conflict_penalty
            quadratic.append(
                QUBOTerm((variables[left], variables[right]), coefficient)
            )
    return QUBO(tuple(variables), tuple(linear), tuple(quadratic), constant)


def _validate_inputs(
    state: SimulationState,
    phases: tuple[SignalPhaseConfig, ...],
    current_phase_index: int,
    road_capacities: Mapping[str, int] | None,
) -> None:
    if not phases:
        raise DomainValidationError("at least one candidate signal phase is required")
    if not 0 <= current_phase_index < len(phases):
        raise DomainValidationError("current phase index is out of range")
    if len({phase.name for phase in phases}) != len(phases):
        raise DomainValidationError("candidate signal phase names must be unique")
    for phase in phases:
        validate_phase_movements(phase.movements)
        validate_signal_timing(
            phase.min_green_seconds,
            phase.max_green_seconds,
            phase.yellow_seconds,
            phase.all_red_seconds,
        )
    if road_capacities is not None:
        if any(capacity <= 0 for capacity in road_capacities.values()):
            raise DomainValidationError("road capacities must be positive")
        if any(
            road_id not in road_capacities for road_id in state.road_occupancy
        ):
            raise DomainValidationError("road capacity is missing for occupied road")


def _phase_components(
    state: SimulationState,
    phase: SignalPhaseConfig,
    movement_road_map: Mapping[str, tuple[str, ...]] | None,
    road_capacities: Mapping[str, int] | None,
) -> tuple[float, float, float, float]:
    road_ids = _roads_for_phase(state, phase, movement_road_map)
    queue = sum(state.queue_lengths.get(road_id, 0) for road_id in road_ids)
    waiting_time = sum(
        vehicle.waiting_time_seconds
        for vehicle in state.vehicles.values()
        if vehicle.current_road_id in road_ids and not vehicle.completed
    )
    waiting_vehicles = sum(
        1
        for vehicle in state.vehicles.values()
        if vehicle.current_road_id in road_ids
        and vehicle.waiting_time_seconds > 0
        and not vehicle.completed
    )
    if road_capacities is None:
        occupancy = sum(
            len(state.road_occupancy.get(road_id, ())) for road_id in road_ids
        )
    else:
        occupancy = sum(
            len(state.road_occupancy.get(road_id, ()))
            / road_capacities[road_id]
            for road_id in road_ids
        )
    return queue, waiting_time, waiting_vehicles, occupancy


def _roads_for_phase(
    state: SimulationState,
    phase: SignalPhaseConfig,
    movement_road_map: Mapping[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    known_roads = tuple(sorted(set(state.queue_lengths) | set(state.road_occupancy)))
    matches: set[str] = set()
    for movement in sorted(phase.movements):
        if movement_road_map and movement in movement_road_map:
            matches.update(
                road_id for road_id in movement_road_map[movement] if road_id in known_roads
            )
        normalized = movement.lower()
        matches.update(
            road_id
            for road_id in known_roads
            if normalized == road_id.lower() or normalized in road_id.lower()
        )
    return tuple(sorted(matches)) or known_roads


def _phases_conflict(left: SignalPhaseConfig, right: SignalPhaseConfig) -> bool:
    """Return whether selecting both phases could request conflicting greens."""

    for left_movement in left.movements:
        for right_movement in right.movements:
            if frozenset({left_movement, right_movement}) in _CONFLICTING_MOVEMENTS:
                return True
    return False