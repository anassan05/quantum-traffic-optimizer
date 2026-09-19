"""Hybrid QUBO/QAOA traffic signal controller with safety gating."""

from dataclasses import dataclass, field
from math import ceil
from typing import Mapping

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
    validate_signal_transition,
)
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.quantum import qaoa_solver
from traffic_opt.quantum.classical_solver import ClassicalSolveResult, solve_qubo
from traffic_opt.quantum.qaoa_solver import QAOASolveResult
from traffic_opt.quantum.qubo import QUBOWeights, build_signal_phase_qubo
from traffic_opt.simulation.state import IntersectionSignalState, SimulationState

from .base import SignalDecision


@dataclass(frozen=True, slots=True)
class QuantumHybridComparison:
    """QAOA and exact-classical results for one intersection."""

    qaoa_phase_index: int | None
    qaoa_objective_value: float | None
    classical_phase_index: int
    classical_objective_value: float
    qaoa_matches_classical: bool
    used_fallback: bool


@dataclass(slots=True)
class QuantumHybridController:
    """Coordinate QUBO construction, QAOA sampling, and safety validation.

    The controller never mutates ``SimulationState``. A QAOA result is only a
    candidate; it must be one-hot, reference a known phase, and pass the
    existing signal transition validator before becoming a ``SignalDecision``.
    """

    qaoa_depth: int = 1
    shots: int = 256
    seed: int | None = 0
    weights: QUBOWeights = field(default_factory=QUBOWeights)
    movement_road_map: Mapping[str, tuple[str, ...]] | None = None
    comparisons: dict[str, QuantumHybridComparison] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.qaoa_depth, bool)
            or not isinstance(self.qaoa_depth, int)
            or self.qaoa_depth <= 0
        ):
            raise ValueError("QAOA depth must be a positive integer")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if self.movement_road_map is not None:
            self.movement_road_map = {
                key: tuple(sorted(value))
                for key, value in self.movement_road_map.items()
            }

    def decide(self, state: SimulationState) -> tuple[SignalDecision, ...]:
        """Return safe decisions for every intersection in sorted order."""

        self.comparisons.clear()
        decisions = []
        for intersection_id in sorted(state.intersection_states):
            signal = state.intersection_states[intersection_id]
            self._validate_phases(signal.phases)
            decisions.append(self._decide_intersection(intersection_id, signal, state))
        return tuple(decisions)

    def _decide_intersection(
        self,
        intersection_id: str,
        signal: IntersectionSignalState,
        state: SimulationState,
    ) -> SignalDecision:
        current = signal.current_phase
        if signal.mode == "yellow":
            if signal.elapsed_seconds < current.yellow_seconds:
                return SignalDecision(
                    intersection_id,
                    signal.current_phase_index,
                    "yellow",
                    max(1, ceil(current.yellow_seconds - signal.elapsed_seconds)),
                )
            return SignalDecision(
                intersection_id,
                signal.current_phase_index,
                "all_red",
                current.all_red_seconds,
            )
        if signal.mode == "all_red":
            if signal.elapsed_seconds < current.all_red_seconds:
                return SignalDecision(
                    intersection_id,
                    signal.current_phase_index,
                    "all_red",
                    max(1, ceil(current.all_red_seconds - signal.elapsed_seconds)),
                )
            selected_index = self._select_phase(signal, state)
            self._validate_transition(current, signal.phases[selected_index], signal)
            return SignalDecision(
                intersection_id,
                selected_index,
                "green",
                signal.phases[selected_index].min_green_seconds,
            )

        if signal.elapsed_seconds < current.min_green_seconds:
            return SignalDecision(
                intersection_id,
                signal.current_phase_index,
                "green",
                max(1, ceil(current.min_green_seconds - signal.elapsed_seconds)),
            )

        selected_index = self._select_phase(signal, state)
        if (
            selected_index != signal.current_phase_index
            or signal.elapsed_seconds >= current.max_green_seconds
        ):
            self._validate_transition(current, signal.phases[selected_index], signal)
            return SignalDecision(
                intersection_id,
                signal.current_phase_index,
                "yellow",
                current.yellow_seconds,
            )
        return SignalDecision(
            intersection_id,
            signal.current_phase_index,
            "green",
            max(1, ceil(current.max_green_seconds - signal.elapsed_seconds)),
        )

    def _select_phase(
        self,
        signal: IntersectionSignalState,
        state: SimulationState,
    ) -> int:
        phases = signal.phases
        qubo = build_signal_phase_qubo(
            state,
            phases,
            current_phase_index=signal.current_phase_index,
            weights=self.weights,
            movement_road_map=self.movement_road_map,
        )
        classical = solve_qubo(qubo)
        classical_index = self._phase_from_result(classical, phases, qubo)
        qaoa_result: QAOASolveResult | None = None
        used_fallback = False
        try:
            qaoa_result = qaoa_solver.solve_qaoa(
                qubo,
                qaoa_depth=self.qaoa_depth,
                shots=self.shots,
                seed=self.seed,
                classical_result=classical,
            )
            qaoa_index = self._phase_from_result(qaoa_result, phases, qubo)
            if qaoa_index is None:
                used_fallback = True
        except (ValueError, RuntimeError, TypeError, DomainValidationError):
            qaoa_index = None
            used_fallback = True
        if qaoa_index is None:
            selected_index = classical_index
        else:
            selected_index = qaoa_index
        self.comparisons[signal.intersection_id] = QuantumHybridComparison(
            qaoa_phase_index=qaoa_index,
            qaoa_objective_value=(
                None if qaoa_result is None else qaoa_result.objective_value
            ),
            classical_phase_index=classical_index,
            classical_objective_value=classical.objective_value,
            qaoa_matches_classical=(qaoa_index == classical_index),
            used_fallback=used_fallback,
        )
        return selected_index

    @staticmethod
    def _phase_from_result(
        result,
        phases: tuple[SignalPhaseConfig, ...],
        qubo,
    ) -> int | None:
        selected = tuple(result.selected_variables)
        if len(selected) != 1:
            return None
        if set(result.assignment) != set(qubo.variables):
            return None
        if any(value not in (0, 1) for value in result.assignment.values()):
            return None
        if tuple(
            variable for variable in qubo.variables if result.assignment[variable]
        ) != selected:
            return None
        if result.objective_value != qubo.evaluate(result.assignment):
            return None
        prefix, index_text = selected[0].split("_")[-2:]
        if prefix != "phase" or not index_text.isdigit():
            return None
        index = int(index_text)
        if not 0 <= index < len(phases):
            return None
        return index

    @staticmethod
    def _validate_phases(phases: tuple[SignalPhaseConfig, ...]) -> None:
        if not phases:
            raise DomainValidationError("quantum-hybrid controller requires phases")
        for phase in phases:
            validate_phase_movements(phase.movements)
            validate_signal_timing(
                phase.min_green_seconds,
                phase.max_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )

    @staticmethod
    def _validate_transition(
        current: SignalPhaseConfig,
        candidate: SignalPhaseConfig,
        signal: IntersectionSignalState,
    ) -> None:
        green_elapsed = min(
            max(int(signal.elapsed_seconds), current.min_green_seconds),
            current.max_green_seconds,
        )
        validate_signal_transition(
            current,
            candidate,
            green_elapsed,
            current.yellow_seconds,
            current.all_red_seconds,
        )