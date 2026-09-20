"""Deterministic rule-based adaptive traffic signal controller."""

from dataclasses import dataclass
from math import ceil
from typing import Mapping

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
    validate_signal_transition,
)
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.simulation.state import SimulationState

from .base import SignalDecision


@dataclass(frozen=True, slots=True)
class AdaptiveController:
    """Choose signal phases from queue, occupancy, and waiting-time demand.

    ``movement_road_map`` is optional because the Phase 2 domain model stores
    movement names but does not own a road graph. When supplied, it maps phase
    movement names to road IDs. Without it, exact or substring matches are
    inferred from road IDs; if none can be inferred, total demand is used as a
    deterministic compatibility fallback.
    """

    movement_road_map: Mapping[str, tuple[str, ...]] | None = None
    queue_weight: float = 10.0
    waiting_time_weight: float = 1.0
    occupancy_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.queue_weight < 0 or self.waiting_time_weight < 0:
            raise ValueError("adaptive demand weights cannot be negative")
        if self.occupancy_weight < 0:
            raise ValueError("adaptive occupancy weight cannot be negative")
        if self.movement_road_map is not None:
            object.__setattr__(
                self,
                "movement_road_map",
                {key: tuple(sorted(value)) for key, value in self.movement_road_map.items()},
            )

    def decide(self, state: SimulationState) -> tuple[SignalDecision, ...]:
        """Return one deterministic, safety-validated decision per intersection."""

        decisions: list[SignalDecision] = []
        for intersection_id in sorted(state.intersection_states):
            signal = state.intersection_states[intersection_id]
            self._validate_configuration(signal.phases)
            decisions.append(self._decide_intersection(intersection_id, signal, state))
        return tuple(decisions)

    def _decide_intersection(self, intersection_id: str, signal, state) -> SignalDecision:
        current = signal.current_phase
        if signal.mode == "yellow":
            return SignalDecision(
                intersection_id,
                signal.current_phase_index,
                "yellow",
                max(1, ceil(current.yellow_seconds - signal.elapsed_seconds)),
            )
        if signal.mode == "all_red":
            target_index = self._highest_demand_phase(signal.phases, state)
            target = signal.phases[target_index]
            validate_signal_transition(
                current,
                target,
                current.min_green_seconds,
                current.yellow_seconds,
                current.all_red_seconds,
            )
            return SignalDecision(
                intersection_id,
                target_index,
                "green",
                target.min_green_seconds,
            )

        if signal.elapsed_seconds < current.min_green_seconds:
            return SignalDecision(
                intersection_id,
                signal.current_phase_index,
                "green",
                max(1, ceil(current.min_green_seconds - signal.elapsed_seconds)),
            )

        target_index = self._highest_demand_phase(signal.phases, state)
        current_score = self._phase_demand(current, state)
        target_score = self._phase_demand(signal.phases[target_index], state)
        if signal.elapsed_seconds >= current.max_green_seconds or (
            target_index != signal.current_phase_index and target_score > current_score
        ):
            next_phase = signal.phases[target_index]
            validate_signal_transition(
                current,
                next_phase,
                min(int(signal.elapsed_seconds), current.max_green_seconds),
                current.yellow_seconds,
                current.all_red_seconds,
            )
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

    def _validate_configuration(self, phases: tuple[SignalPhaseConfig, ...]) -> None:
        if not phases:
            raise DomainValidationError("adaptive controller requires signal phases")
        for phase in phases:
            validate_signal_timing(
                phase.min_green_seconds,
                phase.max_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )
            validate_phase_movements(phase.movements)
        for index, phase in enumerate(phases):
            validate_signal_transition(
                phase,
                phases[(index + 1) % len(phases)],
                phase.min_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )

    def _highest_demand_phase(
        self,
        phases: tuple[SignalPhaseConfig, ...],
        state: SimulationState,
    ) -> int:
        scores = [self._phase_demand(phase, state) for phase in phases]
        return max(range(len(phases)), key=lambda index: (scores[index], -index))

    def _phase_demand(self, phase: SignalPhaseConfig, state: SimulationState) -> float:
        road_ids = self._roads_for_phase(phase, state)
        if road_ids is None:
            road_ids = tuple(sorted(state.queue_lengths))
        queue_demand = sum(state.queue_lengths.get(road_id, 0) for road_id in road_ids)
        occupancy_demand = sum(
            len(state.road_occupancy.get(road_id, ())) for road_id in road_ids
        )
        waiting_demand = sum(
            vehicle.waiting_time_seconds
            for vehicle in state.vehicles.values()
            if vehicle.current_road_id in road_ids and not vehicle.completed
        )
        waiting_vehicle_count = sum(
            1
            for vehicle in state.vehicles.values()
            if vehicle.current_road_id in road_ids
            and vehicle.waiting_time_seconds > 0
            and not vehicle.completed
        )
        return (
            self.queue_weight * (queue_demand + waiting_vehicle_count)
            + self.waiting_time_weight * waiting_demand
            + self.occupancy_weight * occupancy_demand
        )

    def _roads_for_phase(
        self,
        phase: SignalPhaseConfig,
        state: SimulationState,
    ) -> tuple[str, ...] | None:
        known_roads = tuple(sorted(state.queue_lengths))
        matches: set[str] = set()
        has_explicit_mapping = False
        for movement in sorted(phase.movements):
            if self.movement_road_map and movement in self.movement_road_map:
                has_explicit_mapping = True
                matches.update(
                    road_id
                    for road_id in self.movement_road_map[movement]
                    if road_id in known_roads
                )
            normalized = movement.lower()
            matches.update(
                road_id
                for road_id in known_roads
                if normalized == road_id.lower() or normalized in road_id.lower()
            )
        if matches or has_explicit_mapping:
            return tuple(sorted(matches))
        return None