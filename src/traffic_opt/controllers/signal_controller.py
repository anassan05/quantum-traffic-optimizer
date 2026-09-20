"""Minimal safety-preserving signal controller boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_signal_transition,
)
from traffic_opt.domain.models import Intersection, SignalState


class EmergencyPriorityStatus(str, Enum):
    """State of an emergency priority request."""

    ACTIVE = "active"
    TRANSITIONING = "transitioning"
    WAITING_FOR_CLEARANCE = "waiting_for_clearance"
    REJECTED = "rejected"
    RELEASED = "released"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class PriorityResult:
    """Controller result and state information for a priority operation."""

    status: EmergencyPriorityStatus
    intersection_id: str
    incoming_road_id: str | None = None
    outgoing_road_id: str | None = None
    requested_phase_index: int | None = None
    signal_state: SignalState | None = None
    message: str = ""
    yellow_elapsed_seconds: int = 0
    all_red_elapsed_seconds: int = 0


@dataclass(slots=True)
class _PendingTransition:
    intersection_id: str
    incoming_road_id: str
    outgoing_road_id: str
    current_state: SignalState
    target_phase_index: int
    yellow_elapsed_seconds: int = 0
    all_red_elapsed_seconds: int = 0


class SignalController:
    """Apply emergency requests only through existing signal safety rules.

    The domain model intentionally stores phases but does not define how a
    pair of roads maps to a movement name. That mapping is therefore injected
    at this boundary instead of being guessed or added to the domain model.
    """

    def __init__(
        self,
        intersections: Mapping[str, Intersection],
        movement_map: Mapping[tuple[str, str, str], str],
    ) -> None:
        self.intersections = dict(intersections)
        self.movement_map = dict(movement_map)
        self._pending: dict[str, _PendingTransition] = {}
        self._active: dict[str, tuple[str, str, int]] = {}

    def request_emergency_priority(
        self,
        intersection_id: str,
        incoming_road_id: str,
        outgoing_road_id: str,
        signal_state: SignalState,
    ) -> PriorityResult:
        """Request a safe priority phase for one road-to-road movement."""

        intersection = self.intersections.get(intersection_id)
        if intersection is None:
            return self._rejected(intersection_id, "intersection is not configured")
        if signal_state.intersection_id != intersection_id:
            return self._rejected(intersection_id, "signal state belongs to another intersection")
        if incoming_road_id not in intersection.incoming_road_ids:
            return self._rejected(intersection_id, "incoming road is not connected")
        if outgoing_road_id not in intersection.outgoing_road_ids:
            return self._rejected(intersection_id, "outgoing road is not connected")

        movement = self.movement_map.get(
            (intersection_id, incoming_road_id, outgoing_road_id)
        )
        if movement is None:
            return self._rejected(intersection_id, "movement has no configured signal mapping")

        target_phase_index = self._phase_for_movement(intersection, movement)
        if target_phase_index is None:
            return self._rejected(
                intersection_id,
                "no signal phase permits the requested movement",
            )
        if signal_state.current_phase_index >= len(intersection.signal_phases):
            return self._rejected(intersection_id, "current signal phase is invalid")

        current_phase = intersection.signal_phases[signal_state.current_phase_index]
        target_phase = intersection.signal_phases[target_phase_index]
        if signal_state.current_phase_index == target_phase_index:
            if signal_state.phase_elapsed_seconds > current_phase.max_green_seconds:
                return self._rejected(
                    intersection_id,
                    "current green phase has exceeded its maximum duration",
                    target_phase_index,
                )
            self._active[intersection_id] = (
                incoming_road_id,
                outgoing_road_id,
                target_phase_index,
            )
            return PriorityResult(
                EmergencyPriorityStatus.ACTIVE,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                target_phase_index,
                SignalState(
                    intersection_id,
                    target_phase_index,
                    signal_state.phase_elapsed_seconds,
                    True,
                ),
                "requested movement is already green",
            )

        if signal_state.phase_elapsed_seconds < current_phase.min_green_seconds:
            return PriorityResult(
                EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                target_phase_index,
                signal_state,
                "minimum green time has not elapsed",
            )
        if signal_state.phase_elapsed_seconds > current_phase.max_green_seconds:
            return self._rejected(
                intersection_id,
                "current green phase has exceeded its maximum duration",
                target_phase_index,
            )

        self._pending[intersection_id] = _PendingTransition(
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            signal_state,
            target_phase_index,
        )
        return PriorityResult(
            EmergencyPriorityStatus.TRANSITIONING,
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            target_phase_index,
            signal_state,
            "yellow clearance must complete before all-red clearance",
        )

    def advance_clearance(
        self,
        intersection_id: str,
        elapsed_seconds: int,
    ) -> PriorityResult:
        """Advance a pending safe transition without simulating traffic."""

        if elapsed_seconds < 0:
            raise ValueError("clearance elapsed time cannot be negative")
        pending = self._pending.get(intersection_id)
        if pending is None:
            return self._rejected(intersection_id, "no transition is pending")

        current_phase = self.intersections[intersection_id].signal_phases[
            pending.current_state.current_phase_index
        ]
        total_yellow = min(elapsed_seconds, current_phase.yellow_seconds)
        total_all_red = max(
            0,
            min(
                elapsed_seconds - current_phase.yellow_seconds,
                current_phase.all_red_seconds,
            ),
        )
        pending.yellow_elapsed_seconds = total_yellow
        pending.all_red_elapsed_seconds = total_all_red
        if total_yellow < current_phase.yellow_seconds:
            return self._clearance_result(
                pending,
                EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
                "yellow clearance is still in progress",
            )
        if total_all_red < current_phase.all_red_seconds:
            return self._clearance_result(
                pending,
                EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
                "all-red clearance is still in progress",
            )

        phases = self.intersections[intersection_id].signal_phases
        target_phase = phases[pending.target_phase_index]
        try:
            validate_signal_transition(
                current_phase,
                target_phase,
                pending.current_state.phase_elapsed_seconds,
                total_yellow,
                total_all_red,
            )
        except DomainValidationError as error:
            return self._rejected(intersection_id, str(error), pending.target_phase_index)

        self._pending.pop(intersection_id)
        self._active[intersection_id] = (
            pending.incoming_road_id,
            pending.outgoing_road_id,
            pending.target_phase_index,
        )
        state = SignalState(intersection_id, pending.target_phase_index, 0, True)
        return PriorityResult(
            EmergencyPriorityStatus.ACTIVE,
            intersection_id,
            pending.incoming_road_id,
            pending.outgoing_road_id,
            pending.target_phase_index,
            state,
            "safe emergency phase is active",
            total_yellow,
            total_all_red,
        )

    def release_emergency_priority(self, intersection_id: str) -> PriorityResult:
        """Release emergency priority so normal control can resume."""

        self._pending.pop(intersection_id, None)
        active = self._active.pop(intersection_id, None)
        if active is None:
            return self._rejected(intersection_id, "no emergency priority is active")
        return PriorityResult(
            EmergencyPriorityStatus.RELEASED,
            intersection_id,
            active[0],
            active[1],
            active[2],
            SignalState(intersection_id, active[2], 0, False),
            "emergency priority released",
        )

    def _phase_for_movement(
        self,
        intersection: Intersection,
        movement: str,
    ) -> int | None:
        for index, phase in enumerate(intersection.signal_phases):
            if movement in phase.movements:
                return index
        return None

    def _clearance_result(
        self,
        pending: _PendingTransition,
        status: EmergencyPriorityStatus,
        message: str,
    ) -> PriorityResult:
        return PriorityResult(
            status,
            pending.intersection_id,
            pending.incoming_road_id,
            pending.outgoing_road_id,
            pending.target_phase_index,
            pending.current_state,
            message,
            pending.yellow_elapsed_seconds,
            pending.all_red_elapsed_seconds,
        )

    @staticmethod
    def _rejected(
        intersection_id: str,
        message: str,
        phase_index: int | None = None,
    ) -> PriorityResult:
        return PriorityResult(
            EmergencyPriorityStatus.REJECTED,
            intersection_id,
            requested_phase_index=phase_index,
            message=message,
        )