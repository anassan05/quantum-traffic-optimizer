"""Safe boundary for requesting emergency signal priority."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
    validate_signal_transition,
)
from traffic_opt.domain.models import Intersection, SignalPhaseConfig, SignalState


class EmergencyPriorityStatus(str, Enum):
    ACCEPTED = "accepted"
    WAITING_FOR_CLEARANCE = "waiting_for_clearance"
    ACTIVE = "active"
    REJECTED = "rejected"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class EmergencyPriorityResult:
    status: EmergencyPriorityStatus
    intersection_id: str
    incoming_road_id: str | None = None
    outgoing_road_id: str | None = None
    phase_index: int | None = None
    reason: str = ""
    clearance_required: bool = False

    @property
    def accepted(self) -> bool:
        return self.status in {
            EmergencyPriorityStatus.ACCEPTED,
            EmergencyPriorityStatus.ACTIVE,
            EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
        }


@dataclass(slots=True)
class _PriorityRequest:
    incoming_road_id: str
    outgoing_road_id: str
    phase_index: int


class EmergencyPriorityController:
    """Validate emergency phase requests without directly changing signals."""

    def __init__(
        self,
        intersections: tuple[Intersection, ...],
        phase_route_map: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        self._intersections = {intersection.id: intersection for intersection in intersections}
        if len(self._intersections) != len(intersections):
            raise DomainValidationError("intersection IDs must be unique")
        self._phase_route_map = dict(phase_route_map or {})
        self._requests: dict[str, _PriorityRequest] = {}
        for intersection in intersections:
            self._validate_phases(intersection.signal_phases)

    def request_emergency_priority(
        self,
        intersection_id: str,
        incoming_road_id: str,
        outgoing_road_id: str,
        signal_state: SignalState,
    ) -> EmergencyPriorityResult:
        intersection = self._intersections.get(intersection_id)
        if intersection is None:
            return self._rejected(intersection_id, "unknown intersection")
        if signal_state.intersection_id != intersection_id:
            return self._rejected(intersection_id, "signal state belongs to another intersection")
        if not 0 <= signal_state.current_phase_index < len(intersection.signal_phases):
            return self._rejected(intersection_id, "signal state phase index is out of range")
        if incoming_road_id not in intersection.incoming_road_ids:
            return self._rejected(intersection_id, "incoming road is not connected to intersection")
        if outgoing_road_id not in intersection.outgoing_road_ids:
            return self._rejected(intersection_id, "outgoing road is not connected to intersection")
        phase_index = self._resolve_phase(intersection, incoming_road_id, outgoing_road_id)
        if phase_index is None:
            return self._rejected(intersection_id, "no unambiguous safe phase serves this route")
        current = intersection.signal_phases[signal_state.current_phase_index]
        target = intersection.signal_phases[phase_index]
        request = _PriorityRequest(incoming_road_id, outgoing_road_id, phase_index)
        existing = self._requests.get(intersection_id)
        if existing is not None and existing != request:
            return self._rejected(intersection_id, "another emergency request is active")
        if signal_state.current_phase_index == phase_index:
            if signal_state.phase_elapsed_seconds > target.max_green_seconds:
                return self._rejected(intersection_id, "maximum green time has been exceeded")
            self._requests[intersection_id] = request
            if signal_state.phase_elapsed_seconds < target.min_green_seconds:
                return EmergencyPriorityResult(
                    EmergencyPriorityStatus.ACCEPTED,
                    intersection_id,
                    incoming_road_id,
                    outgoing_road_id,
                    phase_index,
                    "minimum green time is still running",
                    True,
                )
            return EmergencyPriorityResult(
                EmergencyPriorityStatus.ACTIVE,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                phase_index,
                "requested phase is safely active",
            )
        self._requests[intersection_id] = request
        if signal_state.phase_elapsed_seconds < current.min_green_seconds:
            return EmergencyPriorityResult(
                EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                phase_index,
                "waiting for current phase minimum green time",
                True,
            )
        if signal_state.phase_elapsed_seconds > current.max_green_seconds:
            return self._rejected(intersection_id, "maximum green time has been exceeded")
        return EmergencyPriorityResult(
            EmergencyPriorityStatus.WAITING_FOR_CLEARANCE,
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            phase_index,
            "waiting for yellow and all-red clearance",
            True,
        )

    def release_emergency_priority(self, intersection_id: str) -> EmergencyPriorityResult:
        request = self._requests.pop(intersection_id, None)
        return EmergencyPriorityResult(
            EmergencyPriorityStatus.RELEASED,
            intersection_id,
            request.incoming_road_id if request else None,
            request.outgoing_road_id if request else None,
            request.phase_index if request else None,
            "emergency priority released" if request else "no active emergency request",
        )

    def _resolve_phase(self, intersection: Intersection, incoming: str, outgoing: str) -> int | None:
        route = (incoming, outgoing)
        if route in self._phase_route_map:
            return self._phase_route_map[route]
        tokens = {incoming, outgoing, f"{incoming}->{outgoing}"}
        matches = [index for index, phase in enumerate(intersection.signal_phases) if phase.movements & tokens]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _validate_phases(phases: tuple[SignalPhaseConfig, ...]) -> None:
        for phase in phases:
            validate_phase_movements(phase.movements)
            validate_signal_timing(
                phase.min_green_seconds,
                phase.max_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )
        for index, phase in enumerate(phases):
            validate_signal_transition(
                phase,
                phases[(index + 1) % len(phases)],
                phase.min_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )

    @staticmethod
    def _rejected(intersection_id: str, reason: str) -> EmergencyPriorityResult:
        return EmergencyPriorityResult(
            EmergencyPriorityStatus.REJECTED,
            intersection_id,
            reason=reason,
        )
