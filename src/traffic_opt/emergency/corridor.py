"""Emergency route coordination through the existing signal controller."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import networkx as nx

from traffic_opt.controllers import (
    EmergencyPriorityStatus,
    PriorityResult,
    SignalController,
)
from traffic_opt.domain.models import SignalState


class CorridorStatus(str, Enum):
    """Lifecycle status for an emergency green corridor."""

    CREATED = "created"
    ACTIVE = "active"
    TRANSITIONING = "transitioning"
    WAITING_FOR_CLEARANCE = "waiting_for_clearance"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class CorridorMovement:
    """One ordered ambulance movement through an intersection."""

    intersection_id: str
    incoming_road_id: str
    outgoing_road_id: str


@dataclass(frozen=True, slots=True)
class CorridorState:
    """Inspectable immutable snapshot of corridor coordination state."""

    corridor_id: str
    route_road_ids: tuple[str, ...]
    movements: tuple[CorridorMovement, ...]
    current_movement_index: int
    completed_intersections: tuple[str, ...]
    status: CorridorStatus
    priority_result: PriorityResult | None

    @property
    def current_movement(self) -> CorridorMovement | None:
        if self.current_movement_index >= len(self.movements):
            return None
        return self.movements[self.current_movement_index]

    @property
    def is_active(self) -> bool:
        return self.status in {
            CorridorStatus.ACTIVE,
            CorridorStatus.TRANSITIONING,
            CorridorStatus.WAITING_FOR_CLEARANCE,
        }

    @property
    def is_complete(self) -> bool:
        return self.status is CorridorStatus.COMPLETED


class EmergencyGreenCorridor:
    """Coordinate an ordered ambulance route with ``SignalController``.

    This class never changes signal state directly. Timing, phase changes,
    movement conflicts, and emergency-preemption state remain owned by the
    injected controller.
    """

    def __init__(
        self,
        corridor_id: str,
        route_road_ids: tuple[str, ...],
        topology: nx.DiGraph,
        controller: SignalController,
        signal_states: Mapping[str, SignalState],
    ) -> None:
        if not corridor_id.strip():
            raise ValueError("corridor ID must be non-empty")
        self._validate_route(route_road_ids, topology)
        self.corridor_id = corridor_id
        self.topology = topology
        self.controller = controller
        self.signal_states = dict(signal_states)
        self._movements = self._resolve_movements(route_road_ids, topology)
        self._current_index = 0
        self._completed_intersections: list[str] = []
        self._status = CorridorStatus.CREATED
        self._priority_result: PriorityResult | None = None
        self._route_road_ids = tuple(route_road_ids)

    @property
    def state(self) -> CorridorState:
        return CorridorState(
            corridor_id=self.corridor_id,
            route_road_ids=self._route_road_ids,
            movements=self._movements,
            current_movement_index=self._current_index,
            completed_intersections=tuple(self._completed_intersections),
            status=self._status,
            priority_result=self._priority_result,
        )

    def start(self) -> CorridorState:
        """Start coordination and request priority for the first movement."""

        if self._status is not CorridorStatus.CREATED:
            raise RuntimeError("corridor has already been started")
        if not self._movements:
            self._status = CorridorStatus.COMPLETED
            return self.state
        self._request_current_priority()
        return self.state

    def request_priority(self, signal_state: SignalState | None = None) -> CorridorState:
        """Retry the current request, optionally with an updated signal state."""

        self._ensure_started_and_incomplete()
        if signal_state is not None:
            movement = self.state.current_movement
            assert movement is not None
            if signal_state.intersection_id != movement.intersection_id:
                raise ValueError("signal state does not match current intersection")
            self.signal_states[movement.intersection_id] = signal_state
        self._request_current_priority()
        return self.state

    def advance_clearance(self, elapsed_seconds: int) -> CorridorState:
        """Delegate pending yellow/all-red progression to the controller."""

        self._ensure_started_and_incomplete()
        movement = self.state.current_movement
        assert movement is not None
        result = self.controller.advance_clearance(
            movement.intersection_id,
            elapsed_seconds,
        )
        self._record_result(result)
        return self.state

    def clear_current_intersection(self) -> CorridorState:
        """Release the active intersection and request the next movement."""

        self._ensure_started_and_incomplete()
        movement = self.state.current_movement
        assert movement is not None
        if (
            self._priority_result is None
            or self._priority_result.status is not EmergencyPriorityStatus.ACTIVE
        ):
            raise RuntimeError("current emergency movement is not active")

        released = self.controller.release_emergency_priority(
            movement.intersection_id
        )
        self._record_result(released)
        if released.status is not EmergencyPriorityStatus.RELEASED:
            raise RuntimeError(released.message or "emergency priority release failed")

        self._completed_intersections.append(movement.intersection_id)
        self._current_index += 1
        if self._current_index == len(self._movements):
            self._status = CorridorStatus.COMPLETED
            return self.state

        self._request_current_priority()
        return self.state

    def _request_current_priority(self) -> None:
        movement = self.state.current_movement
        assert movement is not None
        signal_state = self.signal_states.get(movement.intersection_id)
        if signal_state is None:
            result = PriorityResult(
                EmergencyPriorityStatus.REJECTED,
                movement.intersection_id,
                movement.incoming_road_id,
                movement.outgoing_road_id,
                message="signal configuration is missing",
            )
        else:
            result = self.controller.request_emergency_priority(
                movement.intersection_id,
                movement.incoming_road_id,
                movement.outgoing_road_id,
                signal_state,
            )
        self._record_result(result)

    def _record_result(self, result: PriorityResult) -> None:
        self._priority_result = result
        status_map = {
            EmergencyPriorityStatus.ACTIVE: CorridorStatus.ACTIVE,
            EmergencyPriorityStatus.TRANSITIONING: CorridorStatus.TRANSITIONING,
            EmergencyPriorityStatus.WAITING_FOR_CLEARANCE: CorridorStatus.WAITING_FOR_CLEARANCE,
            EmergencyPriorityStatus.REJECTED: CorridorStatus.REJECTED,
        }
        if result.status in status_map:
            self._status = status_map[result.status]
        if result.signal_state is not None:
            self.signal_states[result.signal_state.intersection_id] = result.signal_state

    def _ensure_started_and_incomplete(self) -> None:
        if self._status is CorridorStatus.CREATED:
            raise RuntimeError("corridor has not been started")
        if self._status is CorridorStatus.COMPLETED:
            raise RuntimeError("corridor is already completed")

    @staticmethod
    def _validate_route(route_road_ids: tuple[str, ...], topology: nx.DiGraph) -> None:
        if not route_road_ids:
            return
        road_ids = [attributes.get("road_id") for _, _, attributes in topology.edges(data=True)]
        if len(set(route_road_ids)) != len(route_road_ids):
            raise ValueError("ambulance route cannot repeat a road")
        missing = [road_id for road_id in route_road_ids if road_id not in road_ids]
        if missing:
            raise ValueError(f"ambulance route contains unknown road: {missing[0]}")

    @staticmethod
    def _resolve_movements(
        route_road_ids: tuple[str, ...], topology: nx.DiGraph
    ) -> tuple[CorridorMovement, ...]:
        roads = {
            attributes["road_id"]: attributes["road_segment"]
            for _, _, attributes in topology.edges(data=True)
            if "road_id" in attributes and "road_segment" in attributes
        }
        movements: list[CorridorMovement] = []
        for incoming_id, outgoing_id in zip(route_road_ids, route_road_ids[1:]):
            incoming = roads.get(incoming_id)
            outgoing = roads.get(outgoing_id)
            if incoming is None or outgoing is None:
                raise ValueError("route road is missing its road segment")
            if incoming.end_intersection_id != outgoing.start_intersection_id:
                raise ValueError("ambulance route contains disconnected roads")
            movements.append(
                CorridorMovement(
                    incoming.end_intersection_id,
                    incoming_id,
                    outgoing_id,
                )
            )
        intersections = [movement.intersection_id for movement in movements]
        if len(set(intersections)) != len(intersections):
            raise ValueError("ambulance route repeats an intersection")
        return tuple(movements)