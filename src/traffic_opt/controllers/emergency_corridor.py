"""Safe, temporary emergency green-corridor coordination."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import networkx as nx

from traffic_opt.domain.models import Intersection, SignalState
from .emergency_priority import EmergencyPriorityController, EmergencyPriorityStatus

if TYPE_CHECKING:
    from traffic_opt.simulation.state import IntersectionSignalState
    from traffic_opt.simulation.vehicles import VehicleState


class EmergencyCorridorStatus(str, Enum):
    """Lifecycle states exposed by the corridor manager."""

    INACTIVE = "inactive"
    WAITING_FOR_CLEARANCE = "waiting_for_clearance"
    ACTIVE = "active"
    RELEASED = "released"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EmergencyCorridorResult:
    """Result of evaluating emergency corridor priority for one step."""

    status: EmergencyCorridorStatus
    vehicle_id: str | None = None
    intersection_id: str | None = None
    incoming_road_id: str | None = None
    outgoing_road_id: str | None = None
    phase_index: int | None = None
    reason: str = ""


class EmergencyCorridorManager:
    """Coordinate one emergency vehicle without bypassing signal safety."""

    def __init__(
        self,
        graph: nx.DiGraph,
        intersections: tuple[Intersection, ...],
        phase_route_map: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        self._graph = graph
        self._intersections = {intersection.id: intersection for intersection in intersections}
        self._priority = EmergencyPriorityController(
            intersections,
            dict(phase_route_map or self._build_phase_route_map()),
        )
        self._requested_phases: dict[str, int] = {}
        self._active_vehicle_id: str | None = None
        self._active_intersection_id: str | None = None
        self._last_result = EmergencyCorridorResult(EmergencyCorridorStatus.INACTIVE)

    @property
    def active_vehicle_id(self) -> str | None:
        """Return the emergency vehicle currently owning the corridor."""

        return self._active_vehicle_id

    @property
    def last_result(self) -> EmergencyCorridorResult:
        """Return the most recent deterministic lifecycle result."""

        return self._last_result

    def next_required_movement(
        self,
        vehicle: VehicleState,
    ) -> tuple[str, str, str] | None:
        """Return ``(intersection, incoming road, outgoing road)`` for a vehicle."""

        if vehicle.completed or vehicle.route_index >= len(vehicle.route_road_ids) - 1:
            return None
        incoming_road_id = vehicle.current_road_id
        outgoing_road_id = vehicle.route_road_ids[vehicle.route_index + 1]
        incoming_road = self._road_for_id(incoming_road_id)
        outgoing_road = self._road_for_id(outgoing_road_id)
        if incoming_road.end_intersection_id != outgoing_road.start_intersection_id:
            raise ValueError("vehicle route contains disconnected roads")
        return (
            incoming_road.end_intersection_id,
            incoming_road_id,
            outgoing_road_id,
        )

    def affected_intersections(self, vehicle: VehicleState) -> tuple[str, ...]:
        """Return the next intersection affected by this vehicle's corridor."""

        movement = self.next_required_movement(vehicle)
        return () if movement is None else (movement[0],)

    def requested_phase(self, intersection_id: str) -> int | None:
        """Return a safely requested phase for the simulator transition loop."""

        return self._requested_phases.get(intersection_id)

    def update(
        self,
        vehicles: Iterable[VehicleState],
        signal_states: Mapping[str, IntersectionSignalState],
        time_seconds: float,
        is_road_closed: Callable[[str, int], bool],
    ) -> EmergencyCorridorResult:
        """Evaluate corridor ownership and request priority near an intersection."""

        emergency_vehicles = sorted(
            (vehicle for vehicle in vehicles if vehicle.is_emergency and not vehicle.completed),
            key=lambda vehicle: vehicle.id,
        )
        self._requested_phases.clear()
        if not emergency_vehicles:
            if self._active_intersection_id is not None:
                self._priority.release_emergency_priority(self._active_intersection_id)
            self._active_intersection_id = None
            self._active_vehicle_id = None
            return self._finish(
                EmergencyCorridorStatus.RELEASED
                if self._last_result.status is not EmergencyCorridorStatus.INACTIVE
                else EmergencyCorridorStatus.INACTIVE,
                reason="no active emergency vehicle",
            )

        selected = emergency_vehicles[0]
        if len(emergency_vehicles) > 1 and selected.id != self._active_vehicle_id:
            self._active_vehicle_id = selected.id
        if self._active_vehicle_id is not None and selected.id != self._active_vehicle_id:
            return self._finish(
                EmergencyCorridorStatus.REJECTED,
                selected.id,
                reason="another emergency vehicle owns the corridor",
            )
        self._active_vehicle_id = selected.id

        movement = self.next_required_movement(selected)
        if movement is None:
            if self._active_intersection_id is not None:
                self._priority.release_emergency_priority(self._active_intersection_id)
            self._active_intersection_id = None
            self._active_vehicle_id = None
            return self._finish(
                EmergencyCorridorStatus.RELEASED,
                selected.id,
                reason="emergency vehicle passed its final intersection",
            )
        intersection_id, incoming_road_id, outgoing_road_id = movement
        if (
            self._active_intersection_id is not None
            and self._active_intersection_id != intersection_id
        ):
            self._priority.release_emergency_priority(self._active_intersection_id)
        self._active_intersection_id = intersection_id
        if is_road_closed(outgoing_road_id, int(time_seconds)):
            return self._finish(
                EmergencyCorridorStatus.REJECTED,
                selected.id,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                reason="required downstream road is closed",
            )
        road = self._road_for_id(incoming_road_id)
        distance_remaining = road.length_meters - selected.position_meters
        approach_distance = selected.speed_kmh / 3.6
        if distance_remaining > approach_distance:
            return self._finish(
                EmergencyCorridorStatus.INACTIVE,
                selected.id,
                intersection_id,
                incoming_road_id,
                outgoing_road_id,
                reason="emergency vehicle is not yet approaching intersection",
            )

        signal = signal_states[intersection_id]
        result = self._priority.request_emergency_priority(
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            SignalState(
                intersection_id,
                signal.current_phase_index,
                int(signal.elapsed_seconds),
                signal.current_phase_index == self._priority_phase_for(signal, movement),
            ),
        )
        if result.phase_index is not None:
            self._requested_phases[intersection_id] = result.phase_index
        status = {
            EmergencyPriorityStatus.ACCEPTED: EmergencyCorridorStatus.WAITING_FOR_CLEARANCE,
            EmergencyPriorityStatus.WAITING_FOR_CLEARANCE: EmergencyCorridorStatus.WAITING_FOR_CLEARANCE,
            EmergencyPriorityStatus.ACTIVE: EmergencyCorridorStatus.ACTIVE,
            EmergencyPriorityStatus.REJECTED: EmergencyCorridorStatus.REJECTED,
        }.get(result.status, EmergencyCorridorStatus.REJECTED)
        return self._finish(
            status,
            selected.id,
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            result.phase_index,
            result.reason,
        )

    def _priority_phase_for(
        self,
        signal: IntersectionSignalState,
        movement: tuple[str, str, str],
    ) -> int:
        _, incoming_road_id, outgoing_road_id = movement
        tokens = {
            incoming_road_id,
            outgoing_road_id,
            f"{incoming_road_id}->{outgoing_road_id}",
            self._movement_for_road(incoming_road_id),
        }
        matches = [
            index
            for index, phase in enumerate(signal.phases)
            if phase.movements & tokens
        ]
        return matches[0] if len(matches) == 1 else -1

    def _build_phase_route_map(self) -> dict[tuple[str, str], int]:
        road_endpoints = {
            attributes["road_id"]: (start, end)
            for start, end, attributes in self._graph.edges(data=True)
        }
        route_map: dict[tuple[str, str], int] = {}
        for intersection in self._intersections.values():
            for incoming_road_id in intersection.incoming_road_ids:
                for outgoing_road_id in intersection.outgoing_road_ids:
                    if road_endpoints[incoming_road_id][1] != intersection.id:
                        continue
                    if road_endpoints[outgoing_road_id][0] != intersection.id:
                        continue
                    movement = self._movement_for_road(incoming_road_id)
                    matches = [
                        index
                        for index, phase in enumerate(intersection.signal_phases)
                        if movement in phase.movements
                    ]
                    if len(matches) == 1:
                        route_map[(incoming_road_id, outgoing_road_id)] = matches[0]
        return route_map

    def _finish(
        self,
        status: EmergencyCorridorStatus,
        vehicle_id: str | None = None,
        intersection_id: str | None = None,
        incoming_road_id: str | None = None,
        outgoing_road_id: str | None = None,
        phase_index: int | None = None,
        reason: str = "",
    ) -> EmergencyCorridorResult:
        result = EmergencyCorridorResult(
            status,
            vehicle_id,
            intersection_id,
            incoming_road_id,
            outgoing_road_id,
            phase_index,
            reason,
        )
        self._last_result = result
        return result

    def _road_for_id(self, road_id: str):
        for _, _, attributes in self._graph.edges(data=True):
            if attributes["road_id"] == road_id:
                return attributes["road_segment"]
        raise ValueError(f"unknown road: {road_id}")

    @staticmethod
    def _movement_for_road(road_id: str) -> str:
        return "east_west" if road_id.startswith("R_") else road_id
