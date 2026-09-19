"""Small deterministic discrete-time traffic simulator."""

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx

from traffic_opt.domain.models import Intersection
from traffic_opt.domain.models import TrafficEvent

from .events import (
    EffectiveRoadCondition,
    EventImpactConfig,
    effective_road_conditions,
)
from .state import IntersectionSignalState, SimulationState
from .vehicles import VehicleState


@dataclass(slots=True)
class TrafficSimulator:
    """Advance vehicles through a fixed network under fixed signal states.

    This class intentionally does not choose signal phases. Tests and later
    controller implementations may set a phase explicitly through
    ``set_signal_phase``; all movement still passes the runtime safety state.
    """

    graph: nx.DiGraph
    intersections: tuple[Intersection, ...]
    time_step_seconds: float = 1.0
    seed: int = 0
    events: tuple[TrafficEvent, ...] = ()
    event_impact: EventImpactConfig = field(default_factory=EventImpactConfig)
    state: SimulationState = field(init=False)
    _scheduled_vehicles: list[tuple[float, VehicleState, str | None]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        if self.time_step_seconds <= 0:
            raise ValueError("time step must be greater than zero")
        self.state = SimulationState()
        intersection_by_id = {intersection.id: intersection for intersection in self.intersections}
        if set(intersection_by_id) != set(self.graph.nodes):
            raise ValueError("intersection configuration must match graph nodes")
        self.state.intersection_states = {
            intersection_id: IntersectionSignalState(
                intersection_id=intersection_id,
                phases=intersection_by_id[intersection_id].signal_phases,
            )
            for intersection_id in sorted(intersection_by_id)
        }
        self.state.queue_lengths = {
            attributes["road_id"]: 0
            for _, _, attributes in self.graph.edges(data=True)
        }
        self.state.refresh_road_occupancy()

    def add_vehicles(self, vehicles: Iterable[VehicleState]) -> None:
        """Add vehicles after validating IDs and road references."""

        pending: list[VehicleState] = []
        occupancy = {
            road_id: len(vehicle_ids)
            for road_id, vehicle_ids in self.state.road_occupancy.items()
        }
        for vehicle in vehicles:
            if vehicle.id in self.state.vehicles or any(
                existing.id == vehicle.id for existing in pending
            ):
                raise ValueError(f"duplicate vehicle id: {vehicle.id}")
            for road_id in vehicle.route_road_ids:
                if not self._has_road_id(road_id):
                    raise ValueError(f"unknown route road: {road_id}")
            if not self._has_road_id(vehicle.current_road_id):
                raise ValueError(f"unknown road: {vehicle.current_road_id}")
            if not self._condition_for(vehicle.current_road_id).available:
                raise ValueError(f"road is closed: {vehicle.current_road_id}")
            if occupancy.get(vehicle.current_road_id, 0) >= self._capacity_for_id(
                vehicle.current_road_id,
                self._current_conditions(),
            ):
                raise ValueError(f"road capacity exceeded: {vehicle.current_road_id}")
            occupancy[vehicle.current_road_id] = (
                occupancy.get(vehicle.current_road_id, 0) + 1
            )
            pending.append(vehicle)
        for vehicle in pending:
            self.state.vehicles[vehicle.id] = vehicle
        self.state.refresh_road_occupancy()
        self._refresh_queues()

    def set_events(self, events: Iterable[TrafficEvent]) -> None:
        """Replace the scheduled event input without mutating the topology."""

        scheduled = tuple(events)
        event_ids = [event.id for event in scheduled]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("scheduled event IDs must be unique")
        self.events = scheduled

    def schedule_vehicle(
        self,
        vehicle: VehicleState,
        spawn_time_seconds: float,
        *,
        arrival_event_id: str | None = None,
    ) -> None:
        """Schedule an existing vehicle state for later insertion.

        When ``arrival_event_id`` is supplied, insertion waits until that
        special-demand event is active. Vehicle movement still occurs only in
        the simulator's normal ``step`` loop.
        """

        if spawn_time_seconds < 0:
            raise ValueError("vehicle spawn time cannot be negative")
        if vehicle.id in self.state.vehicles or any(
            item[1].id == vehicle.id for item in self._scheduled_vehicles
        ):
            raise ValueError(f"duplicate vehicle id: {vehicle.id}")
        self._scheduled_vehicles.append(
            (spawn_time_seconds, vehicle, arrival_event_id)
        )
        self._scheduled_vehicles.sort(key=lambda item: (item[0], item[1].id))

    def active_events(self, time_seconds: float | None = None) -> tuple[TrafficEvent, ...]:
        """Return scheduled events active under the existing [start, end) rule."""

        current_time = self.state.time_seconds if time_seconds is None else time_seconds
        return tuple(
            event
            for event in self.events
            if event.start_time_seconds <= current_time
            < event.start_time_seconds + event.duration_seconds
        )

    def road_conditions(
        self,
        active_events: Iterable[TrafficEvent] | None = None,
    ) -> dict[str, EffectiveRoadCondition]:
        """Return effective conditions for the current simulation instant."""

        events = self.active_events() if active_events is None else tuple(active_events)
        return dict(
            effective_road_conditions(
                self._road_ids(), events, impact=self.event_impact
            )
        )

    def set_signal_phase(
        self,
        intersection_id: str,
        phase_index: int,
        mode: str = "green",
    ) -> None:
        """Set a test/configuration signal state without bypassing mode safety."""

        if mode not in {"green", "yellow", "all_red"}:
            raise ValueError("signal mode must be green, yellow, or all_red")
        signal = self.state.intersection_states[intersection_id]
        if not 0 <= phase_index < len(signal.phases):
            raise ValueError("signal phase index is out of range")
        signal.current_phase_index = phase_index
        signal.mode = mode  # type: ignore[assignment]
        signal.elapsed_seconds = 0.0

    def step(
        self,
        active_events: Iterable[TrafficEvent] | None = None,
    ) -> SimulationState:
        """Advance signals and vehicles by one configured time step."""

        current_events = (
            self.active_events() if active_events is None else tuple(active_events)
        )
        conditions = self.road_conditions(current_events)
        self._activate_scheduled_vehicles(conditions, current_events)
        self._advance_signal_clocks()
        self.state.refresh_road_occupancy()
        occupancy = {
            road_id: len(vehicle_ids)
            for road_id, vehicle_ids in self.state.road_occupancy.items()
        }
        for vehicle in sorted(self.state.vehicles.values(), key=lambda item: item.id):
            if vehicle.completed:
                continue
            road = self._road_for_id(vehicle.current_road_id)
            condition = conditions[vehicle.current_road_id]
            distance = (
                vehicle.speed_kmh
                * condition.speed_multiplier
                / 3.6
                * self.time_step_seconds
            )
            if vehicle.position_meters + distance < road.length_meters:
                vehicle.position_meters += distance
                continue

            remaining_distance = road.length_meters - vehicle.position_meters
            endpoint = road.end_intersection_id
            if self._can_exit_vehicle(vehicle, endpoint, occupancy, conditions):
                occupancy[vehicle.current_road_id] -= 1
                if vehicle.route_index == len(vehicle.route_road_ids) - 1:
                    vehicle.position_meters = road.length_meters
                    vehicle.completed = True
                else:
                    vehicle.route_index += 1
                    vehicle.current_road_id = vehicle.route_road_ids[vehicle.route_index]
                    vehicle.position_meters = min(
                        distance - remaining_distance,
                        self._road_for_id(vehicle.current_road_id).length_meters,
                    )
                    occupancy[vehicle.current_road_id] = (
                        occupancy.get(vehicle.current_road_id, 0) + 1
                    )
            else:
                vehicle.position_meters = road.length_meters
                vehicle.waiting_time_seconds += self.time_step_seconds

        self.state.time_seconds += self.time_step_seconds
        self.state.refresh_road_occupancy()
        self._refresh_queues()
        return self.state

    def run(self, steps: int) -> SimulationState:
        """Run a fixed number of steps and return the final state."""

        if steps < 0:
            raise ValueError("step count cannot be negative")
        for _ in range(steps):
            self.step()
        return self.state

    def _advance_signal_clocks(self) -> None:
        for signal in self.state.intersection_states.values():
            signal.elapsed_seconds += self.time_step_seconds
            phase = signal.current_phase
            if signal.mode == "green" and signal.elapsed_seconds >= phase.max_green_seconds:
                signal.mode = "yellow"
                signal.elapsed_seconds = 0.0
            elif signal.mode == "yellow" and signal.elapsed_seconds >= phase.yellow_seconds:
                signal.mode = "all_red"
                signal.elapsed_seconds = 0.0
            elif signal.mode == "all_red" and signal.elapsed_seconds >= phase.all_red_seconds:
                signal.current_phase_index = (signal.current_phase_index + 1) % len(signal.phases)
                signal.mode = "green"
                signal.elapsed_seconds = 0.0

    def _can_exit_vehicle(
        self,
        vehicle: VehicleState,
        intersection_id: str,
        occupancy: dict[str, int],
        conditions: dict[str, EffectiveRoadCondition],
    ) -> bool:
        signal = self.state.intersection_states[intersection_id]
        if not signal.permits_movement:
            return False
        movement = self._movement_for_road(vehicle.current_road_id)
        if movement not in signal.current_phase.movements:
            return False
        if vehicle.route_index < len(vehicle.route_road_ids) - 1:
            next_road_id = vehicle.route_road_ids[vehicle.route_index + 1]
            next_road = self._road_for_id(next_road_id)
            if next_road.start_intersection_id != intersection_id:
                raise ValueError("vehicle route contains disconnected roads")
            next_condition = conditions[next_road_id]
            if not next_condition.available:
                return False
            if occupancy.get(next_road_id, 0) >= self._capacity_for_id(
                next_road_id, conditions
            ):
                return False
        return True

    def _refresh_queues(self) -> None:
        self.state.queue_lengths = {
            road_id: 0 for road_id in self.state.queue_lengths
        }
        for vehicle in self.state.vehicles.values():
            if vehicle.completed:
                continue
            road = self._road_for_id(vehicle.current_road_id)
            endpoint = road.end_intersection_id
            if vehicle.position_meters >= road.length_meters:
                self.state.queue_lengths[vehicle.current_road_id] += 1

    def _road_for_id(self, road_id: str):
        for start, end, attributes in self.graph.edges(data=True):
            if attributes["road_id"] == road_id:
                return attributes["road_segment"]
        raise ValueError(f"unknown road: {road_id}")

    def _capacity_for_id(
        self,
        road_id: str,
        conditions: dict[str, EffectiveRoadCondition] | None = None,
    ) -> int:
        for _, _, attributes in self.graph.edges(data=True):
            if attributes["road_id"] == road_id:
                base_capacity = attributes["capacity"]
                if conditions is None:
                    return base_capacity
                return max(
                    1,
                    int(base_capacity * conditions[road_id].capacity_multiplier),
                )
        raise ValueError(f"unknown road: {road_id}")

    def _road_endpoints(self, road_id: str) -> tuple[str, str]:
        road = self._road_for_id(road_id)
        return road.start_intersection_id, road.end_intersection_id

    def _has_road_id(self, road_id: str) -> bool:
        try:
            self._road_for_id(road_id)
        except ValueError:
            return False
        return True

    def _road_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                attributes["road_id"]
                for _, _, attributes in self.graph.edges(data=True)
            )
        )

    def _current_conditions(self) -> dict[str, EffectiveRoadCondition]:
        return self.road_conditions()

    def _condition_for(self, road_id: str) -> EffectiveRoadCondition:
        return self._current_conditions()[road_id]

    def _activate_scheduled_vehicles(
        self,
        conditions: dict[str, EffectiveRoadCondition],
        active_events: tuple[TrafficEvent, ...],
    ) -> None:
        active_event_ids = {event.id for event in active_events}
        pending: list[tuple[float, VehicleState, str | None]] = []
        for spawn_time, vehicle, arrival_event_id in self._scheduled_vehicles:
            if spawn_time > self.state.time_seconds:
                pending.append((spawn_time, vehicle, arrival_event_id))
                continue
            if arrival_event_id is not None and arrival_event_id not in active_event_ids:
                pending.append((spawn_time, vehicle, arrival_event_id))
                continue
            if not conditions[vehicle.current_road_id].available:
                pending.append((spawn_time, vehicle, arrival_event_id))
                continue
            if self.state.road_occupancy.get(vehicle.current_road_id, ()) and (
                len(self.state.road_occupancy[vehicle.current_road_id])
                >= self._capacity_for_id(vehicle.current_road_id, conditions)
            ):
                pending.append((spawn_time, vehicle, arrival_event_id))
                continue
            self.add_vehicles((vehicle,))
        self._scheduled_vehicles = pending

    @staticmethod
    def _movement_for_road(road_id: str) -> str:
        """Map the prototype corridor roads to their east-west phase."""

        return "east_west" if road_id.startswith("R_") else road_id