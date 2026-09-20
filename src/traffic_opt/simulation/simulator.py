"""Small deterministic discrete-time traffic simulator."""

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx

from traffic_opt.controllers.emergency_corridor import EmergencyCorridorManager
from traffic_opt.domain.models import Intersection, TrafficEvent

from .events import EventManager
from .metrics import MetricsCollector, SimulationMetrics
from .state import IntersectionSignalState, SimulationState
from .vehicles import VehicleState, create_emergency_vehicle


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
    state: SimulationState = field(init=False)
    event_manager: EventManager = field(init=False)
    corridor_manager: EmergencyCorridorManager = field(init=False)
    metrics_collector: MetricsCollector = field(init=False)

    def __post_init__(self) -> None:
        if self.time_step_seconds <= 0:
            raise ValueError("time step must be greater than zero")
        self.event_manager = EventManager(self.events, self.graph)
        self.corridor_manager = EmergencyCorridorManager(self.graph, self.intersections)
        self.state = SimulationState()
        self._refresh_active_event_ids()
        self._refresh_emergency_vehicle_ids()
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
        self.metrics_collector = MetricsCollector(self.graph)
        self.metrics_collector.record(self.state)

    @property
    def metrics(self) -> SimulationMetrics:
        """Return deterministic metrics for the current simulation state."""

        return self.metrics_collector.snapshot(self.state)

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
            if self._is_road_closed(vehicle.current_road_id):
                raise ValueError(f"road is closed: {vehicle.current_road_id}")
            if occupancy.get(vehicle.current_road_id, 0) >= self._capacity_for_id(
                vehicle.current_road_id
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
        self._refresh_emergency_vehicle_ids()
        self.metrics_collector.record(self.state)

    def add_emergency_vehicle(
        self,
        vehicle_id: str,
        origin_intersection_id: str,
        destination_intersection_id: str,
        route_road_ids: tuple[str, ...],
        speed_kmh: float | None = None,
    ) -> VehicleState:
        """Create and add an ambulance using the normal admission checks."""

        vehicle = create_emergency_vehicle(
            self.graph,
            vehicle_id,
            origin_intersection_id,
            destination_intersection_id,
            route_road_ids,
            speed_kmh,
        )
        self.add_vehicles((vehicle,))
        return vehicle

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

    def step(self) -> SimulationState:
        """Advance signals and vehicles by one configured time step."""

        self.corridor_manager.update(
            self.state.vehicles.values(),
            self.state.intersection_states,
            self.state.time_seconds,
            self.event_manager.is_road_closed,
        )
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
            distance = vehicle.speed_kmh / 3.6 * self.time_step_seconds
            if vehicle.position_meters + distance < road.length_meters:
                vehicle.position_meters += distance
                continue

            remaining_distance = road.length_meters - vehicle.position_meters
            endpoint = road.end_intersection_id
            if self._can_exit_vehicle(vehicle, endpoint, occupancy):
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
        self._refresh_active_event_ids()
        self.state.refresh_road_occupancy()
        self._refresh_queues()
        self._refresh_emergency_vehicle_ids()
        self.metrics_collector.record(self.state)
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
            requested_phase = self.corridor_manager.requested_phase(
                signal.intersection_id
            )
            if (
                signal.mode == "green"
                and requested_phase is not None
                and requested_phase != signal.current_phase_index
                and signal.elapsed_seconds >= phase.min_green_seconds
            ):
                signal.mode = "yellow"
                signal.elapsed_seconds = 0.0
            elif signal.mode == "green" and signal.elapsed_seconds >= phase.max_green_seconds:
                signal.mode = "yellow"
                signal.elapsed_seconds = 0.0
            elif signal.mode == "yellow" and signal.elapsed_seconds >= phase.yellow_seconds:
                signal.mode = "all_red"
                signal.elapsed_seconds = 0.0
            elif signal.mode == "all_red" and signal.elapsed_seconds >= phase.all_red_seconds:
                requested_phase = self.corridor_manager.requested_phase(
                    signal.intersection_id
                )
                signal.current_phase_index = (
                    requested_phase
                    if requested_phase is not None
                    else (signal.current_phase_index + 1) % len(signal.phases)
                )
                signal.mode = "green"
                signal.elapsed_seconds = 0.0

    def _can_exit_vehicle(
        self,
        vehicle: VehicleState,
        intersection_id: str,
        occupancy: dict[str, int],
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
            if self._is_road_closed(next_road_id):
                return False
            if occupancy.get(next_road_id, 0) >= self._capacity_for_id(next_road_id):
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

    def _capacity_for_id(self, road_id: str) -> int:
        for _, _, attributes in self.graph.edges(data=True):
            if attributes["road_id"] == road_id:
                return attributes["capacity"]
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

    def _is_road_closed(self, road_id: str) -> bool:
        return self.event_manager.is_road_closed(road_id, self._event_time())

    def _refresh_active_event_ids(self) -> None:
        self.state.active_event_ids = self.event_manager.active_event_ids(
            self._event_time()
        )

    def _refresh_emergency_vehicle_ids(self) -> None:
        self.state.emergency_vehicle_ids = tuple(
            sorted(
                vehicle.id
                for vehicle in self.state.vehicles.values()
                if vehicle.is_emergency
            )
        )

    def _event_time(self) -> int:
        return int(self.state.time_seconds)

    @staticmethod
    def _movement_for_road(road_id: str) -> str:
        """Map the prototype corridor roads to their east-west phase."""

        return "east_west" if road_id.startswith("R_") else road_id