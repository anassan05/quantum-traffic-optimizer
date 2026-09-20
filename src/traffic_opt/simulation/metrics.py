"""Deterministic simulation metrics and environmental estimates."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import networkx as nx

from .state import SimulationState

# Transparent engineering estimates, not measured environmental data.
DEFAULT_FUEL_LITERS_PER_KILOMETER = 0.08
DEFAULT_IDLE_FUEL_LITERS_PER_HOUR = 0.6
DEFAULT_CO2_KILOGRAMS_PER_LITER = 2.31


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    """Immutable metrics snapshot for one simulation run."""

    total_vehicles_completed: int = 0
    average_waiting_time_seconds: float = 0.0
    maximum_waiting_time_seconds: float = 0.0
    total_waiting_time_seconds: float = 0.0
    average_queue_length: float = 0.0
    maximum_queue_length: int = 0
    throughput_vehicles_per_second: float = 0.0
    estimated_fuel_consumption_liters: float = 0.0
    estimated_co2_emissions_kg: float = 0.0
    emergency_vehicle_travel_time_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    emergency_vehicle_delay_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def average_emergency_vehicle_travel_time_seconds(self) -> float:
        """Average observed travel time for emergency vehicles."""

        return _average(self.emergency_vehicle_travel_time_seconds.values())

    @property
    def average_emergency_vehicle_delay_seconds(self) -> float:
        """Average waiting delay for emergency vehicles."""

        return _average(self.emergency_vehicle_delay_seconds.values())


class MetricsCollector:
    """Collect deterministic metrics without mutating simulation behavior."""

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        fuel_liters_per_kilometer: float = DEFAULT_FUEL_LITERS_PER_KILOMETER,
        idle_fuel_liters_per_hour: float = DEFAULT_IDLE_FUEL_LITERS_PER_HOUR,
        co2_kilograms_per_liter: float = DEFAULT_CO2_KILOGRAMS_PER_LITER,
    ) -> None:
        if fuel_liters_per_kilometer < 0:
            raise ValueError("fuel rate cannot be negative")
        if idle_fuel_liters_per_hour < 0:
            raise ValueError("idle fuel rate cannot be negative")
        if co2_kilograms_per_liter < 0:
            raise ValueError("CO2 rate cannot be negative")
        self._graph = graph
        self._fuel_rate = fuel_liters_per_kilometer
        self._idle_rate = idle_fuel_liters_per_hour
        self._co2_rate = co2_kilograms_per_liter
        self._queue_samples: list[float] = []
        self._maximum_queue_length = 0
        self._start_times: dict[str, float] = {}
        self._completion_times: dict[str, float] = {}

    def record(self, state: SimulationState) -> None:
        """Record one state observation at its current simulation time."""

        if state.time_seconds < 0:
            raise ValueError("simulation time cannot be negative")
        self._queue_samples.append(_average(state.queue_lengths.values()))
        self._maximum_queue_length = max(
            self._maximum_queue_length,
            max(state.queue_lengths.values(), default=0),
        )
        for vehicle in state.vehicles.values():
            self._start_times.setdefault(vehicle.id, state.time_seconds)
            if vehicle.completed:
                self._completion_times.setdefault(vehicle.id, state.time_seconds)

    observe = record

    def snapshot(self, state: SimulationState) -> SimulationMetrics:
        """Calculate a deterministic metrics snapshot from the latest state."""

        if state.time_seconds < 0:
            raise ValueError("simulation time cannot be negative")
        vehicles = tuple(state.vehicles.values())
        completed = tuple(vehicle for vehicle in vehicles if vehicle.completed)
        total_waiting = sum(vehicle.waiting_time_seconds for vehicle in vehicles)
        queue_max = max(
            self._maximum_queue_length,
            max(state.queue_lengths.values(), default=0),
        )
        duration = state.time_seconds
        emergency_travel: dict[str, float] = {}
        emergency_delay: dict[str, float] = {}
        fuel = 0.0
        for vehicle in vehicles:
            distance_km = self._distance_kilometers(vehicle)
            fuel += distance_km * self._fuel_rate
            fuel += vehicle.waiting_time_seconds / 3600 * self._idle_rate
            if vehicle.is_emergency:
                end_time = self._completion_times.get(vehicle.id, duration)
                start_time = self._start_times.get(vehicle.id, 0.0)
                emergency_travel[vehicle.id] = max(0.0, end_time - start_time)
                emergency_delay[vehicle.id] = vehicle.waiting_time_seconds

        return SimulationMetrics(
            total_vehicles_completed=len(completed),
            average_waiting_time_seconds=_average(
                vehicle.waiting_time_seconds for vehicle in vehicles
            ),
            maximum_waiting_time_seconds=max(
                (vehicle.waiting_time_seconds for vehicle in vehicles), default=0.0
            ),
            total_waiting_time_seconds=total_waiting,
            average_queue_length=_average(self._queue_samples),
            maximum_queue_length=queue_max,
            throughput_vehicles_per_second=(len(completed) / duration if duration else 0.0),
            estimated_fuel_consumption_liters=fuel,
            estimated_co2_emissions_kg=fuel * self._co2_rate,
            emergency_vehicle_travel_time_seconds=MappingProxyType(emergency_travel),
            emergency_vehicle_delay_seconds=MappingProxyType(emergency_delay),
        )

    def _distance_kilometers(self, vehicle) -> float:
        distance_meters = 0.0
        for road_id in vehicle.route_road_ids[: vehicle.route_index]:
            distance_meters += self._road_length_meters(road_id)
        distance_meters += min(
            vehicle.position_meters,
            self._road_length_meters(vehicle.current_road_id),
        )
        return distance_meters / 1000

    def _road_length_meters(self, road_id: str) -> float:
        for _, _, attributes in self._graph.edges(data=True):
            if attributes["road_id"] == road_id:
                return float(attributes["length_meters"])
        raise ValueError(f"unknown road: {road_id}")


def _average(values) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0
