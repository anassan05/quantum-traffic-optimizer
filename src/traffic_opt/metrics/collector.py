"""Deterministic metrics derived from existing simulation snapshots."""

from dataclasses import dataclass, field
from statistics import fmean
from types import MappingProxyType
from typing import Mapping

import networkx as nx

from traffic_opt.domain.enums import VehicleType
from traffic_opt.simulation.state import SimulationState


def _default_rates(value: float) -> dict[VehicleType, float]:
    return {vehicle_type: value for vehicle_type in VehicleType}


@dataclass(frozen=True, slots=True)
class FuelModel:
    """Configurable simulation estimate, not physical fuel measurement.

    Distance fuel is litres per kilometre and idle fuel is litres per hour.
    The default values are illustrative experiment assumptions and should be
    replaced when calibrated project parameters become available.
    """

    fuel_rate_l_per_km: Mapping[VehicleType, float] = field(
        default_factory=lambda: _default_rates(0.08)
    )
    idle_fuel_rate_l_per_hour: Mapping[VehicleType, float] = field(
        default_factory=lambda: _default_rates(0.6)
    )
    co2_grams_per_litre: float = 2640.0

    def __post_init__(self) -> None:
        distance_rates = dict(self.fuel_rate_l_per_km)
        idle_rates = dict(self.idle_fuel_rate_l_per_hour)
        if set(distance_rates) != set(VehicleType) or set(idle_rates) != set(VehicleType):
            raise ValueError("fuel rates must define every VehicleType")
        if any(rate < 0 for rate in distance_rates.values()) or any(
            rate < 0 for rate in idle_rates.values()
        ):
            raise ValueError("fuel rates cannot be negative")
        if self.co2_grams_per_litre < 0:
            raise ValueError("CO2 emission factor cannot be negative")
        object.__setattr__(self, "fuel_rate_l_per_km", MappingProxyType(distance_rates))
        object.__setattr__(self, "idle_fuel_rate_l_per_hour", MappingProxyType(idle_rates))


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Immutable metrics calculated at one simulation time."""

    simulation_time_seconds: float
    total_vehicles: int
    active_vehicles: int
    completed_vehicles: int
    total_waiting_time_seconds: float
    average_waiting_time_seconds: float
    maximum_waiting_time_seconds: float
    total_queue_length: int
    average_queue_length: float
    maximum_queue_length: int
    throughput_vehicles_per_second: float
    total_travel_time_seconds: float
    average_travel_time_seconds: float
    maximum_travel_time_seconds: float
    fuel_consumed_litres: float
    co2_emissions_grams: float
    ambulance_count: int
    completed_ambulances: int
    average_ambulance_travel_time_seconds: float
    average_ambulance_waiting_time_seconds: float


@dataclass(frozen=True, slots=True)
class MetricsAggregate:
    """Deterministic aggregate over collected snapshots."""

    observation_duration_seconds: float
    total_vehicles: int
    completed_vehicles: int
    average_waiting_time_seconds: float
    average_queue_length: float
    throughput_vehicles_per_second: float
    average_travel_time_seconds: float
    total_fuel_consumed_litres: float
    total_co2_emissions_grams: float


class MetricsCollector:
    """Collect performance metrics without controlling simulation state."""

    def __init__(
        self,
        topology: nx.DiGraph,
        *,
        fuel_model: FuelModel | None = None,
        spawn_times: Mapping[str, float] | None = None,
    ) -> None:
        self.topology = topology
        self.fuel_model = fuel_model or FuelModel()
        self.spawn_times = dict(spawn_times or {})
        if any(time < 0 for time in self.spawn_times.values()):
            raise ValueError("spawn times cannot be negative")
        self._completion_times: dict[str, float] = {}
        self._first_seen_times: dict[str, float] = {}
        self._snapshots: list[MetricsSnapshot] = []

    @property
    def snapshots(self) -> tuple[MetricsSnapshot, ...]:
        return tuple(self._snapshots)

    def collect(self, state: SimulationState) -> MetricsSnapshot:
        """Calculate and store one immutable snapshot from simulation state."""

        if state.time_seconds < 0:
            raise ValueError("simulation time cannot be negative")
        for vehicle in state.vehicles.values():
            self._first_seen_times.setdefault(vehicle.id, state.time_seconds)
            if vehicle.completed:
                self._completion_times.setdefault(vehicle.id, state.time_seconds)

        vehicles = tuple(state.vehicles.values())
        completed = tuple(vehicle for vehicle in vehicles if vehicle.completed)
        waiting_times = tuple(vehicle.waiting_time_seconds for vehicle in vehicles)
        queue_values = tuple(state.queue_lengths.values())
        travel_times = tuple(self._travel_time(vehicle) for vehicle in completed)
        fuel = sum(self._fuel_for_vehicle(vehicle) for vehicle in vehicles)
        ambulances = tuple(
            vehicle for vehicle in vehicles if vehicle.vehicle_type is VehicleType.AMBULANCE
        )
        completed_ambulances = tuple(vehicle for vehicle in ambulances if vehicle.completed)
        ambulance_travel = tuple(self._travel_time(vehicle) for vehicle in completed_ambulances)

        snapshot = MetricsSnapshot(
            simulation_time_seconds=state.time_seconds,
            total_vehicles=len(vehicles),
            active_vehicles=len(vehicles) - len(completed),
            completed_vehicles=len(completed),
            total_waiting_time_seconds=sum(waiting_times),
            average_waiting_time_seconds=_average(waiting_times),
            maximum_waiting_time_seconds=max(waiting_times, default=0.0),
            total_queue_length=sum(queue_values),
            average_queue_length=_average(queue_values),
            maximum_queue_length=max(queue_values, default=0),
            throughput_vehicles_per_second=(
                len(completed) / state.time_seconds if state.time_seconds > 0 else 0.0
            ),
            total_travel_time_seconds=sum(travel_times),
            average_travel_time_seconds=_average(travel_times),
            maximum_travel_time_seconds=max(travel_times, default=0.0),
            fuel_consumed_litres=fuel,
            co2_emissions_grams=fuel * self.fuel_model.co2_grams_per_litre,
            ambulance_count=len(ambulances),
            completed_ambulances=len(completed_ambulances),
            average_ambulance_travel_time_seconds=_average(ambulance_travel),
            average_ambulance_waiting_time_seconds=_average(
                tuple(vehicle.waiting_time_seconds for vehicle in ambulances)
            ),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def aggregate(self) -> MetricsAggregate:
        """Aggregate collected observations, or return a zero aggregate."""

        if not self._snapshots:
            return MetricsAggregate(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        first, last = self._snapshots[0], self._snapshots[-1]
        duration = max(0.0, last.simulation_time_seconds - first.simulation_time_seconds)
        return MetricsAggregate(
            observation_duration_seconds=duration,
            total_vehicles=last.total_vehicles,
            completed_vehicles=last.completed_vehicles,
            average_waiting_time_seconds=fmean(
                snapshot.average_waiting_time_seconds for snapshot in self._snapshots
            ),
            average_queue_length=fmean(
                snapshot.average_queue_length for snapshot in self._snapshots
            ),
            throughput_vehicles_per_second=(
                last.completed_vehicles / duration if duration > 0 else 0.0
            ),
            average_travel_time_seconds=last.average_travel_time_seconds,
            total_fuel_consumed_litres=last.fuel_consumed_litres,
            total_co2_emissions_grams=last.co2_emissions_grams,
        )

    def _travel_time(self, vehicle) -> float:
        spawn_time = self.spawn_times.get(
            vehicle.id,
            self._first_seen_times.get(vehicle.id, self._completion_times.get(vehicle.id, 0.0)),
        )
        completion_time = self._completion_times.get(vehicle.id)
        if completion_time is None:
            return 0.0
        return max(0.0, completion_time - spawn_time)

    def _fuel_for_vehicle(self, vehicle) -> float:
        distance_km = self._distance_travelled_km(vehicle)
        distance_fuel = distance_km * self.fuel_model.fuel_rate_l_per_km[vehicle.vehicle_type]
        idle_fuel = (
            vehicle.waiting_time_seconds / 3600
        ) * self.fuel_model.idle_fuel_rate_l_per_hour[vehicle.vehicle_type]
        return distance_fuel + idle_fuel

    def _distance_travelled_km(self, vehicle) -> float:
        if not 0 <= vehicle.route_index < len(vehicle.route_road_ids):
            raise ValueError(f"vehicle has an invalid route index: {vehicle.id}")
        if vehicle.current_road_id != vehicle.route_road_ids[vehicle.route_index]:
            raise ValueError(f"vehicle current road does not match route: {vehicle.id}")
        lengths = {
            attributes["road_id"]: attributes["length_meters"]
            for _, _, attributes in self.topology.edges(data=True)
            if "road_id" in attributes and "length_meters" in attributes
        }
        try:
            travelled_meters = sum(
                lengths[road_id] for road_id in vehicle.route_road_ids[: vehicle.route_index]
            ) + min(vehicle.position_meters, lengths[vehicle.current_road_id])
        except KeyError as error:
            raise ValueError(f"vehicle route references an unknown road: {vehicle.id}") from error
        return travelled_meters / 1000


def _average(values) -> float:
    return fmean(values) if values else 0.0