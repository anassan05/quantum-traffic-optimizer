"""Vehicle state and deterministic vehicle generation for simulation."""

from dataclasses import dataclass
from random import Random
from typing import Final, Sequence

import networkx as nx

from traffic_opt.domain.enums import VehicleType
from traffic_opt.domain.models import Vehicle as DomainVehicle

DEFAULT_SPEED_KMH: Final[dict[VehicleType, float]] = {
    VehicleType.CAR: 40.0,
    VehicleType.BUS: 30.0,
    VehicleType.TRUCK: 25.0,
    VehicleType.AMBULANCE: 50.0,
}


@dataclass(slots=True)
class VehicleState:
    """Mutable simulation state for one vehicle.

    ``route_index`` points at ``current_road_id``. Vehicles never skip a road
    or an intersection, and emergency vehicles use exactly the same movement
    permission checks as regular vehicles.
    """

    id: str
    vehicle_type: VehicleType
    origin_intersection_id: str
    destination_intersection_id: str
    route_road_ids: tuple[str, ...]
    current_road_id: str
    speed_kmh: float = 40.0
    position_meters: float = 0.0
    waiting_time_seconds: float = 0.0
    route_index: int = 0
    completed: bool = False
    is_emergency: bool = False

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("vehicle id must be non-empty")
        if not isinstance(self.is_emergency, bool):
            raise ValueError("vehicle emergency flag must be boolean")
        if self.vehicle_type is VehicleType.AMBULANCE:
            self.is_emergency = True
        if not self.route_road_ids:
            raise ValueError("vehicle route must contain at least one road")
        if self.current_road_id not in self.route_road_ids:
            raise ValueError("current road must be included in vehicle route")
        if not 0 <= self.route_index < len(self.route_road_ids):
            raise ValueError("vehicle route index is out of range")
        if self.route_road_ids[self.route_index] != self.current_road_id:
            raise ValueError("route index must point to current road")
        if self.speed_kmh <= 0:
            raise ValueError("vehicle speed must be greater than zero")
        if self.position_meters < 0 or self.waiting_time_seconds < 0:
            raise ValueError("vehicle position and waiting time cannot be negative")

    @classmethod
    def from_domain_vehicle(
        cls,
        vehicle: DomainVehicle,
        speed_kmh: float | None = None,
    ) -> "VehicleState":
        """Create mutable simulation state from an existing domain vehicle."""

        current_road_id = vehicle.current_road_id or vehicle.route_road_ids[0]
        return cls(
            id=vehicle.id,
            vehicle_type=vehicle.vehicle_type,
            origin_intersection_id=vehicle.origin_intersection_id,
            destination_intersection_id=vehicle.destination_intersection_id,
            route_road_ids=vehicle.route_road_ids,
            current_road_id=current_road_id,
            speed_kmh=speed_kmh or DEFAULT_SPEED_KMH[vehicle.vehicle_type],
            position_meters=vehicle.position_meters,
            waiting_time_seconds=vehicle.waiting_time_seconds,
            route_index=vehicle.route_road_ids.index(current_road_id),
            completed=vehicle.completed,
            is_emergency=vehicle.is_emergency,
        )

    @property
    def is_emergency_vehicle(self) -> bool:
        """Whether this vehicle is identified as an emergency vehicle."""

        return self.is_emergency


def create_emergency_vehicle(
    graph: nx.DiGraph,
    vehicle_id: str,
    origin_intersection_id: str,
    destination_intersection_id: str,
    route_road_ids: tuple[str, ...],
    speed_kmh: float | None = None,
) -> VehicleState:
    """Create an ambulance after validating its route against ``graph``."""

    if not isinstance(vehicle_id, str) or not vehicle_id.strip():
        raise ValueError("vehicle ID must be non-empty")
    if not isinstance(origin_intersection_id, str) or not origin_intersection_id.strip():
        raise ValueError("vehicle origin must be non-empty")
    if not isinstance(destination_intersection_id, str) or not destination_intersection_id.strip():
        raise ValueError("vehicle destination must be non-empty")
    if not isinstance(route_road_ids, tuple) or not route_road_ids:
        raise ValueError("emergency vehicle route must contain road IDs")

    roads_by_id = {
        attributes["road_id"]: (start, end)
        for start, end, attributes in graph.edges(data=True)
    }
    missing = [road_id for road_id in route_road_ids if road_id not in roads_by_id]
    if missing:
        raise ValueError(f"unknown route road: {missing[0]}")
    if roads_by_id[route_road_ids[0]][0] != origin_intersection_id:
        raise ValueError("emergency route does not start at origin")
    for current_road_id, next_road_id in zip(route_road_ids, route_road_ids[1:]):
        if roads_by_id[current_road_id][1] != roads_by_id[next_road_id][0]:
            raise ValueError("emergency route contains disconnected roads")
    if roads_by_id[route_road_ids[-1]][1] != destination_intersection_id:
        raise ValueError("emergency route does not end at destination")

    return VehicleState(
        id=vehicle_id,
        vehicle_type=VehicleType.AMBULANCE,
        origin_intersection_id=origin_intersection_id,
        destination_intersection_id=destination_intersection_id,
        route_road_ids=route_road_ids,
        current_road_id=route_road_ids[0],
        speed_kmh=speed_kmh or DEFAULT_SPEED_KMH[VehicleType.AMBULANCE],
        is_emergency=True,
    )


def generate_vehicles(
    graph: nx.DiGraph,
    count: int,
    seed: int = 0,
    vehicle_type: VehicleType = VehicleType.CAR,
    excluded_road_ids: frozenset[str] = frozenset(),
) -> tuple[VehicleState, ...]:
    """Generate deterministic vehicles from sorted graph roads.

    Generated vehicles use one-road routes, making this factory useful for
    deterministic smoke tests and later extensible route generation.
    """

    if count < 0:
        raise ValueError("vehicle count cannot be negative")
    roads = sorted(
        (
            attributes["road_id"],
            start,
            end,
        )
        for start, end, attributes in graph.edges(data=True)
        if attributes["road_id"] not in excluded_road_ids
    )
    if not roads and count:
        raise ValueError("cannot generate vehicles without road segments")
    random = Random(seed)
    vehicles = []
    for number in range(1, count + 1):
        road_id, origin, destination = roads[random.randrange(len(roads))]
        vehicles.append(
            VehicleState(
                id=f"V{number:04d}",
                vehicle_type=vehicle_type,
                origin_intersection_id=origin,
                destination_intersection_id=destination,
                route_road_ids=(road_id,),
                current_road_id=road_id,
                speed_kmh=DEFAULT_SPEED_KMH[vehicle_type],
            )
        )
    return tuple(vehicles)