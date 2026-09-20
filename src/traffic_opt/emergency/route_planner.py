"""Static shortest-path planning for virtual ambulances."""

from dataclasses import dataclass

import networkx as nx

from traffic_opt.demand import VehicleDemand
from traffic_opt.domain.enums import VehicleType


@dataclass(frozen=True, slots=True)
class RouteDetails:
    """Optional descriptive information for a planned road-ID route."""

    intersection_ids: tuple[str, ...]
    road_ids: tuple[str, ...]
    total_distance_meters: float


class AmbulanceRoutePlanner:
    """Plan deterministic static routes using NetworkX shortest paths."""

    def __init__(self, topology: nx.Graph) -> None:
        self.topology = topology

    def plan(
        self,
        origin_intersection_id: str,
        destination_intersection_id: str,
    ) -> tuple[str, ...]:
        """Return road IDs on the shortest physical-distance route."""

        return self.plan_details(
            origin_intersection_id,
            destination_intersection_id,
        ).road_ids

    def plan_for_demand(self, demand: VehicleDemand) -> tuple[str, ...]:
        """Plan a route only for a demand item representing an ambulance."""

        if demand.vehicle_type is not VehicleType.AMBULANCE:
            raise ValueError("ambulance route planning requires an ambulance demand")
        return self.plan(
            demand.origin_intersection_id,
            demand.destination_intersection_id,
        )

    def plan_details(
        self,
        origin_intersection_id: str,
        destination_intersection_id: str,
    ) -> RouteDetails:
        """Return intersections, road IDs, and total static distance."""

        self._validate_endpoints(origin_intersection_id, destination_intersection_id)
        try:
            intersection_ids = tuple(
                nx.shortest_path(
                    self.topology,
                    origin_intersection_id,
                    destination_intersection_id,
                    weight="length_meters",
                )
            )
        except nx.NetworkXNoPath as error:
            raise ValueError(
                "no route exists between the requested intersections"
            ) from error

        road_ids: list[str] = []
        total_distance_meters = 0.0
        for start, end in zip(intersection_ids, intersection_ids[1:]):
            attributes = self.topology.edges[start, end]
            try:
                road_id = attributes["road_id"]
                distance = attributes["length_meters"]
            except KeyError as error:
                raise ValueError("topology route edges require road_id and length_meters") from error
            road_ids.append(road_id)
            total_distance_meters += distance

        return RouteDetails(
            intersection_ids=intersection_ids,
            road_ids=tuple(road_ids),
            total_distance_meters=total_distance_meters,
        )

    def _validate_endpoints(self, origin: str, destination: str) -> None:
        if origin not in self.topology:
            raise ValueError(f"origin intersection is not in the topology: {origin}")
        if destination not in self.topology:
            raise ValueError(
                f"destination intersection is not in the topology: {destination}"
            )
        if origin == destination:
            raise ValueError("origin and destination intersections must differ")