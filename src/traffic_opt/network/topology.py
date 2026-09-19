"""Deterministic NetworkX road topology for the initial prototype."""

from typing import Final

import networkx as nx

from traffic_opt.domain.models import RoadSegment

INTERSECTION_IDS: Final[tuple[str, ...]] = ("I1", "I2", "I3", "I4")
CORRIDOR_LINKS: Final[tuple[tuple[str, str], ...]] = (
    ("I1", "I2"),
    ("I2", "I3"),
    ("I3", "I4"),
)
DEFAULT_ROAD_LENGTH_METERS: Final[float] = 250.0
DEFAULT_ROAD_CAPACITY: Final[int] = 40
DEFAULT_FREE_FLOW_SPEED_KMH: Final[float] = 45.0


def _add_directed_road(
    graph: nx.DiGraph,
    start_intersection_id: str,
    end_intersection_id: str,
) -> None:
    road_id = f"R_{start_intersection_id}_{end_intersection_id}"
    road = RoadSegment(
        id=road_id,
        start_intersection_id=start_intersection_id,
        end_intersection_id=end_intersection_id,
        capacity=DEFAULT_ROAD_CAPACITY,
        length_meters=DEFAULT_ROAD_LENGTH_METERS,
        free_flow_speed_kmh=DEFAULT_FREE_FLOW_SPEED_KMH,
    )
    travel_time_seconds = road.length_meters / (
        road.free_flow_speed_kmh / 3.6
    )
    graph.add_edge(
        start_intersection_id,
        end_intersection_id,
        road_id=road.id,
        road_segment=road,
        length_meters=road.length_meters,
        capacity=road.capacity,
        free_flow_speed_kmh=road.free_flow_speed_kmh,
        free_flow_travel_time_seconds=travel_time_seconds,
    )


def create_default_topology() -> nx.DiGraph:
    """Create the deterministic four-intersection bidirectional corridor.

    The graph contains six directed road segments: one in each direction for
    the three links I1-I2, I2-I3, and I3-I4. No random state is used.
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (intersection_id, {"intersection_id": intersection_id})
        for intersection_id in INTERSECTION_IDS
    )
    for start_intersection_id, end_intersection_id in CORRIDOR_LINKS:
        _add_directed_road(graph, start_intersection_id, end_intersection_id)
        _add_directed_road(graph, end_intersection_id, start_intersection_id)
    return graph