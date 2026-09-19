"""Default signal configurations for the prototype corridor."""

from typing import Final

import networkx as nx

from traffic_opt.domain.constraints import DomainValidationError
from traffic_opt.domain.models import Intersection, SignalPhaseConfig

DEFAULT_MIN_GREEN_SECONDS: Final[int] = 10
DEFAULT_MAX_GREEN_SECONDS: Final[int] = 60
DEFAULT_YELLOW_SECONDS: Final[int] = 3
DEFAULT_ALL_RED_SECONDS: Final[int] = 1


def create_default_signal_phases() -> tuple[SignalPhaseConfig, ...]:
    """Return the two mutually exclusive through-movement phases.

    Clearance durations are attached to each green phase and must be consumed
    before a following green phase can be applied.
    """

    return (
        SignalPhaseConfig(
            name="north_south_green",
            movements=frozenset({"north_south"}),
            min_green_seconds=DEFAULT_MIN_GREEN_SECONDS,
            max_green_seconds=DEFAULT_MAX_GREEN_SECONDS,
            yellow_seconds=DEFAULT_YELLOW_SECONDS,
            all_red_seconds=DEFAULT_ALL_RED_SECONDS,
        ),
        SignalPhaseConfig(
            name="east_west_green",
            movements=frozenset({"east_west"}),
            min_green_seconds=DEFAULT_MIN_GREEN_SECONDS,
            max_green_seconds=DEFAULT_MAX_GREEN_SECONDS,
            yellow_seconds=DEFAULT_YELLOW_SECONDS,
            all_red_seconds=DEFAULT_ALL_RED_SECONDS,
        ),
    )


def create_default_intersections(graph: nx.DiGraph) -> tuple[Intersection, ...]:
    """Create signalized domain intersections from a topology graph.

    Nodes are returned in sorted order, and road IDs are sorted to make the
    resulting configuration deterministic across runs.
    """

    expected_ids = {"I1", "I2", "I3", "I4"}
    if set(graph.nodes) != expected_ids:
        raise DomainValidationError(
            "default intersections require exactly nodes I1, I2, I3, and I4"
        )

    phases = create_default_signal_phases()
    intersections = []
    for intersection_id in sorted(graph.nodes):
        incoming = tuple(
            sorted(
                graph.edges[neighbor, intersection_id]["road_id"]
                for neighbor in graph.predecessors(intersection_id)
            )
        )
        outgoing = tuple(
            sorted(
                graph.edges[intersection_id, neighbor]["road_id"]
                for neighbor in graph.successors(intersection_id)
            )
        )
        intersections.append(
            Intersection(intersection_id, incoming, outgoing, phases)
        )
    return tuple(intersections)