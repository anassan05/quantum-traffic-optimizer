"""Deterministic management of scheduled traffic events."""

import networkx as nx

from traffic_opt.domain.constraints import DomainValidationError
from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent


class EventManager:
    """Validate and query the scheduled events for a road network."""

    def __init__(
        self,
        events: tuple[TrafficEvent, ...],
        graph: nx.Graph,
    ) -> None:
        if not isinstance(events, tuple):
            raise TypeError("events must be a tuple of TrafficEvent objects")
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX graph")

        self._events = tuple(events)
        self._road_ids = self._road_ids_from_graph(graph)
        self._validate_events()
        self._events_by_id = {event.id: event for event in self._events}

    def active_events(self, time_seconds: int) -> tuple[TrafficEvent, ...]:
        """Return events active at ``time_seconds``, sorted by event ID."""

        self._validate_time(time_seconds)
        active = (
            event
            for event in self._events
            if event.start_time_seconds
            <= time_seconds
            < event.start_time_seconds + event.duration_seconds
        )
        return tuple(sorted(active, key=lambda event: event.id))

    def active_event_ids(self, time_seconds: int) -> tuple[str, ...]:
        """Return active event IDs in deterministic sorted order."""

        return tuple(event.id for event in self.active_events(time_seconds))

    def is_road_closed(self, road_id: str, time_seconds: int) -> bool:
        """Return whether a road has an active road-closure event."""

        if not isinstance(road_id, str) or not road_id:
            raise TypeError("road_id must be a non-empty string")
        return any(
            event.event_type is TrafficEventType.ROAD_CLOSURE
            and road_id in event.affected_road_ids
            for event in self.active_events(time_seconds)
        )

    def affected_road_ids(self, time_seconds: int) -> tuple[str, ...]:
        """Return unique active affected road IDs in sorted order."""

        return tuple(
            sorted(
                {
                    road_id
                    for event in self.active_events(time_seconds)
                    for road_id in event.affected_road_ids
                }
            )
        )

    @staticmethod
    def _road_ids_from_graph(graph: nx.Graph) -> frozenset[str]:
        road_ids: set[str] = set()
        for edge in graph.edges(data=True):
            if len(edge) != 3:
                raise DomainValidationError("graph edges must include road attributes")
            road_id = edge[2].get("road_id")
            if not isinstance(road_id, str) or not road_id:
                raise DomainValidationError("graph edges must have non-empty road_id attributes")
            road_ids.add(road_id)
        return frozenset(road_ids)

    def _validate_events(self) -> None:
        event_ids: set[str] = set()
        events_by_road: dict[str, list[TrafficEvent]] = {}
        for event in self._events:
            if not isinstance(event, TrafficEvent):
                raise TypeError("events must contain only TrafficEvent objects")
            if event.id in event_ids:
                raise DomainValidationError(f"duplicate event ID: {event.id}")
            event_ids.add(event.id)
            missing_roads = set(event.affected_road_ids) - self._road_ids
            if missing_roads:
                missing = ", ".join(sorted(missing_roads))
                raise DomainValidationError(
                    f"event {event.id} references unknown road IDs: {missing}"
                )
            for road_id in event.affected_road_ids:
                events_by_road.setdefault(road_id, []).append(event)

        for road_id, road_events in events_by_road.items():
            ordered = sorted(road_events, key=lambda event: event.start_time_seconds)
            for previous, current in zip(ordered, ordered[1:]):
                previous_end = previous.start_time_seconds + previous.duration_seconds
                if current.start_time_seconds < previous_end:
                    raise DomainValidationError(
                        f"events {previous.id} and {current.id} overlap on road {road_id}"
                    )

    @staticmethod
    def _validate_time(time_seconds: int) -> None:
        if isinstance(time_seconds, bool) or not isinstance(time_seconds, int):
            raise TypeError("time_seconds must be a non-negative integer")
        if time_seconds < 0:
            raise ValueError("time_seconds cannot be negative")
