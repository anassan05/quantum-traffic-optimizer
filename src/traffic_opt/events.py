"""Declarative traffic-event creation and deterministic scheduling."""

from dataclasses import dataclass
from enum import Enum
import random
from typing import Final, Iterable, Mapping

import networkx as nx

from traffic_opt.demand import VehicleDemand
from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import TrafficEvent


SUPPORTED_EVENT_TYPES: Final[frozenset[TrafficEventType]] = frozenset(
    {
        TrafficEventType.CONGESTION,
        TrafficEventType.ACCIDENT,
        TrafficEventType.ROAD_CLOSURE,
        TrafficEventType.SPECIAL_DEMAND,
    }
)


class EventScenario(str, Enum):
    """Named event presets available to callers."""

    DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class TrafficEventConfig:
    """Bounds and controls for deterministic event generation."""

    random_seed: int = 0
    simulation_horizon: int = 3600
    event_count: int = 0
    enabled_event_types: tuple[TrafficEventType, ...] = (
        TrafficEventType.CONGESTION,
        TrafficEventType.ACCIDENT,
        TrafficEventType.ROAD_CLOSURE,
    )
    minimum_duration_seconds: int = 60
    maximum_duration_seconds: int = 600

    def __post_init__(self) -> None:
        if self.simulation_horizon < 0:
            raise ValueError("simulation horizon cannot be negative")
        if self.event_count < 0:
            raise ValueError("event count cannot be negative")
        if self.minimum_duration_seconds <= 0:
            raise ValueError("minimum event duration must be positive")
        if self.maximum_duration_seconds < self.minimum_duration_seconds:
            raise ValueError("maximum event duration must not be less than minimum")
        enabled = tuple(self.enabled_event_types)
        if any(event_type not in SUPPORTED_EVENT_TYPES for event_type in enabled):
            raise ValueError("enabled event types contain an unsupported value")
        object.__setattr__(self, "enabled_event_types", enabled)


class TrafficEventManager:
    """Create, store, and query declarative traffic events."""

    def __init__(
        self,
        network: nx.Graph,
        config: TrafficEventConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.network = network
        self.config = config or TrafficEventConfig()
        self.seed = self.config.random_seed if seed is None else seed
        self._events: list[TrafficEvent] = []

    @property
    def events(self) -> tuple[TrafficEvent, ...]:
        """Return scheduled events in deterministic scheduling order."""

        return tuple(self._events)

    def schedule(self, event: TrafficEvent) -> TrafficEvent:
        """Validate and store an existing domain event."""

        self._validate_event(event)
        if any(existing.id == event.id for existing in self._events):
            raise ValueError(f"event ID is already scheduled: {event.id}")
        self._events.append(event)
        self._events.sort(key=lambda item: (item.start_time_seconds, item.id))
        return event

    def create_congestion(
        self,
        event_id: str,
        start_time_seconds: int,
        duration_seconds: int,
        affected_road_ids: Iterable[str],
        *,
        severity: str = "moderate",
    ) -> TrafficEvent:
        return self._create_road_event(
            event_id,
            TrafficEventType.CONGESTION,
            start_time_seconds,
            duration_seconds,
            affected_road_ids,
            {"severity": severity},
        )

    def create_accident(
        self,
        event_id: str,
        start_time_seconds: int,
        duration_seconds: int,
        affected_road_ids: Iterable[str],
        *,
        severity: str = "moderate",
    ) -> TrafficEvent:
        return self._create_road_event(
            event_id,
            TrafficEventType.ACCIDENT,
            start_time_seconds,
            duration_seconds,
            affected_road_ids,
            {"severity": severity},
        )

    def create_road_closure(
        self,
        event_id: str,
        start_time_seconds: int,
        duration_seconds: int,
        affected_road_ids: Iterable[str],
    ) -> TrafficEvent:
        return self._create_road_event(
            event_id,
            TrafficEventType.ROAD_CLOSURE,
            start_time_seconds,
            duration_seconds,
            affected_road_ids,
        )

    def create_emergency_arrival(
        self,
        event_id: str,
        demand: VehicleDemand,
        *,
        duration_seconds: int = 1,
    ) -> TrafficEvent:
        """Represent an ambulance demand using SPECIAL_DEMAND metadata."""

        if demand.vehicle_type is not VehicleType.AMBULANCE:
            raise ValueError("emergency arrival demand must be an ambulance")
        event = TrafficEvent(
            id=event_id,
            event_type=TrafficEventType.SPECIAL_DEMAND,
            start_time_seconds=int(demand.spawn_time),
            duration_seconds=duration_seconds,
            metadata={
                "demand_kind": "emergency_arrival",
                "vehicle_id": demand.vehicle_id,
                "spawn_time": str(demand.spawn_time),
                "origin_intersection_id": demand.origin_intersection_id,
                "destination_intersection_id": demand.destination_intersection_id,
                "vehicle_type": demand.vehicle_type.value,
            },
        )
        return self.schedule(event)

    def generate(self) -> tuple[TrafficEvent, ...]:
        """Generate and schedule seeded road events within the horizon."""

        if not self.config.enabled_event_types or self.config.event_count == 0:
            return self.events
        road_ids = self._road_ids()
        if not road_ids:
            raise ValueError("event generation requires a topology with roads")
        random_state = random.Random(self.seed)
        generated: list[TrafficEvent] = []
        for index in range(self.config.event_count):
            event_type = random_state.choice(self.config.enabled_event_types)
            duration = random_state.randint(
                self.config.minimum_duration_seconds,
                self.config.maximum_duration_seconds,
            )
            latest_start = self.config.simulation_horizon - duration
            if latest_start < 0:
                raise ValueError("event duration exceeds simulation horizon")
            start_time = random_state.randint(0, latest_start)
            affected_road_ids = (random_state.choice(road_ids),)
            event = TrafficEvent(
                id=f"event-{index:06d}",
                event_type=event_type,
                start_time_seconds=start_time,
                duration_seconds=duration,
                affected_road_ids=affected_road_ids,
                metadata=self._generated_metadata(event_type),
            )
            generated.append(self.schedule(event))
        return tuple(generated)

    def events_at(self, time_seconds: int) -> tuple[TrafficEvent, ...]:
        """Return events active at ``time_seconds`` using [start, end)."""

        if time_seconds < 0:
            raise ValueError("query time cannot be negative")
        return tuple(
            event
            for event in self._events
            if event.start_time_seconds <= time_seconds
            < event.start_time_seconds + event.duration_seconds
        )

    def _create_road_event(
        self,
        event_id: str,
        event_type: TrafficEventType,
        start_time_seconds: int,
        duration_seconds: int,
        affected_road_ids: Iterable[str],
        metadata: Mapping[str, str] | None = None,
    ) -> TrafficEvent:
        event = TrafficEvent(
            id=event_id,
            event_type=event_type,
            start_time_seconds=start_time_seconds,
            duration_seconds=duration_seconds,
            affected_road_ids=tuple(affected_road_ids),
            metadata=metadata or {},
        )
        return self.schedule(event)

    def _validate_event(self, event: TrafficEvent) -> None:
        if event.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError("event type is not supported")
        if event.start_time_seconds + event.duration_seconds > self.config.simulation_horizon:
            raise ValueError("event ends after the simulation horizon")
        road_ids = set(self._road_ids())
        if any(road_id not in road_ids for road_id in event.affected_road_ids):
            raise ValueError("event references a road not present in the network")
        if len(event.affected_road_ids) != len(set(event.affected_road_ids)):
            raise ValueError("event contains duplicate affected road IDs")

    def _road_ids(self) -> tuple[str, ...]:
        road_ids = {
            attributes["road_id"]
            for _, _, attributes in self.network.edges(data=True)
            if "road_id" in attributes
        }
        return tuple(sorted(road_ids))

    @staticmethod
    def _generated_metadata(event_type: TrafficEventType) -> Mapping[str, str]:
        if event_type is TrafficEventType.CONGESTION:
            return {"severity": "moderate"}
        if event_type is TrafficEventType.ACCIDENT:
            return {"severity": "moderate"}
        return {}