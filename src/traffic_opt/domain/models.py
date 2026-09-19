"""Immutable, dependency-free domain entities for traffic simulation."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_road_dimensions,
    validate_signal_timing,
)
from .enums import TrafficEventType, VehicleType


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RoadSegment:
    """A directed road connecting two intersections."""

    id: str
    start_intersection_id: str
    end_intersection_id: str
    capacity: int
    length_meters: float
    free_flow_speed_kmh: float = 40.0

    def __post_init__(self) -> None:
        _require_identifier(self.id, "road id")
        _require_identifier(self.start_intersection_id, "start intersection id")
        _require_identifier(self.end_intersection_id, "end intersection id")
        if self.start_intersection_id == self.end_intersection_id:
            raise DomainValidationError("road endpoints must be different intersections")
        validate_road_dimensions(self.capacity, self.length_meters)
        if self.free_flow_speed_kmh <= 0:
            raise DomainValidationError("free-flow speed must be greater than zero")


@dataclass(frozen=True, slots=True)
class SignalPhaseConfig:
    """Timing and permitted movements for one signal phase."""

    name: str
    movements: frozenset[str]
    min_green_seconds: int
    max_green_seconds: int
    yellow_seconds: int = 3
    all_red_seconds: int = 1

    def __post_init__(self) -> None:
        _require_identifier(self.name, "signal phase name")
        normalized_movements = frozenset(self.movements)
        object.__setattr__(self, "movements", normalized_movements)
        validate_phase_movements(normalized_movements)
        validate_signal_timing(
            self.min_green_seconds,
            self.max_green_seconds,
            self.yellow_seconds,
            self.all_red_seconds,
        )


@dataclass(frozen=True, slots=True)
class Intersection:
    """An intersection and its incoming, outgoing, and signal configuration."""

    id: str
    incoming_road_ids: tuple[str, ...]
    outgoing_road_ids: tuple[str, ...]
    signal_phases: tuple[SignalPhaseConfig, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.id, "intersection id")
        incoming = tuple(self.incoming_road_ids)
        outgoing = tuple(self.outgoing_road_ids)
        phases = tuple(self.signal_phases)
        object.__setattr__(self, "incoming_road_ids", incoming)
        object.__setattr__(self, "outgoing_road_ids", outgoing)
        object.__setattr__(self, "signal_phases", phases)
        if not incoming:
            raise DomainValidationError("intersection must have incoming roads")
        if not outgoing:
            raise DomainValidationError("intersection must have outgoing roads")
        if not phases:
            raise DomainValidationError("intersection must have signal phases")
        if len({phase.name for phase in phases}) != len(phases):
            raise DomainValidationError("intersection signal phase names must be unique")


@dataclass(frozen=True, slots=True)
class Vehicle:
    """A vehicle with a deterministic route through the road network."""

    id: str
    vehicle_type: VehicleType
    origin_intersection_id: str
    destination_intersection_id: str
    route_road_ids: tuple[str, ...]
    current_road_id: str | None = None
    position_meters: float = 0.0
    waiting_time_seconds: float = 0.0
    completed: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.id, "vehicle id")
        _require_identifier(self.origin_intersection_id, "vehicle origin")
        _require_identifier(self.destination_intersection_id, "vehicle destination")
        route = tuple(self.route_road_ids)
        object.__setattr__(self, "route_road_ids", route)
        if not route or any(not road_id for road_id in route):
            raise DomainValidationError("vehicle route must contain road ids")
        if self.position_meters < 0:
            raise DomainValidationError("vehicle position cannot be negative")
        if self.waiting_time_seconds < 0:
            raise DomainValidationError("vehicle waiting time cannot be negative")
        if self.current_road_id is not None and self.current_road_id not in route:
            raise DomainValidationError("current road must be included in vehicle route")


@dataclass(frozen=True, slots=True)
class SignalState:
    """Current phase state for one intersection."""

    intersection_id: str
    current_phase_index: int = 0
    phase_elapsed_seconds: int = 0
    emergency_preemption: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.intersection_id, "signal state intersection id")
        if self.current_phase_index < 0:
            raise DomainValidationError("current phase index cannot be negative")
        if self.phase_elapsed_seconds < 0:
            raise DomainValidationError("phase elapsed time cannot be negative")


@dataclass(frozen=True, slots=True)
class TrafficEvent:
    """A scheduled event that changes simulated traffic conditions."""

    id: str
    event_type: TrafficEventType
    start_time_seconds: int
    duration_seconds: int
    affected_road_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.id, "event id")
        if self.start_time_seconds < 0:
            raise DomainValidationError("event start time cannot be negative")
        if self.duration_seconds <= 0:
            raise DomainValidationError("event duration must be greater than zero")
        affected_roads = tuple(self.affected_road_ids)
        object.__setattr__(self, "affected_road_ids", affected_roads)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SimulationState:
    """Complete immutable snapshot passed between future simulation components."""

    time_seconds: int = 0
    vehicles: tuple[Vehicle, ...] = ()
    signal_states: tuple[SignalState, ...] = ()
    active_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.time_seconds < 0:
            raise DomainValidationError("simulation time cannot be negative")
        object.__setattr__(self, "vehicles", tuple(self.vehicles))
        object.__setattr__(self, "signal_states", tuple(self.signal_states))
        object.__setattr__(self, "active_event_ids", tuple(self.active_event_ids))