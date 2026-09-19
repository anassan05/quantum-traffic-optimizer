"""Temporary effective road conditions derived from active traffic events."""

from dataclasses import dataclass
from math import prod
from typing import Iterable, Mapping

from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent


@dataclass(frozen=True, slots=True)
class EventImpactConfig:
    """Configurable simulation approximations for traffic-event effects."""

    congestion_speed_multiplier: float = 0.5
    accident_speed_multiplier: float = 0.5
    accident_capacity_multiplier: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.congestion_speed_multiplier,
            self.accident_speed_multiplier,
            self.accident_capacity_multiplier,
        )
        if any(value <= 0 for value in values):
            raise ValueError("event impact multipliers must be greater than zero")


@dataclass(frozen=True, slots=True)
class EffectiveRoadCondition:
    """Effective conditions for one road during one simulation step."""

    road_id: str
    available: bool = True
    speed_multiplier: float = 1.0
    capacity_multiplier: float = 1.0


def effective_road_conditions(
    road_ids: Iterable[str],
    active_events: Iterable[TrafficEvent],
    *,
    impact: EventImpactConfig | None = None,
) -> Mapping[str, EffectiveRoadCondition]:
    """Combine active events deterministically without mutating road models.

    A road closure takes precedence over all other effects. Otherwise,
    congestion and accident speed/capacity multipliers are multiplied. Event
    metadata may override accident ``speed_multiplier`` and
    ``capacity_multiplier`` with positive numeric values.
    """

    configuration = impact or EventImpactConfig()
    grouped: dict[str, list[TrafficEvent]] = {road_id: [] for road_id in road_ids}
    for event in active_events:
        for road_id in event.affected_road_ids:
            if road_id in grouped:
                grouped[road_id].append(event)

    conditions: dict[str, EffectiveRoadCondition] = {}
    for road_id, events in grouped.items():
        closed = any(event.event_type is TrafficEventType.ROAD_CLOSURE for event in events)
        speed_multipliers: list[float] = []
        capacity_multipliers: list[float] = []
        for event in events:
            if event.event_type is TrafficEventType.CONGESTION:
                speed_multipliers.append(
                    _metadata_multiplier(
                        event.metadata,
                        "speed_multiplier",
                        configuration.congestion_speed_multiplier,
                    )
                )
            elif event.event_type is TrafficEventType.ACCIDENT:
                speed_multipliers.append(
                    _metadata_multiplier(
                        event.metadata,
                        "speed_multiplier",
                        configuration.accident_speed_multiplier,
                    )
                )
                capacity_multipliers.append(
                    _metadata_multiplier(
                        event.metadata,
                        "capacity_multiplier",
                        configuration.accident_capacity_multiplier,
                    )
                )
        conditions[road_id] = EffectiveRoadCondition(
            road_id=road_id,
            available=not closed,
            speed_multiplier=prod(speed_multipliers) if speed_multipliers else 1.0,
            capacity_multiplier=(
                prod(capacity_multipliers) if capacity_multipliers else 1.0
            ),
        )
    return conditions


def _metadata_multiplier(metadata: Mapping[str, str], key: str, default: float) -> float:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        multiplier = float(value)
    except ValueError as error:
        raise ValueError(f"event metadata {key} must be numeric") from error
    if multiplier <= 0:
        raise ValueError(f"event metadata {key} must be greater than zero")
    return multiplier