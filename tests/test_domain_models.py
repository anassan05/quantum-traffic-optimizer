import pytest

from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import (
    Intersection,
    RoadSegment,
    SignalPhaseConfig,
    SignalState,
    SimulationState,
    TrafficEvent,
    Vehicle,
)


def phase(name: str = "north-south") -> SignalPhaseConfig:
    return SignalPhaseConfig(name, frozenset({"north_south"}), 10, 60)


def test_valid_road_and_intersection_creation() -> None:
    road = RoadSegment("r1", "i1", "i2", capacity=20, length_meters=150)
    intersection = Intersection("i1", (road.id,), ("r2",), (phase(),))

    assert road.capacity == 20
    assert intersection.signal_phases[0].min_green_seconds == 10


@pytest.mark.parametrize(
    ("capacity", "length"),
    [(0, 100), (-1, 100), (2, 0), (2, -1)],
)
def test_invalid_road_dimensions(capacity: int, length: float) -> None:
    with pytest.raises(ValueError, match="capacity|length"):
        RoadSegment("r1", "i1", "i2", capacity, length)


def test_invalid_signal_duration_values() -> None:
    with pytest.raises(ValueError, match="minimum green"):
        SignalPhaseConfig("bad", frozenset({"north_south"}), 60, 10)
    with pytest.raises(ValueError, match="yellow"):
        SignalPhaseConfig("bad", frozenset({"north_south"}), 10, 60, yellow_seconds=0)


def test_valid_and_invalid_vehicle_creation() -> None:
    vehicle = Vehicle("v1", VehicleType.CAR, "i1", "i2", ("r1",), "r1")
    assert vehicle.route_road_ids == ("r1",)
    with pytest.raises(ValueError, match="route"):
        Vehicle("v2", VehicleType.CAR, "i1", "i2", ())
    with pytest.raises(ValueError, match="current road"):
        Vehicle("v3", VehicleType.CAR, "i1", "i2", ("r1",), "r2")


def test_other_models_are_valid_and_deterministic() -> None:
    event = TrafficEvent("closure", TrafficEventType.ROAD_CLOSURE, 20, 30, ("r1",))
    first = SimulationState(
        vehicles=(Vehicle("v1", VehicleType.AMBULANCE, "i1", "i2", ("r1",)),),
        signal_states=(SignalState("i1"),),
        active_event_ids=(event.id,),
    )
    second = SimulationState(
        vehicles=(Vehicle("v1", VehicleType.AMBULANCE, "i1", "i2", ("r1",)),),
        signal_states=(SignalState("i1"),),
        active_event_ids=(event.id,),
    )

    assert first == second
    assert dict(event.metadata) == {}