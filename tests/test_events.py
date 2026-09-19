import networkx as nx
import pytest

from traffic_opt.demand import VehicleDemand
from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.events import TrafficEventConfig, TrafficEventManager
from traffic_opt.network.topology import create_default_topology


def test_seeded_event_generation_is_deterministic() -> None:
    network = create_default_topology()
    config = TrafficEventConfig(random_seed=42, simulation_horizon=1000, event_count=8)

    first = TrafficEventManager(network, config).generate()
    second = TrafficEventManager(network, config).generate()

    assert first == second


def test_different_seeds_normally_generate_different_events() -> None:
    network = create_default_topology()
    config = TrafficEventConfig(simulation_horizon=1000, event_count=8)

    first = TrafficEventManager(network, config, seed=1).generate()
    second = TrafficEventManager(network, config, seed=2).generate()

    assert first != second


def test_generated_events_have_supported_types_roads_and_timing() -> None:
    network = create_default_topology()
    config = TrafficEventConfig(
        simulation_horizon=1000,
        event_count=20,
        minimum_duration_seconds=10,
        maximum_duration_seconds=50,
    )
    events = TrafficEventManager(network, config).generate()
    road_ids = {
        attributes["road_id"]
        for _, _, attributes in network.edges(data=True)
    }

    assert events
    assert all(event.event_type in config.enabled_event_types for event in events)
    assert all(road_id in road_ids for event in events for road_id in event.affected_road_ids)
    assert all(event.start_time_seconds >= 0 for event in events)
    assert all(event.duration_seconds > 0 for event in events)
    assert all(
        event.start_time_seconds + event.duration_seconds <= config.simulation_horizon
        for event in events
    )


def test_active_event_interval_is_start_inclusive_and_end_exclusive() -> None:
    manager = TrafficEventManager(
        create_default_topology(),
        TrafficEventConfig(simulation_horizon=100),
    )
    event = manager.create_congestion("c1", 20, 10, ("R_I1_I2",))

    assert manager.events_at(19) == ()
    assert manager.events_at(20) == (event,)
    assert manager.events_at(25) == (event,)
    assert manager.events_at(30) == ()
    assert manager.events_at(31) == ()


@pytest.mark.parametrize(
    ("factory_name", "event_type"),
    [
        ("create_congestion", TrafficEventType.CONGESTION),
        ("create_accident", TrafficEventType.ACCIDENT),
        ("create_road_closure", TrafficEventType.ROAD_CLOSURE),
    ],
)
def test_event_factories_create_valid_road_events(factory_name: str, event_type: TrafficEventType) -> None:
    manager = TrafficEventManager(
        create_default_topology(), TrafficEventConfig(simulation_horizon=100)
    )
    factory = getattr(manager, factory_name)

    event = factory("e1", 10, 20, ("R_I1_I2",))

    assert event.event_type is event_type
    assert event.affected_road_ids == ("R_I1_I2",)


def test_emergency_arrival_reuses_special_demand_and_ambulance() -> None:
    manager = TrafficEventManager(
        create_default_topology(), TrafficEventConfig(simulation_horizon=100)
    )
    demand = VehicleDemand("ambulance-1", 12.5, "I1", "I4", VehicleType.AMBULANCE)

    event = manager.create_emergency_arrival("arrival-1", demand)

    assert event.event_type is TrafficEventType.SPECIAL_DEMAND
    assert dict(event.metadata) == {
        "demand_kind": "emergency_arrival",
        "vehicle_id": "ambulance-1",
        "spawn_time": "12.5",
        "origin_intersection_id": "I1",
        "destination_intersection_id": "I4",
        "vehicle_type": "ambulance",
    }


def test_emergency_arrival_rejects_non_ambulance_demand() -> None:
    manager = TrafficEventManager(
        create_default_topology(), TrafficEventConfig(simulation_horizon=100)
    )
    demand = VehicleDemand("car-1", 12, "I1", "I4", VehicleType.CAR)

    with pytest.raises(ValueError, match="ambulance"):
        manager.create_emergency_arrival("arrival-1", demand)


def test_empty_event_types_generate_no_events() -> None:
    manager = TrafficEventManager(
        create_default_topology(),
        TrafficEventConfig(event_count=10, enabled_event_types=()),
    )

    assert manager.generate() == ()


def test_event_generation_rejects_topology_without_roads() -> None:
    manager = TrafficEventManager(
        nx.DiGraph(), TrafficEventConfig(event_count=1)
    )

    with pytest.raises(ValueError, match="topology with roads"):
        manager.generate()


def test_event_validation_rejects_invalid_roads_and_horizon() -> None:
    manager = TrafficEventManager(
        create_default_topology(), TrafficEventConfig(simulation_horizon=100)
    )

    with pytest.raises(ValueError, match="road not present"):
        manager.create_accident("bad-road", 10, 10, ("missing",))
    with pytest.raises(ValueError, match="after the simulation horizon"):
        manager.create_accident("late", 95, 10, ("R_I1_I2",))