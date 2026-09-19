import pytest

from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.metrics import MetricsCollector
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.events import EventImpactConfig, effective_road_conditions
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.vehicles import VehicleState


def simulator(*events: TrafficEvent) -> TrafficSimulator:
    topology = create_default_topology()
    return TrafficSimulator(
        topology,
        create_default_intersections(topology),
        time_step_seconds=1,
        events=events,
    )


def car(position: float = 0.0, vehicle_id: str = "v1") -> VehicleState:
    return VehicleState(
        vehicle_id,
        VehicleType.CAR,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
        position_meters=position,
    )


def test_road_closure_blocks_insertion_then_allows_it_after_end() -> None:
    closure = TrafficEvent("close", TrafficEventType.ROAD_CLOSURE, 0, 1, ("R_I1_I2",))
    traffic_simulator = simulator(closure)
    traffic_simulator.schedule_vehicle(car(), 0)

    traffic_simulator.step()
    assert traffic_simulator.state.vehicles == {}
    traffic_simulator.step()
    assert set(traffic_simulator.state.vehicles) == {"v1"}


def test_congestion_changes_speed_and_ends_at_event_boundary() -> None:
    congestion = TrafficEvent(
        "slow", TrafficEventType.CONGESTION, 0, 1, ("R_I1_I2",), {"speed_multiplier": "0.25"}
    )
    traffic_simulator = simulator(congestion)
    traffic_simulator.add_vehicles((car(),))

    traffic_simulator.step()
    congested_position = traffic_simulator.state.vehicles["v1"].position_meters
    traffic_simulator.step()
    normal_position = traffic_simulator.state.vehicles["v1"].position_meters

    assert congested_position == pytest.approx(40 / 3.6 * 0.25)
    assert normal_position - congested_position == pytest.approx(40 / 3.6)


def test_accident_effect_is_temporary_and_metadata_is_supported() -> None:
    accident = TrafficEvent(
        "accident",
        TrafficEventType.ACCIDENT,
        0,
        1,
        ("R_I1_I2",),
        {"speed_multiplier": "0.2", "capacity_multiplier": "0.25"},
    )
    traffic_simulator = simulator(accident)

    during = traffic_simulator.road_conditions()["R_I1_I2"]
    traffic_simulator.step()
    after = traffic_simulator.road_conditions()["R_I1_I2"]

    assert during.speed_multiplier == 0.2
    assert during.capacity_multiplier == 0.25
    assert after.speed_multiplier == 1.0
    assert after.capacity_multiplier == 1.0


def test_closure_takes_precedence_over_other_effects() -> None:
    conditions = effective_road_conditions(
        ("R_I1_I2",),
        (
            TrafficEvent("c", TrafficEventType.CONGESTION, 0, 10, ("R_I1_I2",)),
            TrafficEvent("a", TrafficEventType.ACCIDENT, 0, 10, ("R_I1_I2",)),
            TrafficEvent("x", TrafficEventType.ROAD_CLOSURE, 0, 10, ("R_I1_I2",)),
        ),
        impact=EventImpactConfig(),
    )

    assert conditions["R_I1_I2"].available is False
    assert conditions["R_I1_I2"].speed_multiplier == pytest.approx(0.25)


def test_emergency_arrival_inserts_ambulance_at_event_time_and_metrics_see_it() -> None:
    arrival = TrafficEvent(
        "arrival",
        TrafficEventType.SPECIAL_DEMAND,
        2,
        1,
        metadata={"demand_kind": "emergency_arrival", "vehicle_type": "ambulance"},
    )
    traffic_simulator = simulator(arrival)
    ambulance = VehicleState(
        "ambulance-1",
        VehicleType.AMBULANCE,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
        position_meters=249,
        speed_kmh=50,
    )
    traffic_simulator.schedule_vehicle(ambulance, 2, arrival_event_id="arrival")
    metrics = MetricsCollector(create_default_topology())

    metrics.collect(traffic_simulator.state)
    traffic_simulator.step()
    traffic_simulator.step()
    assert "ambulance-1" not in traffic_simulator.state.vehicles
    traffic_simulator.step()

    assert "ambulance-1" in traffic_simulator.state.vehicles
    assert metrics.collect(traffic_simulator.state).ambulance_count == 1