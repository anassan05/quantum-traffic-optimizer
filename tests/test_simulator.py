from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.vehicles import VehicleState, generate_vehicles
import pytest


def build_simulator(seed: int) -> TrafficSimulator:
    graph = create_default_topology()
    simulator = TrafficSimulator(
        graph,
        create_default_intersections(graph),
        time_step_seconds=1,
        seed=seed,
    )
    simulator.add_vehicles(generate_vehicles(graph, 8, seed=seed))
    return simulator


def test_road_capacity_limits_entry() -> None:
    graph = create_default_topology()
    graph["I1"]["I2"]["capacity"] = 1
    graph["I1"]["I2"]["road_segment"] = graph["I1"]["I2"]["road_segment"]
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    vehicles = generate_vehicles(graph, 2, seed=1)
    for vehicle in vehicles:
        vehicle.current_road_id = "R_I1_I2"
        vehicle.route_road_ids = ("R_I1_I2",)
        vehicle.position_meters = 249
    with pytest.raises(ValueError, match="road capacity exceeded"):
        simulator.add_vehicles(vehicles)


def test_same_seed_produces_same_simulation_state() -> None:
    first = build_simulator(42)
    second = build_simulator(42)

    first.run(5)
    second.run(5)

    assert first.state.time_seconds == second.state.time_seconds
    assert [
        (vehicle.id, vehicle.current_road_id, vehicle.position_meters)
        for vehicle in first.state.vehicles.values()
    ] == [
        (vehicle.id, vehicle.current_road_id, vehicle.position_meters)
        for vehicle in second.state.vehicles.values()
    ]


def test_closure_blocks_movement_into_downstream_road() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(
        graph,
        create_default_intersections(graph),
        events=(
            TrafficEvent(
                "closure",
                TrafficEventType.ROAD_CLOSURE,
                0,
                2,
                ("R_I2_I3",),
            ),
        ),
    )
    simulator.set_signal_phase("I2", 1)
    vehicle = VehicleState(
        "V1",
        VehicleType.CAR,
        "I1",
        "I3",
        ("R_I1_I2", "R_I2_I3"),
        "R_I1_I2",
        position_meters=249.0,
    )
    simulator.add_vehicles((vehicle,))

    simulator.step()

    assert vehicle.current_road_id == "R_I1_I2"
    assert vehicle.position_meters == 250.0
    assert vehicle.waiting_time_seconds == 1.0


def test_closure_activation_and_expiration_updates_active_event_ids() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(
        graph,
        create_default_intersections(graph),
        events=(
            TrafficEvent(
                "closure",
                TrafficEventType.ROAD_CLOSURE,
                1,
                1,
                ("R_I2_I3",),
            ),
        ),
    )

    assert simulator.state.active_event_ids == ()
    simulator.step()
    assert simulator.state.active_event_ids == ("closure",)
    simulator.step()
    assert simulator.state.active_event_ids == ()


def test_no_events_preserve_existing_simulator_behavior() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    vehicle = VehicleState(
        "V1",
        VehicleType.CAR,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
    )
    simulator.add_vehicles((vehicle,))

    simulator.step()

    assert simulator.state.active_event_ids == ()
    assert vehicle.position_meters == pytest.approx(40.0 / 3.6)