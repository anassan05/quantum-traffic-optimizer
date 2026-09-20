from traffic_opt.controllers.emergency_corridor import (
    EmergencyCorridorManager,
    EmergencyCorridorStatus,
)
from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.vehicles import VehicleState


def make_simulator(**kwargs: object) -> TrafficSimulator:
    graph = create_default_topology()
    return TrafficSimulator(graph, create_default_intersections(graph), **kwargs)


def make_vehicle(vehicle_id: str = "AMB-1") -> VehicleState:
    return VehicleState(
        vehicle_id,
        VehicleType.AMBULANCE,
        "I1",
        "I3",
        ("R_I1_I2", "R_I2_I3"),
        "R_I1_I2",
        position_meters=249.0,
        speed_kmh=36.0,
    )


def test_corridor_identifies_required_movement_and_intersection() -> None:
    graph = create_default_topology()
    manager = EmergencyCorridorManager(graph, create_default_intersections(graph))
    vehicle = make_vehicle()

    assert manager.next_required_movement(vehicle) == (
        "I2",
        "R_I1_I2",
        "R_I2_I3",
    )
    assert manager.affected_intersections(vehicle) == ("I2",)


def test_corridor_activates_and_requests_only_required_phase() -> None:
    simulator = make_simulator()
    vehicle = make_vehicle()
    simulator.add_vehicles((vehicle,))

    result = simulator.corridor_manager.update(
        simulator.state.vehicles.values(),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    assert result.status is EmergencyCorridorStatus.WAITING_FOR_CLEARANCE
    assert result.intersection_id == "I2"
    assert result.phase_index == 1
    assert simulator.corridor_manager.requested_phase("I2") == 1


def test_corridor_uses_yellow_and_all_red_before_conflicting_green() -> None:
    simulator = make_simulator()
    vehicle = make_vehicle()
    simulator.add_vehicles((vehicle,))

    simulator.step()
    assert simulator.state.intersection_states["I2"].mode == "green"
    assert vehicle.current_road_id == "R_I1_I2"

    simulator.run(10)
    assert simulator.state.intersection_states["I2"].mode == "yellow"
    assert vehicle.current_road_id == "R_I1_I2"

    simulator.run(2)
    assert simulator.state.intersection_states["I2"].mode == "all_red"
    assert vehicle.current_road_id == "R_I1_I2"

    simulator.step()
    assert simulator.state.intersection_states["I2"].current_phase_index == 1
    assert vehicle.current_road_id == "R_I2_I3"


def test_corridor_blocks_closed_required_road() -> None:
    simulator = make_simulator(
        events=(
            TrafficEvent(
                "closure",
                TrafficEventType.ROAD_CLOSURE,
                0,
                3,
                ("R_I2_I3",),
            ),
        )
    )
    vehicle = make_vehicle()
    simulator.add_vehicles((vehicle,))

    result = simulator.corridor_manager.update(
        simulator.state.vehicles.values(),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    assert result.status is EmergencyCorridorStatus.REJECTED
    assert "closed" in result.reason
    assert simulator.corridor_manager.requested_phase("I2") is None


def test_corridor_releases_after_vehicle_passes_intersection() -> None:
    simulator = make_simulator()
    vehicle = make_vehicle()
    simulator.add_vehicles((vehicle,))
    simulator.corridor_manager.update(
        simulator.state.vehicles.values(),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    vehicle.route_index = 1
    vehicle.current_road_id = "R_I2_I3"
    vehicle.position_meters = 0.0
    result = simulator.corridor_manager.update(
        simulator.state.vehicles.values(),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    assert result.status is EmergencyCorridorStatus.RELEASED
    assert simulator.corridor_manager.active_vehicle_id is None
    assert simulator.corridor_manager.requested_phase("I2") is None


def test_no_emergency_vehicle_keeps_normal_signal_behavior() -> None:
    simulator = make_simulator()

    result = simulator.corridor_manager.update(
        (),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    assert result.status is EmergencyCorridorStatus.INACTIVE
    assert all(
        simulator.corridor_manager.requested_phase(intersection_id) is None
        for intersection_id in simulator.state.intersection_states
    )


def test_multiple_emergency_vehicles_choose_lowest_id_deterministically() -> None:
    simulator = make_simulator()
    first = make_vehicle("AMB-2")
    second = make_vehicle("AMB-1")
    simulator.add_vehicles((first, second))

    result = simulator.corridor_manager.update(
        simulator.state.vehicles.values(),
        simulator.state.intersection_states,
        simulator.state.time_seconds,
        simulator.event_manager.is_road_closed,
    )

    assert result.vehicle_id == "AMB-1"
    assert simulator.corridor_manager.active_vehicle_id == "AMB-1"
