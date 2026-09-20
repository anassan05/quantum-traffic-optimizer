import pytest

from traffic_opt.domain.enums import VehicleType
from traffic_opt.domain.models import Vehicle
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.vehicles import VehicleState, create_emergency_vehicle
from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent


def make_simulator(**kwargs: object) -> TrafficSimulator:
    graph = create_default_topology()
    return TrafficSimulator(
        graph,
        create_default_intersections(graph),
        **kwargs,
    )


def test_create_emergency_vehicle_validates_and_identifies_ambulance() -> None:
    graph = create_default_topology()

    vehicle = create_emergency_vehicle(
        graph,
        "AMB-1",
        "I1",
        "I3",
        ("R_I1_I2", "R_I2_I3"),
    )

    assert vehicle.vehicle_type is VehicleType.AMBULANCE
    assert vehicle.is_emergency is True
    assert vehicle.is_emergency_vehicle is True
    assert vehicle.current_road_id == "R_I1_I2"


def test_emergency_vehicle_route_validation_rejects_invalid_routes() -> None:
    graph = create_default_topology()

    with pytest.raises(ValueError, match="unknown route road"):
        create_emergency_vehicle(
            graph, "AMB-1", "I1", "I2", ("missing-road",)
        )
    with pytest.raises(ValueError, match="disconnected"):
        create_emergency_vehicle(
            graph, "AMB-1", "I1", "I3", ("R_I1_I2", "R_I3_I4")
        )
    with pytest.raises(ValueError, match="does not start"):
        create_emergency_vehicle(
            graph, "AMB-1", "I2", "I3", ("R_I1_I2", "R_I2_I3")
        )


def test_simulator_tracks_emergency_vehicle_ids() -> None:
    simulator = make_simulator()

    vehicle = simulator.add_emergency_vehicle(
        "AMB-1", "I1", "I2", ("R_I1_I2",)
    )

    assert vehicle.id in simulator.state.vehicles
    assert simulator.state.emergency_vehicle_ids == ("AMB-1",)


def test_emergency_vehicle_respects_red_yellow_and_all_red_signals() -> None:
    for mode in ("green", "yellow", "all_red"):
        simulator = make_simulator(time_step_seconds=0.1)
        vehicle = simulator.add_emergency_vehicle(
            "AMB-1", "I1", "I2", ("R_I1_I2",)
        )
        simulator.set_signal_phase("I2", 0, mode)
        vehicle.position_meters = 249.95

        simulator.step()

        assert vehicle.current_road_id == "R_I1_I2"
        assert vehicle.completed is False
        assert vehicle.waiting_time_seconds == 0.1


def test_emergency_vehicle_respects_road_closure() -> None:
    simulator = make_simulator(
        events=(
            TrafficEvent(
                "closure",
                TrafficEventType.ROAD_CLOSURE,
                0,
                2,
                ("R_I2_I3",),
            ),
        )
    )
    vehicle = simulator.add_emergency_vehicle(
        "AMB-1", "I1", "I3", ("R_I1_I2", "R_I2_I3")
    )
    simulator.set_signal_phase("I2", 1, "green")
    vehicle.position_meters = 249.0

    simulator.step()

    assert vehicle.current_road_id == "R_I1_I2"
    assert vehicle.waiting_time_seconds == 1.0


def test_regular_vehicle_remains_non_emergency() -> None:
    simulator = make_simulator()
    regular = VehicleState(
        "CAR-1",
        VehicleType.CAR,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
    )

    simulator.add_vehicles((regular,))

    assert regular.is_emergency is False
    assert simulator.state.emergency_vehicle_ids == ()


def test_ambulance_domain_vehicle_is_identified_as_emergency() -> None:
    vehicle = Vehicle(
        "AMB-1",
        VehicleType.AMBULANCE,
        "I1",
        "I2",
        ("R_I1_I2",),
    )

    assert vehicle.is_emergency is True
    assert vehicle.is_emergency_vehicle is True
