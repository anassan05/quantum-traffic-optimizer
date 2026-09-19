from traffic_opt.domain.enums import VehicleType
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.vehicles import VehicleState


def make_simulator() -> TrafficSimulator:
    graph = create_default_topology()
    return TrafficSimulator(graph, create_default_intersections(graph), time_step_seconds=1)


def make_vehicle(vehicle_id: str = "V0001", vehicle_type: VehicleType = VehicleType.CAR) -> VehicleState:
    return VehicleState(
        id=vehicle_id,
        vehicle_type=vehicle_type,
        origin_intersection_id="I1",
        destination_intersection_id="I2",
        route_road_ids=("R_I1_I2",),
        current_road_id="R_I1_I2",
        speed_kmh=36,
    )


def test_vehicle_waits_at_red_and_queue_updates() -> None:
    simulator = make_simulator()
    simulator.add_vehicles([make_vehicle()])
    simulator.set_signal_phase("I2", 0, "green")

    simulator.run(7)

    vehicle = simulator.state.vehicles["V0001"]
    assert vehicle.position_meters == 70
    assert vehicle.waiting_time_seconds == 0
    assert simulator.state.queue_lengths["R_I1_I2"] == 0


def test_vehicle_moves_during_permitted_green_signal() -> None:
    simulator = make_simulator()
    simulator.add_vehicles([make_vehicle()])
    simulator.set_signal_phase("I2", 1, "green")

    simulator.run(8)

    assert simulator.state.vehicles["V0001"].position_meters == 80


def test_waiting_time_updates_when_red_blocks_exit() -> None:
    simulator = make_simulator()
    simulator.add_vehicles([make_vehicle()])
    simulator.set_signal_phase("I2", 0, "green")
    simulator.state.vehicles["V0001"].position_meters = 249

    simulator.step()

    assert simulator.state.vehicles["V0001"].waiting_time_seconds == 1
    assert simulator.state.queue_lengths["R_I1_I2"] == 1


def test_emergency_vehicle_obeys_red_signal() -> None:
    simulator = make_simulator()
    simulator.add_vehicles([make_vehicle(vehicle_type=VehicleType.AMBULANCE)])
    simulator.set_signal_phase("I2", 0, "green")
    simulator.state.vehicles["V0001"].position_meters = 249

    simulator.step()

    vehicle = simulator.state.vehicles["V0001"]
    assert not vehicle.completed
    assert vehicle.waiting_time_seconds == 1