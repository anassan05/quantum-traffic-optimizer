import pytest

from traffic_opt.domain.enums import VehicleType
from traffic_opt.metrics import FuelModel, MetricsCollector
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.state import SimulationState
from traffic_opt.simulation.vehicles import VehicleState


def vehicle(
    vehicle_id: str,
    *,
    vehicle_type: VehicleType = VehicleType.CAR,
    waiting: float = 0.0,
    position: float = 0.0,
    completed: bool = False,
) -> VehicleState:
    return VehicleState(
        vehicle_id,
        vehicle_type,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
        position_meters=position,
        waiting_time_seconds=waiting,
        completed=completed,
    )


def state(time: float, *vehicles: VehicleState, queue: dict[str, int] | None = None) -> SimulationState:
    return SimulationState(
        time_seconds=time,
        vehicles={item.id: item for item in vehicles},
        queue_lengths=queue or {},
    )


def test_waiting_and_queue_metrics() -> None:
    collector = MetricsCollector(create_default_topology())

    snapshot = collector.collect(
        state(10, vehicle("v1", waiting=4), vehicle("v2", waiting=6), queue={"R_I1_I2": 3, "R_I2_I3": 1})
    )

    assert snapshot.total_waiting_time_seconds == 10
    assert snapshot.average_waiting_time_seconds == 5
    assert snapshot.total_queue_length == 4
    assert snapshot.average_queue_length == 2
    assert snapshot.maximum_queue_length == 3


def test_throughput_and_travel_time_use_completion_observations() -> None:
    collector = MetricsCollector(create_default_topology(), spawn_times={"v1": 2})
    collector.collect(state(4, vehicle("v1", position=100)))
    snapshot = collector.collect(state(12, vehicle("v1", position=250, completed=True)))

    assert snapshot.completed_vehicles == 1
    assert snapshot.throughput_vehicles_per_second == pytest.approx(1 / 12)
    assert snapshot.total_travel_time_seconds == 10
    assert snapshot.average_travel_time_seconds == 10


def test_fuel_and_co2_estimates_are_configurable() -> None:
    fuel_model = FuelModel(
        fuel_rate_l_per_km={vehicle_type: 1.0 for vehicle_type in VehicleType},
        idle_fuel_rate_l_per_hour={vehicle_type: 0.0 for vehicle_type in VehicleType},
        co2_grams_per_litre=100.0,
    )
    snapshot = MetricsCollector(
        create_default_topology(), fuel_model=fuel_model
    ).collect(state(1, vehicle("v1", position=250)))

    assert snapshot.fuel_consumed_litres == pytest.approx(0.25)
    assert snapshot.co2_emissions_grams == pytest.approx(25.0)


def test_ambulance_metrics_and_snapshots_are_supported() -> None:
    collector = MetricsCollector(create_default_topology(), spawn_times={"a1": 1})
    first = collector.collect(state(1, vehicle("a1", vehicle_type=VehicleType.AMBULANCE)))
    second = collector.collect(
        state(11, vehicle("a1", vehicle_type=VehicleType.AMBULANCE, position=250, completed=True))
    )
    aggregate = collector.aggregate()

    assert first.ambulance_count == 1
    assert second.completed_ambulances == 1
    assert second.average_ambulance_travel_time_seconds == 10
    assert len(collector.snapshots) == 2
    assert aggregate.observation_duration_seconds == 10
    assert aggregate.completed_vehicles == 1


def test_empty_and_zero_duration_observations_are_safe() -> None:
    collector = MetricsCollector(create_default_topology())
    snapshot = collector.collect(state(0))

    assert snapshot.total_vehicles == 0
    assert snapshot.throughput_vehicles_per_second == 0
    assert collector.aggregate().throughput_vehicles_per_second == 0


def test_metrics_are_deterministic_and_invalid_routes_fail_clearly() -> None:
    first = MetricsCollector(create_default_topology()).collect(
        state(5, vehicle("v1", waiting=2, position=50))
    )
    second = MetricsCollector(create_default_topology()).collect(
        state(5, vehicle("v1", waiting=2, position=50))
    )
    assert first == second

    invalid = vehicle("bad")
    invalid.route_index = 1
    with pytest.raises(ValueError, match="invalid route index"):
        MetricsCollector(create_default_topology()).collect(state(1, invalid))


def test_invalid_fuel_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        FuelModel(co2_grams_per_litre=-1)