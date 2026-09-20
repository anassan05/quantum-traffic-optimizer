import pytest

from traffic_opt.domain.enums import TrafficEventType, VehicleType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.metrics import (
    DEFAULT_CO2_KILOGRAMS_PER_LITER,
    DEFAULT_FUEL_LITERS_PER_KILOMETER,
    DEFAULT_IDLE_FUEL_LITERS_PER_HOUR,
    MetricsCollector,
    SimulationMetrics,
)
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.state import SimulationState
from traffic_opt.simulation.vehicles import VehicleState


def make_vehicle(
    vehicle_id: str = "V1",
    *,
    emergency: bool = False,
) -> VehicleState:
    return VehicleState(
        vehicle_id,
        VehicleType.AMBULANCE if emergency else VehicleType.CAR,
        "I1",
        "I2",
        ("R_I1_I2",),
        "R_I1_I2",
        speed_kmh=36,
    )


def test_empty_metrics_are_zero_safe() -> None:
    graph = create_default_topology()
    metrics = MetricsCollector(graph)
    state = SimulationState()
    metrics.record(state)

    result = metrics.snapshot(state)

    assert result.total_vehicles_completed == 0
    assert result.average_waiting_time_seconds == 0
    assert result.maximum_queue_length == 0
    assert result.throughput_vehicles_per_second == 0
    assert result.estimated_fuel_consumption_liters == 0
    assert result.estimated_co2_emissions_kg == 0
    assert result.emergency_vehicle_travel_time_seconds == {}


def test_metrics_mapping_defaults_are_immutable_and_independent() -> None:
    first = SimulationMetrics()
    second = SimulationMetrics()

    assert first.emergency_vehicle_travel_time_seconds == {}
    assert first.emergency_vehicle_delay_seconds == {}
    assert first.emergency_vehicle_travel_time_seconds is not second.emergency_vehicle_travel_time_seconds
    assert first.emergency_vehicle_delay_seconds is not second.emergency_vehicle_delay_seconds
    with pytest.raises(TypeError):
        first.emergency_vehicle_travel_time_seconds["AMB-1"] = 1
    with pytest.raises(TypeError):
        first.emergency_vehicle_delay_seconds["AMB-1"] = 1


def test_metrics_calculate_waiting_queue_and_estimated_environmental_values() -> None:
    graph = create_default_topology()
    vehicle = make_vehicle()
    vehicle.position_meters = 100
    vehicle.waiting_time_seconds = 2
    state = SimulationState(time_seconds=10, vehicles={vehicle.id: vehicle})
    state.queue_lengths = {"R_I1_I2": 2}
    collector = MetricsCollector(graph)
    collector.record(state)
    state.queue_lengths["R_I1_I2"] = 4
    collector.record(state)

    result = collector.snapshot(state)
    expected_fuel = (
        0.1 * DEFAULT_FUEL_LITERS_PER_KILOMETER
        + 2 / 3600 * DEFAULT_IDLE_FUEL_LITERS_PER_HOUR
    )

    assert result.total_waiting_time_seconds == 2
    assert result.average_waiting_time_seconds == 2
    assert result.maximum_waiting_time_seconds == 2
    assert result.average_queue_length == 3
    assert result.maximum_queue_length == 4
    assert result.estimated_fuel_consumption_liters == pytest.approx(expected_fuel)
    assert result.estimated_co2_emissions_kg == pytest.approx(
        expected_fuel * DEFAULT_CO2_KILOGRAMS_PER_LITER
    )


def test_simulator_metrics_track_blocked_waiting_and_completion() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    vehicle = make_vehicle()
    simulator.add_vehicles((vehicle,))
    simulator.set_signal_phase("I2", 0, "green")
    vehicle.position_meters = 249

    simulator.step()

    result = simulator.metrics
    assert result.total_vehicles_completed == 0
    assert result.total_waiting_time_seconds == 1
    assert result.maximum_waiting_time_seconds == 1
    assert result.throughput_vehicles_per_second == 0


def test_emergency_travel_time_and_delay_are_tracked() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    vehicle = simulator.add_emergency_vehicle(
        "AMB-1", "I1", "I2", ("R_I1_I2",), speed_kmh=36
    )
    simulator.set_signal_phase("I2", 1, "green")
    vehicle.position_meters = 249

    simulator.step()

    result = simulator.metrics
    assert result.emergency_vehicle_travel_time_seconds == {"AMB-1": 1}
    assert result.emergency_vehicle_delay_seconds == {"AMB-1": 0}
    assert result.average_emergency_vehicle_travel_time_seconds == 1


def test_events_are_preserved_for_metrics_enabled_simulator() -> None:
    graph = create_default_topology()
    simulator = TrafficSimulator(
        graph,
        create_default_intersections(graph),
        events=(
            TrafficEvent(
                "closure", TrafficEventType.ROAD_CLOSURE, 0, 1, ("R_I1_I2",)
            ),
        ),
    )

    assert simulator.state.active_event_ids == ("closure",)
    assert simulator.metrics.total_vehicles_completed == 0
