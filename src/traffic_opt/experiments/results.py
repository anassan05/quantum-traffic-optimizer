"""Structured experiment results and comparison rows."""

from dataclasses import dataclass

from traffic_opt.demand import VehicleDemand
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.metrics import MetricsAggregate, MetricsSnapshot


@dataclass(frozen=True, slots=True)
class ScenarioExperimentResult:
    """Immutable output from one configured controller experiment."""

    experiment_name: str
    controller_name: str
    scenario: str
    seed: int
    simulation_duration_seconds: int
    demands: tuple[VehicleDemand, ...]
    events: tuple[TrafficEvent, ...]
    metrics: MetricsAggregate
    snapshots: tuple[MetricsSnapshot, ...]
    ambulance_count: int
    completed_ambulances: int
    average_ambulance_travel_time_seconds: float
    average_ambulance_waiting_time_seconds: float
    execution_status: str = "completed"

    @property
    def number_of_generated_vehicles(self) -> int:
        return len(self.demands)

    @property
    def number_of_generated_events(self) -> int:
        return len(self.events)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One measured result row suitable for later visualization."""

    experiment_name: str
    controller_name: str
    scenario: str
    average_waiting_time_seconds: float
    average_queue_length: float
    throughput_vehicles_per_second: float
    average_travel_time_seconds: float
    total_fuel_consumed_litres: float
    total_co2_emissions_grams: float
    ambulance_count: int
    completed_ambulances: int
    average_ambulance_travel_time_seconds: float
    average_ambulance_waiting_time_seconds: float


def compare_results(
    results: tuple[ScenarioExperimentResult, ...],
) -> tuple[ComparisonRow, ...]:
    """Convert results into deterministic measured rows without ranking them."""

    return tuple(
        ComparisonRow(
            experiment_name=result.experiment_name,
            controller_name=result.controller_name,
            scenario=result.scenario,
            average_waiting_time_seconds=result.metrics.average_waiting_time_seconds,
            average_queue_length=result.metrics.average_queue_length,
            throughput_vehicles_per_second=result.metrics.throughput_vehicles_per_second,
            average_travel_time_seconds=result.metrics.average_travel_time_seconds,
            total_fuel_consumed_litres=result.metrics.total_fuel_consumed_litres,
            total_co2_emissions_grams=result.metrics.total_co2_emissions_grams,
            ambulance_count=result.ambulance_count,
            completed_ambulances=result.completed_ambulances,
            average_ambulance_travel_time_seconds=result.average_ambulance_travel_time_seconds,
            average_ambulance_waiting_time_seconds=result.average_ambulance_waiting_time_seconds,
        )
        for result in results
    )