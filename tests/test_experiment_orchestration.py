import networkx as nx
import pytest

from traffic_opt.demand import DemandScenarioConfig, TrafficScenario
from traffic_opt.domain.enums import ControllerType
from traffic_opt.events import TrafficEventConfig
from traffic_opt.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    compare_results,
)


def configuration(
    controller: ControllerType = ControllerType.FIXED_TIME,
    *,
    seed: int = 4,
    arrival_rate: float = 0.0,
    event_count: int = 0,
) -> ExperimentConfig:
    horizon = 4
    return ExperimentConfig(
        experiment_name=f"test-{controller.value}",
        seed=seed,
        simulation_horizon_seconds=horizon,
        controller=controller,
        demand_config=DemandScenarioConfig(
            TrafficScenario.NORMAL,
            random_seed=seed,
            simulation_horizon=horizon,
            arrival_rate=arrival_rate,
        ),
        event_config=TrafficEventConfig(
            random_seed=seed,
            simulation_horizon=horizon,
            event_count=event_count,
            minimum_duration_seconds=1,
            maximum_duration_seconds=2,
        ),
    )


def test_configuration_validates_horizon_controller_and_scenario() -> None:
    valid = ExperimentConfig.default("valid", controller=ControllerType.ADAPTIVE_RULE_BASED)
    assert valid.simulation_horizon_seconds > 0

    with pytest.raises(ValueError, match="horizon"):
        ExperimentConfig("bad", 1, 0)
    with pytest.raises(ValueError, match="unknown controller"):
        ExperimentConfig("bad", 1, 2, controller="unknown")


def test_runner_returns_metrics_demands_events_and_snapshots() -> None:
    result = ExperimentRunner().run(configuration(arrival_rate=0.2, event_count=2))

    assert result.execution_status == "completed"
    assert result.controller_name == "fixed_time"
    assert result.number_of_generated_vehicles == len(result.demands)
    assert result.number_of_generated_events == 2
    assert result.metrics is not None
    assert len(result.snapshots) == 5


def test_runner_supports_adaptive_controller() -> None:
    result = ExperimentRunner().run(
        configuration(ControllerType.ADAPTIVE_RULE_BASED, arrival_rate=0.1)
    )

    assert result.controller_name == "adaptive_rule_based"
    assert result.metrics.completed_vehicles >= 0


def test_same_seed_reproduces_structured_inputs_and_metrics() -> None:
    first = ExperimentRunner().run(configuration(seed=9, arrival_rate=0.1, event_count=1))
    second = ExperimentRunner().run(configuration(seed=9, arrival_rate=0.1, event_count=1))

    assert first.demands == second.demands
    assert first.events == second.events
    assert first.metrics == second.metrics
    assert first.snapshots == second.snapshots


def test_comparison_returns_measured_rows_without_ranking() -> None:
    results = (
        ExperimentRunner().run(configuration(seed=1)),
        ExperimentRunner().run(
            configuration(ControllerType.ADAPTIVE_RULE_BASED, seed=1)
        ),
    )

    rows = compare_results(results)

    assert len(rows) == 2
    assert {row.controller_name for row in rows} == {
        "fixed_time",
        "adaptive_rule_based",
    }
    assert not hasattr(rows[0], "rank")


def test_zero_vehicle_and_no_event_scenario_is_supported() -> None:
    result = ExperimentRunner().run(configuration())

    assert result.number_of_generated_vehicles == 0
    assert result.number_of_generated_events == 0
    assert result.metrics.throughput_vehicles_per_second == 0


def test_invalid_topology_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="topology"):
        ExperimentRunner(nx.DiGraph()).run(configuration())