from traffic_opt.dashboard.data import result_to_kpis
from traffic_opt.demand import DemandScenarioConfig, TrafficScenario
from traffic_opt.domain.enums import ControllerType, TrafficEventType
from traffic_opt.events import TrafficEventConfig
from traffic_opt.experiments import ExperimentConfig, ExperimentRunner


def experiment_config(
    *,
    controller: ControllerType = ControllerType.FIXED_TIME,
    seed: int = 21,
    scenario: TrafficScenario = TrafficScenario.NORMAL,
    event_types: tuple[TrafficEventType, ...] = (),
) -> ExperimentConfig:
    horizon = 8
    return ExperimentConfig(
        experiment_name=f"integration-{controller.value}-{scenario.value}",
        seed=seed,
        simulation_horizon_seconds=horizon,
        controller=controller,
        demand_config=DemandScenarioConfig(
            scenario,
            random_seed=seed,
            simulation_horizon=horizon,
            arrival_rate=0.2,
        ),
        event_config=TrafficEventConfig(
            random_seed=seed,
            simulation_horizon=horizon,
            event_count=1 if event_types else 0,
            enabled_event_types=event_types,
            minimum_duration_seconds=1,
            maximum_duration_seconds=1,
        ),
    )


def test_complete_pipeline_returns_dashboard_consumable_result() -> None:
    result = ExperimentRunner().run(experiment_config())

    assert result.number_of_generated_vehicles > 0
    assert result.metrics.total_vehicles <= result.number_of_generated_vehicles
    assert len(result.snapshots) == 9
    assert "Average waiting time (s)" in result_to_kpis(result)


def test_all_controllers_complete_the_same_configured_pipeline() -> None:
    for controller in ControllerType:
        result = ExperimentRunner().run(
            experiment_config(controller=controller, seed=22)
        )
        assert result.controller_name == controller.value
        assert result.execution_status == "completed"
        assert result.metrics is not None


def test_all_demand_scenarios_produce_valid_results() -> None:
    for scenario in TrafficScenario:
        result = ExperimentRunner().run(
            experiment_config(scenario=scenario, seed=23)
        )
        assert result.scenario == scenario.value
        assert result.number_of_generated_vehicles >= 0
        assert result.metrics is not None


def test_event_experiment_contains_scheduled_event_and_is_reproducible() -> None:
    config = experiment_config(
        seed=24,
        event_types=(TrafficEventType.CONGESTION,),
    )
    first = ExperimentRunner().run(config)
    second = ExperimentRunner().run(config)

    assert len(first.events) == 1
    assert first.events[0].event_type is TrafficEventType.CONGESTION
    assert first.events == second.events
    assert first.snapshots == second.snapshots
    assert first.metrics == second.metrics
