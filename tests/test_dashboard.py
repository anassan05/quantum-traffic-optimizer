import json

import plotly.graph_objects as go

from traffic_opt.dashboard.data import (
    result_to_event_rows,
    result_to_json,
    result_to_kpis,
    result_to_snapshot_rows,
    rows_to_csv,
)
from traffic_opt.dashboard.visualizations import event_timeline, metric_time_series, topology_chart
from traffic_opt.domain.enums import ControllerType, TrafficEventType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.experiments import ExperimentConfig, ExperimentRunner, compare_results
from traffic_opt.events import TrafficEventConfig
from traffic_opt.demand import DemandScenarioConfig, TrafficScenario
from traffic_opt.network.topology import create_default_topology


def result():
    config = ExperimentConfig(
        "dashboard-test",
        seed=1,
        simulation_horizon_seconds=3,
        controller=ControllerType.FIXED_TIME,
        demand_config=DemandScenarioConfig(
            TrafficScenario.NORMAL, random_seed=1, simulation_horizon=3, arrival_rate=0.0
        ),
        event_config=TrafficEventConfig(
            random_seed=1,
            simulation_horizon=3,
            event_count=0,
        ),
    )
    return ExperimentRunner().run(config)


def test_kpis_snapshots_and_empty_result_data() -> None:
    experiment = result()
    kpis = result_to_kpis(experiment)
    assert "Average waiting time (s)" in kpis
    assert result_to_snapshot_rows(experiment)
    assert result_to_event_rows(experiment) == []


def test_event_rows_and_exports() -> None:
    experiment = result()
    event = TrafficEvent("close", TrafficEventType.ROAD_CLOSURE, 1, 2, ("R_I1_I2",))
    object.__setattr__(experiment, "events", (event,))

    rows = result_to_event_rows(experiment)
    payload = json.loads(result_to_json(experiment))
    assert rows[0]["end_time_seconds"] == 3
    assert payload["events"][0]["event_type"] == "road_closure"
    assert "event_id" in rows_to_csv(rows)


def test_plotly_figures_and_comparison_data() -> None:
    experiment = result()
    assert isinstance(metric_time_series(experiment, "queue_length", "Queue", "Vehicles"), go.Figure)
    assert isinstance(event_timeline(experiment), go.Figure)
    assert isinstance(topology_chart(create_default_topology()), go.Figure)
    rows = compare_results((experiment,))
    assert len(rows) == 1