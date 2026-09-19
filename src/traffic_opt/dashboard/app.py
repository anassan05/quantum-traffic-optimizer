"""Streamlit entry point for the traffic optimization dashboard."""

import streamlit as st

from traffic_opt.dashboard.data import (
    comparison_rows_to_dicts,
    result_to_event_rows,
    result_to_history_row,
    result_to_json,
    result_to_kpis,
    result_to_snapshot_rows,
    rows_to_csv,
)
from traffic_opt.dashboard.visualizations import (
    comparison_chart,
    event_timeline,
    metric_time_series,
    topology_chart,
)
from traffic_opt.domain.enums import ControllerType, TrafficEventType
from traffic_opt.demand import DemandScenarioConfig, TrafficScenario
from traffic_opt.events import TrafficEventConfig
from traffic_opt.experiments import ExperimentConfig, ExperimentRunner, compare_results
from traffic_opt.network.topology import create_default_topology


DEFAULT_DASHBOARD_HORIZON_SECONDS = 300


st.set_page_config(page_title="Quantum Traffic Optimizer", page_icon="🚦", layout="wide")


def main() -> None:
    st.title("Quantum-Enhanced Urban Traffic Optimization")
    st.caption("Experiment results, event schedules, and measured simulation metrics")
    _initialize_state()
    configuration = _sidebar_config()

    if st.sidebar.button("Run Experiment", type="primary", use_container_width=True):
        try:
            with st.spinner("Running the existing simulator..."):
                result = ExperimentRunner().run(configuration)
            st.session_state.results.append(result)
            st.session_state.selected_result = len(st.session_state.results) - 1
            st.sidebar.success("Experiment completed")
        except Exception as error:
            st.sidebar.error(f"Experiment failed: {error}")

    if st.sidebar.button("Clear history", use_container_width=True):
        st.session_state.results.clear()
        st.session_state.selected_result = None

    if not st.session_state.results:
        st.info("Configure an experiment in the sidebar and run it to see measured results.")
        return
    _render_dashboard()


def _initialize_state() -> None:
    if "results" not in st.session_state:
        st.session_state.results = []
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None


def _sidebar_config() -> ExperimentConfig:
    st.sidebar.header("Experiment")
    name = st.sidebar.text_input("Experiment name", value="dashboard-run")
    seed = st.sidebar.number_input("Random seed", min_value=0, value=7, step=1)
    horizon = st.sidebar.number_input(
        "Simulation horizon (s)",
        min_value=1,
        value=DEFAULT_DASHBOARD_HORIZON_SECONDS,
        step=1,
    )
    timestep = st.sidebar.number_input("Timestep (s)", min_value=0.1, value=1.0, step=0.1)
    scenario = st.sidebar.selectbox("Traffic scenario", list(TrafficScenario), format_func=lambda value: value.value.replace("_", " ").title())
    controller = st.sidebar.selectbox(
        "Controller",
        [ControllerType.FIXED_TIME, ControllerType.ADAPTIVE_RULE_BASED, ControllerType.QUANTUM_HYBRID],
        format_func=lambda value: {
            ControllerType.FIXED_TIME: "Fixed-Time",
            ControllerType.ADAPTIVE_RULE_BASED: "Adaptive",
            ControllerType.QUANTUM_HYBRID: "Quantum Hybrid",
        }[value],
    )
    st.sidebar.header("Events")
    event_types = st.sidebar.multiselect(
        "Enabled event types",
        list(TrafficEventType),
        default=[],
        format_func=lambda value: value.value.replace("_", " ").title(),
    )
    event_count = st.sidebar.number_input("Generated event count", min_value=0, value=0, step=1)
    demand = DemandScenarioConfig.for_scenario(
        scenario, random_seed=int(seed), simulation_horizon=int(horizon)
    )
    events = TrafficEventConfig(
        random_seed=int(seed),
        simulation_horizon=int(horizon),
        event_count=int(event_count),
        enabled_event_types=tuple(event_types),
        minimum_duration_seconds=1,
        maximum_duration_seconds=max(1, min(600, int(horizon))),
    )
    return ExperimentConfig(
        experiment_name=name,
        seed=int(seed),
        simulation_horizon_seconds=int(horizon),
        time_step_seconds=float(timestep),
        controller=controller,
        demand_config=demand,
        event_config=events,
    )


def _render_dashboard() -> None:
    results = st.session_state.results
    options = [f"{index}: {result.experiment_name} · {result.controller_name}" for index, result in enumerate(results)]
    selected = st.selectbox("Experiment result", options, index=st.session_state.selected_result or 0)
    index = int(selected.split(":", 1)[0])
    st.session_state.selected_result = index
    result = results[index]
    kpis = result_to_kpis(result)
    columns = st.columns(5)
    for column, (label, value) in zip(columns, list(kpis.items())[:5]):
        column.metric(label, f"{value:.3f}" if isinstance(value, float) else value)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, list(kpis.items())[5:]):
        column.metric(label, f"{value:.3f}" if isinstance(value, float) else value)

    st.subheader("Traffic performance")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(metric_time_series(result, "waiting_time_seconds", "Average waiting time", "Seconds"), use_container_width=True)
        st.plotly_chart(metric_time_series(result, "queue_length", "Queue length", "Vehicles"), use_container_width=True)
    with right:
        st.plotly_chart(metric_time_series(result, "travel_time_seconds", "Average travel time", "Seconds"), use_container_width=True)
        st.plotly_chart(metric_time_series(result, "co2_grams", "CO2 estimate", "Grams"), use_container_width=True)

    st.subheader("Traffic events")
    if result.events:
        st.plotly_chart(event_timeline(result), use_container_width=True)
        st.dataframe(result_to_event_rows(result), use_container_width=True)
    else:
        st.info("No scheduled events in this experiment.")

    st.subheader("Emergency vehicles")
    st.metric("Ambulances", result.ambulance_count)
    st.metric("Completed ambulances", result.completed_ambulances)
    st.write({
        "Average ambulance travel time (s)": result.average_ambulance_travel_time_seconds,
        "Average ambulance waiting time (s)": result.average_ambulance_waiting_time_seconds,
    })
    st.caption("Emergency Green Corridor execution details are not present in the experiment result.")

    st.subheader("Network topology")
    affected = {road for event in result.events for road in event.affected_road_ids}
    st.plotly_chart(topology_chart(create_default_topology(), affected), use_container_width=True)

    st.subheader("Experiment comparison")
    rows = compare_results(tuple(results))
    if rows:
        st.dataframe(comparison_rows_to_dicts(rows), use_container_width=True)
        st.plotly_chart(comparison_chart(rows, "average_waiting_time_seconds", "Average waiting time", "Seconds"), use_container_width=True)
        st.plotly_chart(comparison_chart(rows, "throughput_vehicles_per_second", "Throughput", "Vehicles/s"), use_container_width=True)

    st.subheader("Experiment history")
    st.dataframe([result_to_history_row(item) for item in results], use_container_width=True)
    st.download_button("Download selected result JSON", result_to_json(result), file_name=f"{result.experiment_name}.json", mime="application/json")
    st.download_button("Download comparison CSV", rows_to_csv(comparison_rows_to_dicts(rows)), file_name="comparison.csv", mime="text/csv")


if __name__ == "__main__":
    main()