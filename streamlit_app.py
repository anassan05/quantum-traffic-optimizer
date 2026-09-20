"""Streamlit dashboard for reproducible traffic-controller comparisons."""

from __future__ import annotations

from collections import Counter

import plotly.express as px
import streamlit as st

from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.experiments.runner import (
    ExperimentAnalysis,
    ExperimentResult,
    analyze_experiment,
    run_experiment,
)
from traffic_opt.experiments.scenarios import TrafficScenario
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.vehicles import create_emergency_vehicle, generate_vehicles


CONTROLLER_LABELS = {
    "fixed_time": "Fixed-Time",
    "adaptive": "Adaptive",
    "quantum_hybrid": "Quantum-Hybrid",
}


def build_scenario(
    seed: int,
    duration_seconds: int,
    time_step_seconds: float,
    vehicle_count: int,
    include_emergency: bool,
    close_emergency_road: bool,
) -> TrafficScenario:
    """Build one deterministic scenario shared by every controller run."""

    graph = create_default_topology()
    intersections = create_default_intersections(graph)
    closed_roads = frozenset({"R_I2_I3"}) if close_emergency_road else frozenset()
    vehicles = list(
        generate_vehicles(
            graph,
            vehicle_count,
            seed=seed,
            excluded_road_ids=closed_roads,
        )
    )
    if include_emergency:
        vehicles.append(
            create_emergency_vehicle(
                graph,
                "AMB-1",
                "I1",
                "I3",
                ("R_I1_I2", "R_I2_I3"),
            )
        )
    events: tuple[TrafficEvent, ...] = ()
    if close_emergency_road:
        events = (
            TrafficEvent(
                "dashboard_closure",
                TrafficEventType.ROAD_CLOSURE,
                0,
                duration_seconds,
                ("R_I2_I3",),
            ),
        )
    return TrafficScenario(
        scenario_id=f"dashboard-{seed}-{vehicle_count}-{int(include_emergency)}",
        seed=seed,
        duration_seconds=duration_seconds,
        time_step_seconds=time_step_seconds,
        graph=graph,
        intersections=intersections,
        initial_vehicles=tuple(vehicles),
        demand=len(vehicles),
        events=events,
    )


def _metric_rows(analysis: ExperimentAnalysis) -> list[dict[str, object]]:
    return [
        {
            "Controller": CONTROLLER_LABELS.get(item.controller_name, item.controller_name),
            "Completed vehicles": item.completed_vehicles,
            "Average wait (s)": round(item.average_waiting_time_seconds, 2),
            "Maximum wait (s)": round(item.maximum_waiting_time_seconds, 2),
            "Throughput (veh/s)": round(item.throughput_vehicles_per_second, 4),
            "Estimated fuel (L)": round(item.estimated_fuel_consumption_liters, 3),
            "Estimated CO2 (kg)": round(item.estimated_co2_emissions_kg, 3),
            "Emergency delay (s)": round(item.emergency_vehicle_delay_seconds, 2),
        }
        for item in analysis.controller_metrics
    ]


def _show_comparison_charts(analysis: ExperimentAnalysis) -> None:
    chart_rows = [
        {
            "Controller": CONTROLLER_LABELS.get(item.controller_name, item.controller_name),
            "Average waiting time (s)": item.average_waiting_time_seconds,
            "Maximum waiting time (s)": item.maximum_waiting_time_seconds,
            "Emergency delay (s)": item.emergency_vehicle_delay_seconds,
        }
        for item in analysis.controller_metrics
    ]
    waiting_data = [
        {"Controller": row["Controller"], "Metric": metric, "Seconds": row[metric]}
        for row in chart_rows
        for metric in (
            "Average waiting time (s)",
            "Maximum waiting time (s)",
            "Emergency delay (s)",
        )
    ]
    st.plotly_chart(
        px.bar(
            waiting_data,
            x="Controller",
            y="Seconds",
            color="Metric",
            barmode="group",
            title="Waiting and emergency delay",
        ),
        use_container_width=True,
    )

    throughput_data = [
        {
            "Controller": CONTROLLER_LABELS.get(item.controller_name, item.controller_name),
            "Throughput (veh/s)": item.throughput_vehicles_per_second,
            "Completed vehicles": item.completed_vehicles,
        }
        for item in analysis.controller_metrics
    ]
    st.plotly_chart(
        px.bar(
            throughput_data,
            x="Controller",
            y=["Throughput (veh/s)", "Completed vehicles"],
            barmode="group",
            title="Throughput and completed vehicles",
        ),
        use_container_width=True,
    )

    environmental_data = [
        {
            "Controller": CONTROLLER_LABELS.get(item.controller_name, item.controller_name),
            "Estimated fuel (L)": item.estimated_fuel_consumption_liters,
            "Estimated CO2 (kg)": item.estimated_co2_emissions_kg,
        }
        for item in analysis.controller_metrics
    ]
    st.plotly_chart(
        px.bar(
            environmental_data,
            x="Controller",
            y=["Estimated fuel (L)", "Estimated CO2 (kg)"],
            barmode="group",
            title="Transparent environmental estimates",
        ),
        use_container_width=True,
    )


def _show_qaoa_chart(analysis: ExperimentAnalysis) -> None:
    if not analysis.qaoa_objectives:
        st.info("No QAOA objective samples were recorded for this run.")
        return
    objective_rows = [
        {
            "Simulation step": item.simulation_step,
            "Intersection": item.intersection_id,
            "Objective type": objective_type,
            "Objective": value,
        }
        for item in analysis.qaoa_objectives
        for objective_type, value in (
            ("QAOA", item.qaoa_objective_value),
            ("Exact classical", item.classical_objective_value),
        )
    ]
    st.plotly_chart(
        px.line(
            objective_rows,
            x="Simulation step",
            y="Objective",
            color="Objective type",
            markers=True,
            hover_data=["Intersection"],
            title="QAOA objective versus exact classical objective",
        ),
        use_container_width=True,
    )
    st.dataframe(
        [
            {
                "Step": item.simulation_step,
                "Intersection": item.intersection_id,
                "QAOA objective": round(item.qaoa_objective_value, 4),
                "Exact classical objective": round(item.classical_objective_value, 4),
                "Matches": item.matches_classical,
                "Fallback used": item.used_fallback,
            }
            for item in analysis.qaoa_objectives
        ],
        use_container_width=True,
        hide_index=True,
    )


def _show_corridor(result: ExperimentResult) -> None:
    emergency_runs = [
        controller_result
        for controller_result in result.controller_results.values()
        if controller_result.metrics.emergency_vehicle_travel_time_seconds
    ]
    if not emergency_runs:
        st.info("No emergency vehicle is active in this scenario.")
        return
    selected = emergency_runs[0]
    statuses = Counter(item.status.value for item in selected.corridor_history)
    st.write(
        "Corridor lifecycle for the shared emergency vehicle: "
        + ", ".join(f"{status}: {count}" for status, count in sorted(statuses.items()))
    )
    metrics = selected.metrics
    st.dataframe(
        [
            {
                "Vehicle": vehicle_id,
                "Travel time (s)": round(travel_time, 2),
                "Delay (s)": round(metrics.emergency_vehicle_delay_seconds[vehicle_id], 2),
            }
            for vehicle_id, travel_time in metrics.emergency_vehicle_travel_time_seconds.items()
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Quantum Traffic Optimizer", layout="wide")
    st.title("Quantum-Enhanced Urban Traffic Optimization")
    st.caption("Deterministic controller comparison with integrated events, metrics, and emergency corridor state.")

    with st.sidebar:
        st.header("Scenario")
        seed = st.number_input("Seed", min_value=0, value=7, step=1)
        duration = st.number_input("Duration (seconds)", min_value=1, value=20, step=1)
        timestep = st.selectbox("Time step (seconds)", (0.5, 1.0, 2.0), index=1)
        vehicle_count = st.number_input("Regular vehicles", min_value=0, value=8, step=1)
        include_emergency = st.checkbox("Include emergency vehicle")
        close_emergency_road = st.checkbox(
            "Close emergency downstream road",
            disabled=not include_emergency,
        )
        run = st.button("Run comparison", type="primary", use_container_width=True)

    if run or "experiment_result" not in st.session_state:
        scenario = build_scenario(
            int(seed),
            int(duration),
            float(timestep),
            int(vehicle_count),
            include_emergency,
            close_emergency_road,
        )
        with st.spinner("Running identical scenario across all controllers..."):
            st.session_state.experiment_result = run_experiment(scenario)

    result: ExperimentResult = st.session_state.experiment_result
    analysis = analyze_experiment(result)
    st.subheader("Controller comparison")
    st.dataframe(_metric_rows(analysis), use_container_width=True, hide_index=True)
    st.caption("Fuel and CO2 values are transparent estimates, not measured environmental data.")

    st.subheader("Measured comparison charts")
    _show_comparison_charts(analysis)

    st.subheader("QAOA and exact classical objectives")
    st.caption("Objective values are reported measurements for this run; they do not establish quantum advantage.")
    _show_qaoa_chart(analysis)

    st.subheader("Emergency corridor")
    _show_corridor(result)

    st.subheader("Scenario integrity")
    st.write(
        f"Scenario `{result.scenario_id}` | seed `{result.seed}` | "
        f"duration `{result.simulation_duration_seconds}s` | timestep `{result.time_step_seconds}s`"
    )
    st.write("Every controller receives the same generated vehicles, graph, events, seed, duration, and timestep.")


if __name__ == "__main__":
    main()
