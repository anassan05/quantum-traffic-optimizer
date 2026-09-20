from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.experiments.runner import analyze_experiment, run_experiment
from traffic_opt.experiments.scenarios import TrafficScenario
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.vehicles import create_emergency_vehicle, generate_vehicles


def make_integrated_scenario() -> TrafficScenario:
    graph = create_default_topology()
    intersections = create_default_intersections(graph)
    closed_road = "R_I2_I3"
    vehicles = list(generate_vehicles(graph, 4, seed=41, excluded_road_ids={closed_road}))
    vehicles.append(
        create_emergency_vehicle(
            graph,
            "AMB-1",
            "I1",
            "I3",
            ("R_I1_I2", closed_road),
        )
    )
    return TrafficScenario(
        scenario_id="phase18-integrated",
        seed=41,
        duration_seconds=4,
        time_step_seconds=1.0,
        graph=graph,
        intersections=intersections,
        initial_vehicles=tuple(vehicles),
        demand=len(vehicles),
        events=(
            TrafficEvent(
                "phase18-closure",
                TrafficEventType.ROAD_CLOSURE,
                0,
                4,
                (closed_road,),
            ),
        ),
    )


def test_phase18_integrates_controllers_events_emergency_and_metrics() -> None:
    scenario = make_integrated_scenario()
    result = run_experiment(scenario)
    analysis = analyze_experiment(result)

    assert set(result.controller_results) == {
        "fixed_time",
        "adaptive",
        "quantum_hybrid",
    }
    assert all(item.seed == scenario.seed for item in result.controller_results.values())
    assert all(item.number_of_vehicles == scenario.demand for item in result.controller_results.values())
    assert all(len(item.corridor_history) == 4 for item in result.controller_results.values())
    assert all(
        entry.status.value == "rejected"
        for entry in result.controller_results["quantum_hybrid"].corridor_history
    )
    assert all(
        item.metrics.emergency_vehicle_travel_time_seconds["AMB-1"] == 4.0
        for item in result.controller_results.values()
    )
    assert all(
        item.estimated_co2_emissions_kg
        == item.estimated_fuel_consumption_liters * 2.31
        for item in analysis.controller_metrics
    )
    assert all(
        comparison.qaoa_objective_value == comparison.classical_objective_value
        for comparison in analysis.qaoa_objectives
    )


def test_phase18_integrated_experiment_is_bitwise_reproducible() -> None:
    scenario = make_integrated_scenario()

    first = run_experiment(scenario)
    second = run_experiment(scenario)

    assert first == second
    assert analyze_experiment(first) == analyze_experiment(second)