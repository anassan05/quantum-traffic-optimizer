from traffic_opt.experiments.runner import analyze_experiment, run_experiment
from traffic_opt.experiments.scenarios import clone_scenario, create_default_scenario
from traffic_opt.experiments.scenarios import TrafficScenario
from traffic_opt.domain.enums import VehicleType
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.metrics import SimulationMetrics
from traffic_opt.simulation.vehicles import VehicleState


def scenario_signature(scenario):
    return tuple(
        (
            vehicle.id,
            vehicle.current_road_id,
            vehicle.origin_intersection_id,
            vehicle.destination_intersection_id,
        )
        for vehicle in scenario.initial_vehicles
    )


def test_same_seed_is_reproducible_and_different_seed_changes_vehicles() -> None:
    first = create_default_scenario(seed=7, vehicle_count=6)
    second = create_default_scenario(seed=7, vehicle_count=6)
    different = create_default_scenario(seed=8, vehicle_count=6)

    assert scenario_signature(first) == scenario_signature(second)
    assert scenario_signature(first) != scenario_signature(different)


def test_scenario_can_be_cloned_without_shared_vehicle_state() -> None:
    original = create_default_scenario(seed=2, vehicle_count=3)
    clone = clone_scenario(original)

    assert scenario_signature(original) == scenario_signature(clone)
    assert original.graph is not clone.graph
    assert original.initial_vehicles[0] is not clone.initial_vehicles[0]


def test_runner_executes_all_controllers_with_same_horizon() -> None:
    scenario = create_default_scenario(
        seed=11,
        duration_seconds=4,
        vehicle_count=3,
    )
    result = run_experiment(scenario)

    assert set(result.controller_results) == {
        "fixed_time",
        "adaptive",
        "quantum_hybrid",
    }
    for controller_result in result.controller_results.values():
        assert controller_result.simulation_duration_seconds == 4
        assert len(controller_result.signal_decisions) == 4
        assert len(controller_result.queue_history) == 4
        assert len(controller_result.waiting_history) == 4
        assert controller_result.number_of_vehicles == 3


def test_controller_runs_receive_equivalent_initial_conditions() -> None:
    scenario = create_default_scenario(seed=13, duration_seconds=3, vehicle_count=5)
    result = run_experiment(scenario)

    assert {
        item.number_of_vehicles for item in result.controller_results.values()
    } == {5}
    assert {
        item.seed for item in result.controller_results.values()
    } == {13}


def test_scenario_is_not_mutated_and_repeat_is_reproducible() -> None:
    scenario = create_default_scenario(seed=19, duration_seconds=3, vehicle_count=4)
    before = scenario_signature(scenario)

    first = run_experiment(scenario)
    second = run_experiment(scenario)

    assert scenario_signature(scenario) == before
    assert first == second


def test_quantum_hybrid_results_include_comparison_records() -> None:
    scenario = create_default_scenario(seed=23, duration_seconds=1, vehicle_count=2)
    result = run_experiment(scenario)
    hybrid = result.controller_results["quantum_hybrid"]

    assert len(hybrid.qaoa_comparisons) == 1
    assert isinstance(hybrid.qaoa_comparisons[0], dict)


def test_all_controller_runs_expose_deterministic_metrics() -> None:
    scenario = create_default_scenario(seed=29, duration_seconds=2, vehicle_count=2)

    result = run_experiment(scenario)

    assert all(
        isinstance(controller_result.metrics, SimulationMetrics)
        for controller_result in result.controller_results.values()
    )
    assert all(
        controller_result.metrics.throughput_vehicles_per_second >= 0
        for controller_result in result.controller_results.values()
    )


def test_asymmetric_queued_traffic_produces_independent_controller_trajectories() -> None:
    graph = create_default_topology()
    intersections = create_default_intersections(graph)
    vehicles = tuple(
        VehicleState(
            f"V{index}",
            VehicleType.CAR,
            "I1",
            "I2",
            ("R_I1_I2",),
            "R_I1_I2",
            position_meters=249.0,
        )
        for index in range(1, 4)
    )
    scenario = TrafficScenario(
        scenario_id="asymmetric-stop-line",
        seed=43,
        duration_seconds=20,
        time_step_seconds=1.0,
        graph=graph,
        intersections=intersections,
        initial_vehicles=vehicles,
        demand=len(vehicles),
    )

    result = run_experiment(scenario)
    fixed = result.controller_results["fixed_time"]
    adaptive = result.controller_results["adaptive"]
    hybrid = result.controller_results["quantum_hybrid"]

    assert fixed.signal_decisions[14][1].phase_index == 1
    assert adaptive.signal_decisions[14][1].phase_index == 1
    assert hybrid.signal_decisions[14][1].phase_index == 1
    assert fixed.queue_history[15]["R_I1_I2"] == 0
    assert adaptive.queue_history[15]["R_I1_I2"] == 0
    assert hybrid.queue_history[15]["R_I1_I2"] == 0
    assert fixed.metrics.total_vehicles_completed == 3
    assert adaptive.metrics.total_vehicles_completed == 3
    assert hybrid.metrics.total_vehicles_completed == 3
    assert fixed.metrics is not adaptive.metrics
    assert adaptive.metrics is not hybrid.metrics
    assert hybrid.qaoa_comparisons[10]["I2"].classical_phase_index == 1
    assert hybrid.qaoa_comparisons[10]["I2"].qaoa_phase_index == 1


def test_analysis_compares_all_controllers_and_shared_scenario_metadata() -> None:
    scenario = create_default_scenario(
        seed=31,
        duration_seconds=2,
        time_step_seconds=0.5,
        vehicle_count=3,
    )

    result = run_experiment(scenario)
    analysis = analyze_experiment(result)

    assert [item.controller_name for item in analysis.controller_metrics] == [
        "fixed_time",
        "adaptive",
        "quantum_hybrid",
    ]
    assert {
        (item.simulation_duration_seconds, item.time_step_seconds, item.seed)
        for item in result.controller_results.values()
    } == {(2, 0.5, 31)}
    assert all(item.completed_vehicles >= 0 for item in analysis.controller_metrics)
    assert all(item.emergency_vehicle_delay_seconds == 0 for item in analysis.controller_metrics)


def test_analysis_is_reproducible_for_the_same_scenario() -> None:
    scenario = create_default_scenario(seed=37, duration_seconds=2, vehicle_count=2)

    assert analyze_experiment(run_experiment(scenario)) == analyze_experiment(
        run_experiment(scenario)
    )