from traffic_opt.experiments.runner import run_experiment
from traffic_opt.experiments.scenarios import clone_scenario, create_default_scenario


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