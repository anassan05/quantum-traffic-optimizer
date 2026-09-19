from unittest.mock import patch

import pytest

from traffic_opt.controllers.quantum_hybrid import QuantumHybridController
from traffic_opt.domain.enums import VehicleType
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.quantum.qaoa_solver import QAOASolveResult
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.state import IntersectionSignalState
from traffic_opt.simulation.vehicles import VehicleState


def make_state():
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    return simulator.state


def custom_state():
    state = make_state()
    phases = (
        SignalPhaseConfig("a", frozenset({"R_A"}), 10, 30),
        SignalPhaseConfig("b", frozenset({"R_B"}), 10, 30),
    )
    state.intersection_states["I1"] = IntersectionSignalState("I1", phases)
    state.queue_lengths = {"R_A": 1, "R_B": 10}
    state.road_occupancy = {"R_A": (), "R_B": ()}
    return state


def test_controller_initialization_and_qubo_comparison() -> None:
    controller = QuantumHybridController(shots=64, qaoa_depth=1, seed=4)
    state = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 10

    decisions = controller.decide(state)

    assert decisions[0].mode in {"yellow", "green"}
    assert "I1" in controller.comparisons
    assert controller.comparisons["I1"].classical_phase_index == 1


def test_qaoa_selected_phase_becomes_only_a_safe_candidate() -> None:
    state = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 10

    decision = QuantumHybridController(shots=64, seed=5).decide(state)[0]

    assert decision.phase_index == 0
    assert decision.mode == "yellow"
    assert decision.duration_seconds == 3


def test_minimum_and_maximum_green_safety() -> None:
    state = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 5
    decision = QuantumHybridController(shots=32).decide(state)[0]
    assert decision.mode == "green"
    assert decision.duration_seconds == 5

    state.intersection_states["I1"].elapsed_seconds = 30
    decision = QuantumHybridController(shots=32).decide(state)[0]
    assert decision.mode == "yellow"


def test_yellow_and_all_red_transitions_are_preserved() -> None:
    state = custom_state()
    signal = state.intersection_states["I1"]
    signal.mode = "yellow"
    signal.elapsed_seconds = 1
    assert QuantumHybridController(shots=16).decide(state)[0].mode == "yellow"
    signal.elapsed_seconds = 3
    assert QuantumHybridController(shots=16).decide(state)[0].mode == "all_red"
    signal.mode = "all_red"
    signal.elapsed_seconds = 0
    assert QuantumHybridController(shots=16).decide(state)[0].mode == "all_red"


def test_qaoa_failure_falls_back_to_exact_classical_solution() -> None:
    state = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 10
    with patch(
        "traffic_opt.controllers.quantum_hybrid.qaoa_solver.solve_qaoa",
        side_effect=RuntimeError("Aer unavailable"),
    ):
        controller = QuantumHybridController(shots=16)
        decision = controller.decide(state)[0]

    assert decision.mode == "yellow"
    assert controller.comparisons["I1"].used_fallback is True
    assert controller.comparisons["I1"].qaoa_phase_index is None


def test_invalid_qaoa_solution_falls_back_safely() -> None:
    state = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 10
    invalid_result = QAOASolveResult(
        assignment={"phase_000": 1, "phase_001": 1},
        selected_variables=("phase_000", "phase_001"),
        objective_value=-999,
        number_of_variables=2,
        shots=16,
        qaoa_depth=1,
    )
    with patch(
        "traffic_opt.controllers.quantum_hybrid.qaoa_solver.solve_qaoa",
        return_value=invalid_result,
    ):
        controller = QuantumHybridController(shots=16)
        decision = controller.decide(state)[0]

    assert decision.mode == "yellow"
    assert controller.comparisons["I1"].used_fallback is True


def test_deterministic_seeded_behavior_and_emergency_safety() -> None:
    first_state = custom_state()
    second_state = custom_state()
    for state in (first_state, second_state):
        state.intersection_states["I1"].elapsed_seconds = 10
        state.vehicles["ambulance"] = VehicleState(
            "ambulance",
            VehicleType.AMBULANCE,
            "I1",
            "I2",
            ("R_B",),
            "R_B",
            waiting_time_seconds=50,
        )
    first = QuantumHybridController(shots=64, seed=9)
    second = QuantumHybridController(shots=64, seed=9)

    assert first.decide(first_state) == second.decide(second_state)
    assert first.decide(first_state)[0].mode == "yellow"


def test_conflicting_phase_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting movements"):
        SignalPhaseConfig("unsafe", frozenset({"north_south", "east_west"}), 10, 30)