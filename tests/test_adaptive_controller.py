import pytest

from traffic_opt.controllers.adaptive import AdaptiveController
from traffic_opt.domain.enums import VehicleType
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.state import IntersectionSignalState
from traffic_opt.simulation.vehicles import VehicleState


def make_state(time_seconds: float = 0):
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    simulator.state.time_seconds = time_seconds
    return simulator.state


def custom_state() -> tuple[object, tuple[str, ...]]:
    state = make_state()
    phases = (
        SignalPhaseConfig("road_a_green", frozenset({"R_A"}), 10, 30),
        SignalPhaseConfig("road_b_green", frozenset({"R_B"}), 10, 30),
    )
    state.intersection_states["I1"] = IntersectionSignalState("I1", phases)
    state.queue_lengths = {"R_A": 0, "R_B": 0}
    state.road_occupancy = {"R_A": (), "R_B": ()}
    return state, ("R_A", "R_B")


def test_higher_queue_receives_priority_after_minimum_green() -> None:
    state, _ = custom_state()
    state.queue_lengths.update({"R_A": 1, "R_B": 8})
    state.intersection_states["I1"].elapsed_seconds = 10

    decision = AdaptiveController().decide(state)[0]

    assert decision.mode == "yellow"
    assert decision.phase_index == 0


def test_waiting_time_affects_priority() -> None:
    state, _ = custom_state()
    state.queue_lengths.update({"R_A": 2, "R_B": 2})
    state.vehicles["a"] = VehicleState("a", VehicleType.CAR, "I1", "I2", ("R_B",), "R_B", waiting_time_seconds=20)
    state.intersection_states["I1"].elapsed_seconds = 10

    assert AdaptiveController().decide(state)[0].mode == "yellow"


def test_minimum_green_is_respected() -> None:
    state, _ = custom_state()
    state.queue_lengths.update({"R_A": 0, "R_B": 20})
    state.intersection_states["I1"].elapsed_seconds = 5

    decision = AdaptiveController().decide(state)[0]

    assert decision.mode == "green"
    assert decision.phase_index == 0
    assert decision.duration_seconds == 5


def test_maximum_green_forces_yellow_transition() -> None:
    state, _ = custom_state()
    state.intersection_states["I1"].elapsed_seconds = 30

    decision = AdaptiveController().decide(state)[0]

    assert decision.mode == "yellow"
    assert decision.duration_seconds == 3


def test_yellow_then_all_red_then_next_green() -> None:
    state, _ = custom_state()
    state.intersection_states["I1"].mode = "yellow"
    state.intersection_states["I1"].elapsed_seconds = 1
    assert AdaptiveController().decide(state)[0].mode == "yellow"

    state.intersection_states["I1"].mode = "all_red"
    state.intersection_states["I1"].elapsed_seconds = 1
    decision = AdaptiveController().decide(state)[0]
    assert decision.mode == "green"
    assert decision.phase_index == 0


def test_conflicting_greens_are_impossible() -> None:
    state = make_state()
    decisions = AdaptiveController().decide(state)

    assert all(decision.mode != "green" or decision.phase_index >= 0 for decision in decisions)
    assert all(
        len(state.intersection_states[decision.intersection_id].phases[decision.phase_index].movements)
        == 1
        for decision in decisions
        if decision.mode == "green"
    )


def test_same_state_and_empty_equal_traffic_are_deterministic() -> None:
    state, _ = custom_state()
    controller = AdaptiveController()

    assert controller.decide(state) == controller.decide(state)
    assert controller.decide(state)[0].phase_index == 0


def test_emergency_vehicle_does_not_bypass_safety() -> None:
    state, _ = custom_state()
    state.vehicles["ambulance"] = VehicleState(
        "ambulance", VehicleType.AMBULANCE, "I1", "I2", ("R_B",), "R_B", waiting_time_seconds=50
    )
    state.intersection_states["I1"].mode = "yellow"

    decision = AdaptiveController().decide(state)[0]

    assert decision.mode == "yellow"
    assert decision.phase_index == 0