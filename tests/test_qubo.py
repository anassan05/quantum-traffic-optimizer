import pytest

from traffic_opt.domain.constraints import DomainValidationError
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.quantum.qubo import QUBOWeights, build_signal_phase_qubo
from traffic_opt.simulation.simulator import TrafficSimulator


def make_state():
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    simulator.state.queue_lengths = {"R_A": 2, "R_B": 7}
    simulator.state.road_occupancy = {"R_A": ("v1",), "R_B": ("v2", "v3")}
    return simulator.state


def phases():
    return (
        SignalPhaseConfig("a", frozenset({"R_A"}), 10, 30),
        SignalPhaseConfig("b", frozenset({"R_B"}), 10, 30),
    )


def test_variable_generation_is_deterministic() -> None:
    first = build_signal_phase_qubo(make_state(), phases())
    second = build_signal_phase_qubo(make_state(), phases())

    assert first.variables == ("phase_000", "phase_001")
    assert first == second


def test_linear_terms_reward_higher_demand_phase() -> None:
    qubo = build_signal_phase_qubo(make_state(), phases(), current_phase_index=0)

    assert qubo.linear_terms[1].coefficient < qubo.linear_terms[0].coefficient


def test_quadratic_conflict_penalty_is_included() -> None:
    candidates = (
        SignalPhaseConfig("ns", frozenset({"north_south"}), 10, 30),
        SignalPhaseConfig("ew", frozenset({"east_west"}), 10, 30),
    )
    qubo = build_signal_phase_qubo(make_state(), candidates)

    assert qubo.quadratic_terms[0].coefficient == 300


def test_switching_penalty_changes_linear_term() -> None:
    without_switch = build_signal_phase_qubo(
        make_state(), phases(), current_phase_index=1,
        weights=QUBOWeights(phase_switching=0),
    )
    with_switch = build_signal_phase_qubo(
        make_state(), phases(), current_phase_index=1,
        weights=QUBOWeights(phase_switching=5),
    )

    assert with_switch.linear_terms[0].coefficient == without_switch.linear_terms[0].coefficient + 5
    assert with_switch.linear_terms[1].coefficient == without_switch.linear_terms[1].coefficient


def test_invalid_phase_and_empty_candidates_are_rejected() -> None:
    with pytest.raises(DomainValidationError, match="at least one"):
        build_signal_phase_qubo(make_state(), ())
    with pytest.raises(DomainValidationError, match="conflicting movements"):
        build_signal_phase_qubo(
            make_state(),
            (SignalPhaseConfig("bad", frozenset({"north_south", "east_west"}), 10, 30),),
        )


def test_small_example_can_be_manually_evaluated() -> None:
    qubo = build_signal_phase_qubo(
        make_state(),
        phases(),
        weights=QUBOWeights(
            queue_length=1,
            waiting_time=0,
            waiting_vehicles=0,
            road_occupancy=0,
            phase_switching=0,
            fairness=0,
            one_hot_penalty=10,
            conflict_penalty=5,
        ),
    )

    assert qubo.evaluate({"phase_000": 1, "phase_001": 0}) == -2
    assert qubo.evaluate({"phase_000": 0, "phase_001": 1}) == -7