import pytest

from traffic_opt.controllers.fixed_time import FixedTimeController
from traffic_opt.domain.constraints import DomainValidationError
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator


def make_state(time_seconds: float = 0):
    graph = create_default_topology()
    simulator = TrafficSimulator(graph, create_default_intersections(graph))
    simulator.state.time_seconds = time_seconds
    return simulator.state


def test_fixed_time_controller_cycles_phases_and_clearance() -> None:
    controller = FixedTimeController(green_seconds=10, yellow_seconds=3, all_red_seconds=1)

    assert controller.decide(make_state(0))[0].mode == "green"
    assert controller.decide(make_state(0))[0].phase_index == 0
    assert controller.decide(make_state(10))[0].mode == "yellow"
    assert controller.decide(make_state(13))[0].mode == "all_red"
    assert controller.decide(make_state(14))[0].phase_index == 1
    assert controller.decide(make_state(14))[0].mode == "green"


def test_green_duration_is_within_phase_bounds() -> None:
    controller = FixedTimeController(green_seconds=12, yellow_seconds=3, all_red_seconds=1)
    decision = controller.decide(make_state())[0]

    assert decision.duration_seconds == 12
    assert 10 <= decision.duration_seconds <= 60


def test_invalid_green_duration_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="below minimum"):
        FixedTimeController(green_seconds=9).decide(make_state())
    with pytest.raises(DomainValidationError, match="exceeds maximum"):
        FixedTimeController(green_seconds=61).decide(make_state())


def test_conflicting_phase_configuration_is_rejected() -> None:
    with pytest.raises(DomainValidationError, match="conflicting movements"):
        SignalPhaseConfig(
            "unsafe",
            frozenset({"north_south", "east_west"}),
            10,
            60,
        )


def test_decisions_are_deterministic_and_transition_validator_is_used() -> None:
    state = make_state(17)
    first = FixedTimeController().decide(state)
    second = FixedTimeController().decide(state)

    assert first == second
    phases = state.intersection_states["I1"].phases
    FixedTimeController.validate_transition(phases[0], phases[1], 10, 3, 1)