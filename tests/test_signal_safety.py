import pytest

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_signal_transition,
)
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.network.intersections import (
    create_default_intersections,
    create_default_signal_phases,
)
from traffic_opt.network.topology import create_default_topology


def test_default_signal_phases_are_valid_and_include_clearance_intervals() -> None:
    phases = create_default_signal_phases()
    intersections = create_default_intersections(create_default_topology())

    assert len(phases) == 2
    assert {phase.movements for phase in phases} == {
        frozenset({"north_south"}),
        frozenset({"east_west"}),
    }
    assert all(phase.min_green_seconds < phase.max_green_seconds for phase in phases)
    assert all(phase.yellow_seconds > 0 for phase in phases)
    assert all(phase.all_red_seconds > 0 for phase in phases)
    assert len(intersections) == 4
    assert all(intersection.signal_phases == phases for intersection in intersections)


def test_conflicting_movements_are_rejected() -> None:
    with pytest.raises(DomainValidationError, match="conflicting movements"):
        SignalPhaseConfig(
            "unsafe",
            frozenset({"north_south", "east_west"}),
            min_green_seconds=10,
            max_green_seconds=60,
        )


def test_invalid_timing_values_are_rejected() -> None:
    with pytest.raises(DomainValidationError, match="minimum green"):
        SignalPhaseConfig("unsafe", frozenset({"north_south"}), 60, 10)
    with pytest.raises(DomainValidationError, match="yellow"):
        SignalPhaseConfig("unsafe", frozenset({"north_south"}), 10, 60, 0, 1)
    with pytest.raises(DomainValidationError, match="all-red"):
        SignalPhaseConfig("unsafe", frozenset({"north_south"}), 10, 60, 3, 0)


def test_signal_transition_requires_green_and_clearance_intervals() -> None:
    current, next_phase = create_default_signal_phases()

    with pytest.raises(DomainValidationError, match="minimum green"):
        validate_signal_transition(current, next_phase, 9, 3, 1)
    with pytest.raises(DomainValidationError, match="yellow"):
        validate_signal_transition(current, next_phase, 10, 2, 1)
    with pytest.raises(DomainValidationError, match="all-red"):
        validate_signal_transition(current, next_phase, 10, 3, 0)

    validate_signal_transition(current, next_phase, 10, 3, 1)


def test_emergency_preemption_does_not_bypass_signal_safety() -> None:
    current, next_phase = create_default_signal_phases()

    with pytest.raises(DomainValidationError, match="all-red"):
        validate_signal_transition(
            current,
            next_phase,
            green_elapsed_seconds=10,
            yellow_elapsed_seconds=3,
            all_red_elapsed_seconds=0,
            emergency_preemption=True,
        )