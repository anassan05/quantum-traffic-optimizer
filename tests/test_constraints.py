import pytest

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
)
from traffic_opt.domain.models import SignalPhaseConfig


def test_signal_configuration_accepts_non_conflicting_movements() -> None:
    validate_phase_movements({"north_south"})
    config = SignalPhaseConfig("through", {"north_south"}, 5, 45, 3, 2)

    assert config.movements == frozenset({"north_south"})


def test_signal_configuration_rejects_conflicting_movements() -> None:
    with pytest.raises(DomainValidationError, match="conflicting movements"):
        validate_phase_movements({"north_south", "east_west"})


def test_clearance_times_must_be_positive() -> None:
    with pytest.raises(DomainValidationError, match="all-red"):
        validate_signal_timing(5, 45, 3, 0)