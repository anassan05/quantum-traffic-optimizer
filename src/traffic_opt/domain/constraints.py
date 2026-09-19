"""Validation rules for traffic domain configuration.

The functions in this module deliberately operate on primitive values and
simple collections so the domain layer remains independent of the simulator
and external libraries.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SignalPhaseConfig


class DomainValidationError(ValueError):
    """Raised when a domain object violates a traffic configuration rule."""


def validate_road_dimensions(capacity: int, length_meters: float) -> None:
    """Validate the physical dimensions required by a road segment."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise DomainValidationError("road capacity must be a positive integer")
    if length_meters <= 0:
        raise DomainValidationError("road length must be greater than zero")


def validate_signal_timing(
    min_green_seconds: int,
    max_green_seconds: int,
    yellow_seconds: int,
    all_red_seconds: int,
) -> None:
    """Validate green, yellow, and all-red signal timing values."""

    values = {
        "minimum green time": min_green_seconds,
        "maximum green time": max_green_seconds,
        "yellow clearance time": yellow_seconds,
        "all-red clearance time": all_red_seconds,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DomainValidationError(f"{name} must be a non-negative integer")
    if min_green_seconds >= max_green_seconds:
        raise DomainValidationError(
            "minimum green time must be less than maximum green time"
        )
    if yellow_seconds == 0:
        raise DomainValidationError("yellow clearance time must be greater than zero")
    if all_red_seconds == 0:
        raise DomainValidationError(
            "all-red clearance time must be greater than zero"
        )


def validate_signal_transition(
    current_phase: "SignalPhaseConfig",
    next_phase: "SignalPhaseConfig",
    green_elapsed_seconds: int,
    yellow_elapsed_seconds: int,
    all_red_elapsed_seconds: int,
    emergency_preemption: bool = False,
) -> None:
    """Validate that a phase change has completed every safety interval.

    ``emergency_preemption`` records the reason for a requested transition,
    but it intentionally does not relax any safety rule. Emergency control is
    implemented in a later phase; this validator remains the final safety gate.
    """

    if green_elapsed_seconds < current_phase.min_green_seconds:
        raise DomainValidationError("minimum green time has not elapsed")
    if green_elapsed_seconds > current_phase.max_green_seconds:
        raise DomainValidationError("maximum green time has been exceeded")
    if yellow_elapsed_seconds < current_phase.yellow_seconds:
        raise DomainValidationError("yellow clearance interval was skipped")
    if all_red_elapsed_seconds < current_phase.all_red_seconds:
        raise DomainValidationError("all-red clearance interval was skipped")
    validate_phase_movements(next_phase.movements)


_CONFLICTING_MOVEMENT_PAIRS = {
    frozenset({"north_south", "east_west"}),
    frozenset({"northbound", "southbound"}),
    frozenset({"eastbound", "westbound"}),
}


def validate_phase_movements(movements: Iterable[str]) -> None:
    """Reject known conflicting movement groups in one signal phase."""

    normalized = frozenset(movements)
    if not normalized or any(not movement for movement in normalized):
        raise DomainValidationError("a signal phase must contain named movements")
    for conflicting_pair in _CONFLICTING_MOVEMENT_PAIRS:
        if conflicting_pair <= normalized:
            names = " and ".join(sorted(conflicting_pair))
            raise DomainValidationError(
                f"signal phase contains conflicting movements: {names}"
            )