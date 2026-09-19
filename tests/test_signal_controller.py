import pytest

from traffic_opt.controllers import EmergencyPriorityStatus, SignalController
from traffic_opt.domain.models import SignalState
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology


def controller() -> SignalController:
    topology = create_default_topology()
    intersections = {
        intersection.id: intersection
        for intersection in create_default_intersections(topology)
    }
    movement_map = {
        ("I2", "R_I1_I2", "R_I2_I3"): "north_south",
        ("I2", "R_I3_I2", "R_I2_I1"): "east_west",
    }
    return SignalController(intersections, movement_map)


def test_priority_is_active_when_requested_movement_is_already_green() -> None:
    result = controller().request_emergency_priority(
        "I2", "R_I1_I2", "R_I2_I3", SignalState("I2", 0, 10)
    )

    assert result.status is EmergencyPriorityStatus.ACTIVE
    assert result.signal_state is not None
    assert result.signal_state.emergency_preemption is True


def test_minimum_green_prevents_early_phase_change() -> None:
    result = controller().request_emergency_priority(
        "I2", "R_I3_I2", "R_I2_I1", SignalState("I2", 0, 9)
    )

    assert result.status is EmergencyPriorityStatus.WAITING_FOR_CLEARANCE


def test_maximum_green_rejects_overdue_phase() -> None:
    result = controller().request_emergency_priority(
        "I2", "R_I3_I2", "R_I2_I1", SignalState("I2", 0, 61)
    )

    assert result.status is EmergencyPriorityStatus.REJECTED
    assert "maximum" in result.message


def test_yellow_and_all_red_clearance_are_required_before_activation() -> None:
    signal_state = SignalState("I2", 0, 10)
    signal_controller = controller()

    requested = signal_controller.request_emergency_priority(
        "I2", "R_I3_I2", "R_I2_I1", signal_state
    )
    yellow = signal_controller.advance_clearance("I2", 2)
    all_red = signal_controller.advance_clearance("I2", 3)
    active = signal_controller.advance_clearance("I2", 4)

    assert requested.status is EmergencyPriorityStatus.TRANSITIONING
    assert yellow.status is EmergencyPriorityStatus.WAITING_FOR_CLEARANCE
    assert all_red.status is EmergencyPriorityStatus.WAITING_FOR_CLEARANCE
    assert active.status is EmergencyPriorityStatus.ACTIVE
    assert active.yellow_elapsed_seconds == 3
    assert active.all_red_elapsed_seconds == 1


def test_release_clears_emergency_preemption() -> None:
    signal_controller = controller()
    signal_controller.request_emergency_priority(
        "I2", "R_I1_I2", "R_I2_I3", SignalState("I2", 0, 10)
    )

    released = signal_controller.release_emergency_priority("I2")

    assert released.status is EmergencyPriorityStatus.RELEASED
    assert released.signal_state is not None
    assert released.signal_state.emergency_preemption is False


@pytest.mark.parametrize(
    "priority_request",
    [
        ("missing", "R_I1_I2", "R_I2_I3", "I2"),
        ("I2", "missing", "R_I2_I3", "I2"),
        ("I2", "R_I1_I2", "missing", "I2"),
        ("I2", "R_I1_I2", "R_I2_I3", "I1"),
    ],
)
def test_invalid_priority_requests_are_rejected(
    priority_request: tuple[str, str, str, str],
) -> None:
    intersection_id, incoming, outgoing, state_intersection = priority_request
    result = controller().request_emergency_priority(
        intersection_id, incoming, outgoing, SignalState(state_intersection, 0, 10)
    )

    assert result.status is EmergencyPriorityStatus.REJECTED


def test_unmapped_or_unsupported_movements_are_rejected() -> None:
    signal_controller = controller()

    unmapped = signal_controller.request_emergency_priority(
        "I2", "R_I1_I2", "R_I2_I1", SignalState("I2", 0, 10)
    )
    unsupported = SignalController(
        signal_controller.intersections,
        {("I2", "R_I1_I2", "R_I2_I3"): "left_turn"},
    ).request_emergency_priority(
        "I2", "R_I1_I2", "R_I2_I3", SignalState("I2", 0, 10)
    )

    assert unmapped.status is EmergencyPriorityStatus.REJECTED
    assert unsupported.status is EmergencyPriorityStatus.REJECTED