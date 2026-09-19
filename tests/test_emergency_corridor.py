import pytest

from traffic_opt.controllers import EmergencyPriorityStatus, SignalController
from traffic_opt.domain.models import SignalState
from traffic_opt.emergency import (
    CorridorStatus,
    EmergencyGreenCorridor,
)
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology


def corridor(
    route: tuple[str, ...] = ("R_I1_I2", "R_I2_I3", "R_I3_I4"),
    *,
    states: dict[str, SignalState] | None = None,
    movement_map: dict[tuple[str, str, str], str] | None = None,
) -> EmergencyGreenCorridor:
    topology = create_default_topology()
    intersections = {
        intersection.id: intersection
        for intersection in create_default_intersections(topology)
    }
    mapping = movement_map if movement_map is not None else {
        ("I2", "R_I1_I2", "R_I2_I3"): "north_south",
        ("I3", "R_I2_I3", "R_I3_I4"): "north_south",
    }
    controller = SignalController(intersections, mapping)
    signal_states = states if states is not None else {
        "I2": SignalState("I2", 0, 10),
        "I3": SignalState("I3", 0, 10),
    }
    return EmergencyGreenCorridor(
        "ambulance-1", route, topology, controller, signal_states
    )


def test_route_mapping_preserves_ordered_intersections_and_movements() -> None:
    state = corridor().state

    assert state.movements == (
        state.movements[0].__class__("I2", "R_I1_I2", "R_I2_I3"),
        state.movements[0].__class__("I3", "R_I2_I3", "R_I3_I4"),
    )


def test_start_requests_first_movement_and_active_result() -> None:
    state = corridor().start()

    assert state.status is CorridorStatus.ACTIVE
    assert state.current_movement is not None
    assert state.current_movement.intersection_id == "I2"
    assert state.priority_result is not None
    assert state.priority_result.status is EmergencyPriorityStatus.ACTIVE


def test_clear_releases_current_intersection_and_progresses() -> None:
    emergency_corridor = corridor()
    emergency_corridor.start()

    state = emergency_corridor.clear_current_intersection()

    assert state.completed_intersections == ("I2",)
    assert state.current_movement is not None
    assert state.current_movement.intersection_id == "I3"
    assert state.priority_result is not None
    assert state.priority_result.status is EmergencyPriorityStatus.ACTIVE


def test_final_clear_releases_and_completes_corridor() -> None:
    emergency_corridor = corridor()
    emergency_corridor.start()
    emergency_corridor.clear_current_intersection()

    state = emergency_corridor.clear_current_intersection()

    assert state.status is CorridorStatus.COMPLETED
    assert state.is_complete
    assert state.completed_intersections == ("I2", "I3")
    assert state.priority_result is not None
    assert state.priority_result.status is EmergencyPriorityStatus.RELEASED


def test_transitioning_result_uses_controller_clearance() -> None:
    emergency_corridor = corridor(
        ("R_I3_I2", "R_I2_I1"),
        states={"I2": SignalState("I2", 0, 10)},
        movement_map={("I2", "R_I3_I2", "R_I2_I1"): "east_west"},
    )

    emergency_corridor.start()
    waiting = emergency_corridor.advance_clearance(2)
    active = emergency_corridor.advance_clearance(4)

    assert waiting.status is CorridorStatus.WAITING_FOR_CLEARANCE
    assert active.status is CorridorStatus.ACTIVE


def test_waiting_for_minimum_green_can_retry_with_updated_signal_state() -> None:
    emergency_corridor = corridor(
        ("R_I3_I2", "R_I2_I1"),
        states={"I2": SignalState("I2", 0, 9)},
        movement_map={("I2", "R_I3_I2", "R_I2_I1"): "east_west"},
    )

    waiting = emergency_corridor.start()
    transitioning = emergency_corridor.request_priority(SignalState("I2", 0, 10))

    assert waiting.status is CorridorStatus.WAITING_FOR_CLEARANCE
    assert transitioning.status is CorridorStatus.TRANSITIONING


def test_rejected_request_is_exposed_without_bypassing_controller() -> None:
    emergency_corridor = corridor(
        ("R_I1_I2", "R_I2_I3"),
        movement_map={},
    )

    state = emergency_corridor.start()

    assert state.status is CorridorStatus.REJECTED
    assert state.priority_result is not None
    assert state.priority_result.status is EmergencyPriorityStatus.REJECTED


def test_invalid_and_empty_routes_are_handled() -> None:
    topology = create_default_topology()
    intersections = {
        intersection.id: intersection
        for intersection in create_default_intersections(topology)
    }
    controller = SignalController(intersections, {})
    states = {"I2": SignalState("I2", 0, 10)}

    assert EmergencyGreenCorridor(
        "empty", (), topology, controller, states
    ).start().status is CorridorStatus.COMPLETED
    with pytest.raises(ValueError, match="unknown road"):
        EmergencyGreenCorridor(
            "bad", ("missing",), topology, controller, states
        )
    with pytest.raises(ValueError, match="disconnected"):
        EmergencyGreenCorridor(
            "broken", ("R_I1_I2", "R_I3_I4"), topology, controller, states
        )


def test_missing_signal_state_rejects_request_and_lifecycle_guards_apply() -> None:
    emergency_corridor = corridor(states={})

    state = emergency_corridor.start()

    assert state.status is CorridorStatus.REJECTED
    with pytest.raises(RuntimeError, match="not active"):
        emergency_corridor.clear_current_intersection()
    with pytest.raises(RuntimeError, match="already been started"):
        emergency_corridor.start()


def test_repeated_intersection_is_rejected() -> None:
    with pytest.raises(ValueError, match="repeats an intersection"):
        corridor(("R_I1_I2", "R_I2_I3", "R_I3_I2", "R_I2_I1"))