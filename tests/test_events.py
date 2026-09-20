import networkx as nx
import pytest

from traffic_opt.domain.enums import TrafficEventType
from traffic_opt.domain.models import TrafficEvent
from traffic_opt.simulation.events import EventManager


def make_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge("I1", "I2", road_id="r1")
    graph.add_edge("I2", "I3", road_id="r2")
    graph.add_edge("I3", "I4", road_id="r3")
    return graph


def event(
    event_id: str,
    event_type: TrafficEventType = TrafficEventType.CONGESTION,
    start: int = 10,
    duration: int = 5,
    roads: tuple[str, ...] = ("r1",),
) -> TrafficEvent:
    return TrafficEvent(event_id, event_type, start, duration, roads)


def test_active_events_include_start_and_expire_at_end() -> None:
    manager = EventManager(
        (event("late"), event("early", start=0, duration=12, roads=("r2",))),
        make_graph(),
    )

    assert manager.active_event_ids(0) == ("early",)
    assert manager.active_event_ids(10) == ("early", "late")
    assert manager.active_event_ids(14) == ("late",)
    assert manager.active_event_ids(15) == ()


def test_active_event_results_are_deterministically_sorted() -> None:
    manager = EventManager(
        (event("zulu", roads=("r2",)), event("alpha", roads=("r1",))),
        make_graph(),
    )

    assert tuple(item.id for item in manager.active_events(10)) == ("alpha", "zulu")
    assert manager.active_event_ids(10) == ("alpha", "zulu")
    assert manager.affected_road_ids(10) == ("r1", "r2")


def test_road_closure_is_only_active_during_its_interval() -> None:
    manager = EventManager(
        (event("closure", TrafficEventType.ROAD_CLOSURE, 20, 3, ("r2",)),),
        make_graph(),
    )

    assert manager.is_road_closed("r2", 19) is False
    assert manager.is_road_closed("r2", 20) is True
    assert manager.is_road_closed("r2", 22) is True
    assert manager.is_road_closed("r2", 23) is False
    assert manager.is_road_closed("r1", 20) is False


def test_empty_events_are_supported() -> None:
    manager = EventManager((), make_graph())

    assert manager.active_events(0) == ()
    assert manager.active_event_ids(0) == ()
    assert manager.affected_road_ids(0) == ()
    assert manager.is_road_closed("r1", 0) is False


def test_unknown_affected_road_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown road IDs"):
        EventManager((event("bad", roads=("missing",)),), make_graph())


def test_duplicate_event_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate event ID"):
        EventManager((event("same", start=0), event("same", start=10)), make_graph())


def test_overlapping_events_on_same_road_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        EventManager(
            (event("first", start=0, duration=10), event("second", start=9)),
            make_graph(),
        )


def test_overlapping_events_on_different_roads_are_allowed() -> None:
    manager = EventManager(
        (event("first", roads=("r1",)), event("second", roads=("r2",))),
        make_graph(),
    )

    assert manager.active_event_ids(10) == ("first", "second")


def test_negative_or_non_integer_time_is_rejected() -> None:
    manager = EventManager((), make_graph())

    with pytest.raises(ValueError, match="cannot be negative"):
        manager.active_events(-1)
    with pytest.raises(TypeError, match="non-negative integer"):
        manager.active_events(1.5)
    with pytest.raises(TypeError, match="non-negative integer"):
        manager.active_events(True)
