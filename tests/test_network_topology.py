import networkx as nx

from traffic_opt.network.topology import create_default_topology


def test_default_topology_has_four_intersections_and_expected_roads() -> None:
    graph = create_default_topology()

    assert set(graph.nodes) == {"I1", "I2", "I3", "I4"}
    assert set(graph.edges) == {
        ("I1", "I2"),
        ("I2", "I1"),
        ("I2", "I3"),
        ("I3", "I2"),
        ("I3", "I4"),
        ("I4", "I3"),
    }


def test_road_attributes_are_valid() -> None:
    graph = create_default_topology()

    for start, end, attributes in graph.edges(data=True):
        assert attributes["road_id"] == f"R_{start}_{end}"
        assert attributes["length_meters"] > 0
        assert attributes["capacity"] > 0
        assert attributes["free_flow_speed_kmh"] > 0
        assert attributes["free_flow_travel_time_seconds"] > 0
        assert attributes["road_segment"].start_intersection_id == start
        assert attributes["road_segment"].end_intersection_id == end


def test_topology_generation_is_deterministic() -> None:
    first = create_default_topology()
    second = create_default_topology()

    assert nx.utils.graphs_equal(first, second)
    assert list(first.edges(data=True)) == list(second.edges(data=True))