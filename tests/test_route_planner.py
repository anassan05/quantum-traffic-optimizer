import networkx as nx
import pytest

from traffic_opt.demand import VehicleDemand
from traffic_opt.domain.enums import VehicleType
from traffic_opt.emergency import AmbulanceRoutePlanner
from traffic_opt.network.topology import create_default_topology


def test_planner_returns_valid_road_ids_from_origin_to_destination() -> None:
    topology = create_default_topology()
    planner = AmbulanceRoutePlanner(topology)

    route = planner.plan("I1", "I4")

    assert route == ("R_I1_I2", "R_I2_I3", "R_I3_I4")
    assert all(
        road_id in {data["road_id"] for _, _, data in topology.edges(data=True)}
        for road_id in route
    )


def test_route_details_include_intersections_and_distance() -> None:
    planner = AmbulanceRoutePlanner(create_default_topology())

    details = planner.plan_details("I4", "I2")

    assert details.intersection_ids == ("I4", "I3", "I2")
    assert details.road_ids == ("R_I4_I3", "R_I3_I2")
    assert details.total_distance_meters == 500.0


def test_planner_uses_length_weighted_shortest_path() -> None:
    topology = nx.DiGraph()
    topology.add_edge("A", "B", road_id="R_AB", length_meters=100.0)
    topology.add_edge("A", "C", road_id="R_AC", length_meters=10.0)
    topology.add_edge("C", "B", road_id="R_CB", length_meters=10.0)

    assert AmbulanceRoutePlanner(topology).plan("A", "B") == ("R_AC", "R_CB")


def test_route_planning_is_deterministic() -> None:
    planner = AmbulanceRoutePlanner(create_default_topology())

    assert planner.plan("I1", "I4") == planner.plan("I1", "I4")


def test_invalid_origin_and_destination_fail_clearly() -> None:
    planner = AmbulanceRoutePlanner(create_default_topology())

    with pytest.raises(ValueError, match="origin intersection"):
        planner.plan("missing", "I2")
    with pytest.raises(ValueError, match="destination intersection"):
        planner.plan("I1", "missing")


def test_same_origin_and_destination_is_rejected() -> None:
    with pytest.raises(ValueError, match="must differ"):
        AmbulanceRoutePlanner(create_default_topology()).plan("I1", "I1")


def test_missing_path_fails_clearly() -> None:
    topology = nx.DiGraph()
    topology.add_node("A")
    topology.add_node("B")

    with pytest.raises(ValueError, match="no route exists"):
        AmbulanceRoutePlanner(topology).plan("A", "B")


def test_plan_for_ambulance_demand_reuses_phase_2_representation() -> None:
    demand = VehicleDemand("ambulance-1", 10.0, "I1", "I4", VehicleType.AMBULANCE)

    route = AmbulanceRoutePlanner(create_default_topology()).plan_for_demand(demand)

    assert route == ("R_I1_I2", "R_I2_I3", "R_I3_I4")


def test_plan_for_demand_rejects_non_ambulance() -> None:
    demand = VehicleDemand("car-1", 10.0, "I1", "I4", VehicleType.CAR)

    with pytest.raises(ValueError, match="requires an ambulance"):
        AmbulanceRoutePlanner(create_default_topology()).plan_for_demand(demand)