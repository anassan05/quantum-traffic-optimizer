import networkx as nx
import pytest

from traffic_opt.demand import (
    DemandScenarioConfig,
    TrafficDemandGenerator,
    TrafficScenario,
)
from traffic_opt.domain.enums import VehicleType
from traffic_opt.network.topology import create_default_topology


def test_same_seed_generates_identical_demand() -> None:
    network = create_default_topology()
    config = DemandScenarioConfig.for_scenario(
        TrafficScenario.NORMAL, random_seed=17
    )

    first = TrafficDemandGenerator(config, network, seed=17).generate()
    second = TrafficDemandGenerator(config, network, seed=17).generate()

    assert first == second


def test_different_seeds_normally_generate_different_demand() -> None:
    network = create_default_topology()
    config = DemandScenarioConfig(
        TrafficScenario.NORMAL,
        simulation_horizon=3600,
        arrival_rate=0.02,
    )

    first = TrafficDemandGenerator(config, network, seed=1).generate()
    second = TrafficDemandGenerator(config, network, seed=2).generate()

    assert first != second


def test_generated_demand_uses_valid_network_endpoints_and_types() -> None:
    network = create_default_topology()
    config = DemandScenarioConfig.for_scenario(
        TrafficScenario.MORNING_PEAK, random_seed=9
    )
    demand = TrafficDemandGenerator(config, network).generate()
    intersection_ids = set(network.nodes)

    assert demand
    assert all(item.origin_intersection_id in intersection_ids for item in demand)
    assert all(item.destination_intersection_id in intersection_ids for item in demand)
    assert all(
        item.origin_intersection_id != item.destination_intersection_id
        for item in demand
    )
    assert all(0 <= item.spawn_time <= config.simulation_horizon for item in demand)
    assert all(isinstance(item.vehicle_type, VehicleType) for item in demand)


def test_scenario_presets_produce_different_intensities() -> None:
    network = create_default_topology()
    low_config = DemandScenarioConfig.for_scenario(
        TrafficScenario.LOW_TRAFFIC, random_seed=123
    )
    normal_config = DemandScenarioConfig.for_scenario(
        TrafficScenario.NORMAL, random_seed=123
    )

    low_demand = TrafficDemandGenerator(low_config, network).generate()
    normal_demand = TrafficDemandGenerator(normal_config, network).generate()

    assert len(normal_demand) > len(low_demand)


@pytest.mark.parametrize("node_count", [0, 1])
def test_generation_rejects_topology_with_fewer_than_two_intersections(
    node_count: int,
) -> None:
    network = nx.DiGraph()
    network.add_nodes_from(f"I{index}" for index in range(node_count))
    config = DemandScenarioConfig.for_scenario(TrafficScenario.NORMAL)

    with pytest.raises(ValueError, match="at least two intersections"):
        TrafficDemandGenerator(config, network).generate()


def test_scenario_configuration_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="positive multiplier"):
        DemandScenarioConfig(
            TrafficScenario.NORMAL,
            arrival_rate=0.01,
            demand_profile=(0.0, 0.0),
        )
