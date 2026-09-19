"""Immutable scenario definitions and deterministic scenario factories."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import networkx as nx

from traffic_opt.domain.models import Intersection, TrafficEvent
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.vehicles import VehicleState, generate_vehicles

from .seeds import DEFAULT_SCENARIO_SEED, validate_seed


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    """Complete external input shared by every controller run."""

    scenario_id: str
    seed: int
    duration_seconds: int
    time_step_seconds: float
    graph: nx.DiGraph
    intersections: tuple[Intersection, ...]
    initial_vehicles: tuple[VehicleState, ...]
    demand: int
    events: tuple[TrafficEvent, ...] = ()
    controller_config: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_seed(self.seed)
        if not self.scenario_id.strip():
            raise ValueError("scenario id must be non-empty")
        if self.duration_seconds <= 0:
            raise ValueError("simulation duration must be greater than zero")
        if self.time_step_seconds <= 0:
            raise ValueError("simulation timestep must be greater than zero")
        if self.demand < 0:
            raise ValueError("traffic demand cannot be negative")
        object.__setattr__(
            self,
            "controller_config",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(configuration))
                    for name, configuration in self.controller_config.items()
                }
            ),
        )


def create_default_scenario(
    seed: int = DEFAULT_SCENARIO_SEED,
    *,
    duration_seconds: int = 20,
    time_step_seconds: float = 1.0,
    vehicle_count: int = 8,
) -> TrafficScenario:
    """Create a deterministic four-intersection scenario."""

    validate_seed(seed)
    if vehicle_count < 0:
        raise ValueError("vehicle count cannot be negative")
    graph = create_default_topology()
    intersections = create_default_intersections(graph)
    vehicles = generate_vehicles(graph, vehicle_count, seed=seed)
    return TrafficScenario(
        scenario_id=f"corridor-{seed}-{vehicle_count}",
        seed=seed,
        duration_seconds=duration_seconds,
        time_step_seconds=time_step_seconds,
        graph=graph,
        intersections=intersections,
        initial_vehicles=vehicles,
        demand=vehicle_count,
    )


def clone_scenario(scenario: TrafficScenario) -> TrafficScenario:
    """Clone mutable scenario members for an independent controller run."""

    return TrafficScenario(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        duration_seconds=scenario.duration_seconds,
        time_step_seconds=scenario.time_step_seconds,
        graph=deepcopy(scenario.graph),
        intersections=scenario.intersections,
        initial_vehicles=tuple(deepcopy(vehicle) for vehicle in scenario.initial_vehicles),
        demand=scenario.demand,
        events=scenario.events,
        controller_config=scenario.controller_config,
    )