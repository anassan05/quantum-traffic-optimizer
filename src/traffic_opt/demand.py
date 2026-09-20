"""Seeded traffic demand generation for scenario experiments."""

from dataclasses import dataclass
from enum import Enum
import random
from typing import Final

import networkx as nx

from traffic_opt.domain.enums import VehicleType


class TrafficScenario(str, Enum):
    """Named demand intensity scenarios for the initial experiments."""

    LOW_TRAFFIC = "low_traffic"
    NORMAL = "normal"
    MORNING_PEAK = "morning_peak"
    EVENING_PEAK = "evening_peak"


@dataclass(frozen=True, slots=True)
class VehicleDemand:
    """A vehicle waiting to enter the simulation at a scheduled time."""

    vehicle_id: str
    spawn_time: float
    origin_intersection_id: str
    destination_intersection_id: str
    vehicle_type: VehicleType


@dataclass(frozen=True, slots=True)
class DemandScenarioConfig:
    """Configuration shared by seeded demand generation runs.

    ``demand_profile`` contains rate multipliers for equal-length periods over
    the horizon. The values are assumptions for repeatable experiments, not
    calibrated traffic measurements.
    """

    scenario: TrafficScenario
    random_seed: int = 0
    simulation_horizon: float = 3600.0
    arrival_rate: float = 0.01
    demand_profile: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.simulation_horizon < 0:
            raise ValueError("simulation horizon cannot be negative")
        if self.arrival_rate < 0:
            raise ValueError("arrival rate cannot be negative")
        profile = tuple(self.demand_profile)
        if not profile or any(multiplier < 0 for multiplier in profile):
            raise ValueError("demand profile must contain non-negative multipliers")
        if self.arrival_rate > 0 and not any(profile):
            raise ValueError("demand profile must contain a positive multiplier")
        object.__setattr__(self, "demand_profile", profile)

    @classmethod
    def for_scenario(
        cls,
        scenario: TrafficScenario,
        *,
        random_seed: int = 0,
        simulation_horizon: float = 3600.0,
    ) -> "DemandScenarioConfig":
        """Build a centralized default configuration for a named scenario."""

        arrival_rate, demand_profile = _SCENARIO_DEFAULTS[scenario]
        return cls(
            scenario=scenario,
            random_seed=random_seed,
            simulation_horizon=simulation_horizon,
            arrival_rate=arrival_rate,
            demand_profile=demand_profile,
        )


_SCENARIO_DEFAULTS: Final[dict[TrafficScenario, tuple[float, tuple[float, ...]]]] = {
    TrafficScenario.LOW_TRAFFIC: (0.004, (1.0,)),
    TrafficScenario.NORMAL: (0.01, (1.0,)),
    TrafficScenario.MORNING_PEAK: (0.01, (0.5, 1.0, 2.0, 1.0)),
    TrafficScenario.EVENING_PEAK: (0.01, (1.0, 2.0, 1.0, 0.5)),
}


class TrafficDemandGenerator:
    """Generate reproducible origin/destination demand from a NetworkX graph."""

    def __init__(
        self,
        config: DemandScenarioConfig,
        network: nx.Graph,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.network = network
        self.seed = config.random_seed if seed is None else seed

    def generate(self) -> tuple[VehicleDemand, ...]:
        """Generate demand using exponential inter-arrival times."""

        intersection_ids = self._intersection_ids()
        random_state = random.Random(self.seed)
        demand: list[VehicleDemand] = []
        spawn_time = 0.0

        while spawn_time <= self.config.simulation_horizon:
            rate = self._arrival_rate_at(spawn_time)
            if rate <= 0:
                break
            spawn_time += random_state.expovariate(rate)
            if spawn_time > self.config.simulation_horizon:
                break

            origin, destination = random_state.sample(intersection_ids, 2)
            demand.append(
                VehicleDemand(
                    vehicle_id=f"demand-{len(demand):06d}",
                    spawn_time=spawn_time,
                    origin_intersection_id=origin,
                    destination_intersection_id=destination,
                    vehicle_type=random_state.choice(tuple(VehicleType)),
                )
            )
        return tuple(demand)

    def _intersection_ids(self) -> tuple[str, ...]:
        intersection_ids = tuple(sorted(self.network.nodes))
        if len(intersection_ids) < 2:
            raise ValueError("demand generation requires at least two intersections")
        if any(not isinstance(intersection_id, str) for intersection_id in intersection_ids):
            raise ValueError("network intersection IDs must be strings")
        return intersection_ids

    def _arrival_rate_at(self, spawn_time: float) -> float:
        profile = self.config.demand_profile
        period_length = self.config.simulation_horizon / len(profile)
        if period_length == 0:
            multiplier = profile[0]
        else:
            period_index = min(int(spawn_time / period_length), len(profile) - 1)
            multiplier = profile[period_index]
        return self.config.arrival_rate * multiplier
