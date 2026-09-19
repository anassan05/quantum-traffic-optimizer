"""Immutable configuration for one controlled traffic experiment."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from traffic_opt.domain.enums import ControllerType
from traffic_opt.demand import DemandScenarioConfig, TrafficScenario
from traffic_opt.events import TrafficEventConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Parameters required to assemble one experiment run."""

    experiment_name: str
    seed: int
    simulation_horizon_seconds: int
    controller: ControllerType | str = ControllerType.FIXED_TIME
    demand_config: DemandScenarioConfig | None = None
    event_config: TrafficEventConfig | None = None
    time_step_seconds: float = 1.0
    controller_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment name must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.simulation_horizon_seconds <= 0:
            raise ValueError("simulation horizon must be greater than zero")
        if self.time_step_seconds <= 0:
            raise ValueError("time step must be greater than zero")
        try:
            controller_value = (
                ControllerType.ADAPTIVE_RULE_BASED
                if self.controller == "adaptive"
                else ControllerType(self.controller)
            )
        except ValueError as error:
            raise ValueError(f"unknown controller: {self.controller}") from error
        object.__setattr__(self, "controller", controller_value)
        if self.demand_config is not None and (
            self.demand_config.simulation_horizon != self.simulation_horizon_seconds
        ):
            raise ValueError("demand horizon must match experiment horizon")
        if self.event_config is not None and (
            self.event_config.simulation_horizon != self.simulation_horizon_seconds
        ):
            raise ValueError("event horizon must match experiment horizon")
        object.__setattr__(
            self,
            "controller_config",
            MappingProxyType(dict(self.controller_config)),
        )

    @classmethod
    def default(
        cls,
        experiment_name: str,
        *,
        seed: int = 0,
        simulation_horizon_seconds: int = 20,
        controller: ControllerType | str = ControllerType.FIXED_TIME,
        scenario: TrafficScenario = TrafficScenario.NORMAL,
    ) -> "ExperimentConfig":
        """Create a configuration using the centralized demand defaults."""

        return cls(
            experiment_name=experiment_name,
            seed=seed,
            simulation_horizon_seconds=simulation_horizon_seconds,
            controller=controller,
            demand_config=DemandScenarioConfig.for_scenario(
                scenario,
                random_seed=seed,
                simulation_horizon=simulation_horizon_seconds,
            ),
            event_config=TrafficEventConfig(
                random_seed=seed,
                simulation_horizon=simulation_horizon_seconds,
                event_count=0,
            ),
        )