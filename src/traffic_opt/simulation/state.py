"""Typed mutable state containers for the discrete-time simulation."""

from dataclasses import dataclass, field
from typing import Literal

from traffic_opt.domain.models import SignalPhaseConfig

from .vehicles import VehicleState

SignalMode = Literal["green", "yellow", "all_red"]


@dataclass(slots=True)
class IntersectionSignalState:
    """Runtime signal state; yellow and all-red never permit movement."""

    intersection_id: str
    phases: tuple[SignalPhaseConfig, ...]
    current_phase_index: int = 0
    mode: SignalMode = "green"
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("signal state requires at least one phase")
        if not 0 <= self.current_phase_index < len(self.phases):
            raise ValueError("signal phase index is out of range")
        if self.elapsed_seconds < 0:
            raise ValueError("signal elapsed time cannot be negative")

    @property
    def current_phase(self) -> SignalPhaseConfig:
        """Return the configured phase currently selected by the signal."""

        return self.phases[self.current_phase_index]

    @property
    def permits_movement(self) -> bool:
        """Whether vehicles may cross this intersection right now."""

        return self.mode == "green"


@dataclass(slots=True)
class SimulationState:
    """Mutable snapshot containing all state needed for one simulation run."""

    time_seconds: float = 0.0
    vehicles: dict[str, VehicleState] = field(default_factory=dict)
    queue_lengths: dict[str, int] = field(default_factory=dict)
    intersection_states: dict[str, IntersectionSignalState] = field(
        default_factory=dict
    )
    road_occupancy: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def refresh_road_occupancy(self) -> None:
        """Rebuild deterministic road occupancy from active vehicles."""

        occupancy: dict[str, list[str]] = {}
        for vehicle in sorted(self.vehicles.values(), key=lambda item: item.id):
            if not vehicle.completed:
                occupancy.setdefault(vehicle.current_road_id, []).append(vehicle.id)
        self.road_occupancy = {
            road_id: tuple(vehicle_ids)
            for road_id, vehicle_ids in sorted(occupancy.items())
        }
