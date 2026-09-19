"""Typed contracts shared by traffic signal controllers."""

from dataclasses import dataclass
from typing import Protocol

from traffic_opt.simulation.state import SignalMode, SimulationState


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """A safe signal command for one intersection at a simulation instant."""

    intersection_id: str
    phase_index: int
    mode: SignalMode
    duration_seconds: int

    def __post_init__(self) -> None:
        if not self.intersection_id.strip():
            raise ValueError("decision intersection id must be non-empty")
        if self.phase_index < 0:
            raise ValueError("decision phase index cannot be negative")
        if self.duration_seconds <= 0:
            raise ValueError("decision duration must be greater than zero")


class SignalController(Protocol):
    """Protocol implemented by controllers that produce signal decisions."""

    def decide(self, state: SimulationState) -> tuple[SignalDecision, ...]:
        """Return deterministic, safety-validated decisions for the state."""
