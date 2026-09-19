"""Deterministic fixed-time signal controller."""

from dataclasses import dataclass
from math import ceil

from traffic_opt.domain.constraints import (
    DomainValidationError,
    validate_phase_movements,
    validate_signal_timing,
    validate_signal_transition,
)
from traffic_opt.domain.models import SignalPhaseConfig
from traffic_opt.simulation.state import SignalMode, SimulationState

from .base import SignalDecision


@dataclass(frozen=True, slots=True)
class FixedTimeController:
    """Cycle every intersection through its configured phases by elapsed time.

    The controller returns decisions but does not mutate ``SimulationState``.
    Its green duration must be within each phase's configured minimum and
    maximum. Yellow and all-red intervals are separate decisions and never
    permit movement.
    """

    green_seconds: int | None = None
    yellow_seconds: int | None = None
    all_red_seconds: int | None = None

    def decide(self, state: SimulationState) -> tuple[SignalDecision, ...]:
        """Return one deterministic signal decision per intersection."""

        decisions = []
        for intersection_id in sorted(state.intersection_states):
            signal = state.intersection_states[intersection_id]
            self._validate_configuration(signal.phases)
            decision = self._decision_for_time(
                intersection_id,
                signal.phases,
                state.time_seconds,
            )
            decisions.append(decision)
        return tuple(decisions)

    def _validate_configuration(
        self,
        phases: tuple[SignalPhaseConfig, ...],
    ) -> None:
        if not phases:
            raise DomainValidationError("fixed-time controller requires signal phases")
        for phase in phases:
            validate_signal_timing(
                phase.min_green_seconds,
                phase.max_green_seconds,
                phase.yellow_seconds,
                phase.all_red_seconds,
            )
            validate_phase_movements(phase.movements)
            green = self._green_duration(phase)
            if green < phase.min_green_seconds:
                raise DomainValidationError(
                    f"green duration for {phase.name} is below minimum green time"
                )
            if green > phase.max_green_seconds:
                raise DomainValidationError(
                    f"green duration for {phase.name} exceeds maximum green time"
                )
            yellow = self._yellow_duration(phase)
            all_red = self._all_red_duration(phase)
            if not isinstance(yellow, int) or yellow <= 0:
                raise DomainValidationError("yellow duration must be a positive integer")
            if not isinstance(all_red, int) or all_red <= 0:
                raise DomainValidationError(
                    "all-red duration must be a positive integer"
                )
        for index, phase in enumerate(phases):
            next_phase = phases[(index + 1) % len(phases)]
            validate_signal_transition(
                phase,
                next_phase,
                self._green_duration(phase),
                self._yellow_duration(phase),
                self._all_red_duration(phase),
            )

    def _decision_for_time(
        self,
        intersection_id: str,
        phases: tuple[SignalPhaseConfig, ...],
        time_seconds: float,
    ) -> SignalDecision:
        durations = [
            (
                self._green_duration(phase),
                self._yellow_duration(phase),
                self._all_red_duration(phase),
            )
            for phase in phases
        ]
        cycle_seconds = sum(sum(duration) for duration in durations)
        offset = time_seconds % cycle_seconds
        for phase_index, (green, yellow, all_red) in enumerate(durations):
            if offset < green:
                return SignalDecision(
                    intersection_id, phase_index, "green", ceil(green - offset)
                )
            offset -= green
            if offset < yellow:
                return SignalDecision(
                    intersection_id, phase_index, "yellow", ceil(yellow - offset)
                )
            offset -= yellow
            if offset < all_red:
                return SignalDecision(
                    intersection_id, phase_index, "all_red", ceil(all_red - offset)
                )
            offset -= all_red
        raise RuntimeError("fixed-time phase schedule did not produce a decision")

    def _green_duration(self, phase: SignalPhaseConfig) -> int:
        return self.green_seconds if self.green_seconds is not None else phase.min_green_seconds

    def _yellow_duration(self, phase: SignalPhaseConfig) -> int:
        return self.yellow_seconds if self.yellow_seconds is not None else phase.yellow_seconds

    def _all_red_duration(self, phase: SignalPhaseConfig) -> int:
        return self.all_red_seconds if self.all_red_seconds is not None else phase.all_red_seconds

    @staticmethod
    def validate_transition(
        current_phase: SignalPhaseConfig,
        next_phase: SignalPhaseConfig,
        green_elapsed_seconds: int,
        yellow_elapsed_seconds: int,
        all_red_elapsed_seconds: int,
    ) -> None:
        """Expose the shared safety validator for controller integrations."""

        validate_signal_transition(
            current_phase,
            next_phase,
            green_elapsed_seconds,
            yellow_elapsed_seconds,
            all_red_elapsed_seconds,
        )