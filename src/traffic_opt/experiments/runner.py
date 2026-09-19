"""Run comparable controller experiments over identical scenarios."""

from dataclasses import dataclass
from typing import Mapping

from traffic_opt.controllers.adaptive import AdaptiveController
from traffic_opt.controllers.base import SignalController, SignalDecision
from traffic_opt.controllers.fixed_time import FixedTimeController
from traffic_opt.controllers.quantum_hybrid import QuantumHybridComparison, QuantumHybridController
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.state import SimulationState

from .scenarios import TrafficScenario, clone_scenario


@dataclass(frozen=True, slots=True)
class ControllerRunResult:
    """Raw trajectory output for one controller and one scenario."""

    controller_name: str
    scenario_id: str
    seed: int
    simulation_duration_seconds: int
    time_step_seconds: float
    number_of_vehicles: int
    completed_vehicles: int
    signal_decisions: tuple[tuple[SignalDecision, ...], ...]
    queue_history: tuple[Mapping[str, int], ...]
    waiting_history: tuple[Mapping[str, float], ...]
    qaoa_comparisons: tuple[Mapping[str, QuantumHybridComparison], ...]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Raw results for all controllers on one shared scenario."""

    scenario_id: str
    seed: int
    simulation_duration_seconds: int
    time_step_seconds: float
    controller_results: Mapping[str, ControllerRunResult]


def run_experiment(
    scenario: TrafficScenario,
    *,
    controller_config: Mapping[str, Mapping[str, object]] | None = None,
) -> ExperimentResult:
    """Run fixed-time, adaptive, and quantum-hybrid controllers fairly."""

    configuration = controller_config or scenario.controller_config
    controllers: tuple[tuple[str, SignalController], ...] = (
        ("fixed_time", FixedTimeController(**configuration.get("fixed_time", {}))),
        ("adaptive", AdaptiveController(**configuration.get("adaptive", {}))),
        (
            "quantum_hybrid",
            QuantumHybridController(**configuration.get("quantum_hybrid", {})),
        ),
    )
    results = {
        name: run_scenario_controller(name, controller, scenario)
        for name, controller in controllers
    }
    return ExperimentResult(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        simulation_duration_seconds=scenario.duration_seconds,
        time_step_seconds=scenario.time_step_seconds,
        controller_results=results,
    )


def run_scenario_controller(
    controller_name: str,
    controller: SignalController,
    scenario: TrafficScenario,
) -> ControllerRunResult:
    """Run one controller against a fresh clone of the scenario."""

    if not controller_name.strip():
        raise ValueError("controller name must be non-empty")
    isolated = clone_scenario(scenario)
    simulator = TrafficSimulator(
        isolated.graph,
        isolated.intersections,
        time_step_seconds=isolated.time_step_seconds,
        seed=isolated.seed,
    )
    simulator.add_vehicles(isolated.initial_vehicles)
    steps = int(isolated.duration_seconds / isolated.time_step_seconds)
    decisions: list[tuple[SignalDecision, ...]] = []
    queues: list[Mapping[str, int]] = []
    waiting: list[Mapping[str, float]] = []
    comparisons: list[Mapping[str, QuantumHybridComparison]] = []
    for _ in range(steps):
        selected = controller.decide(simulator.state)
        decisions.append(selected)
        queues.append(dict(simulator.state.queue_lengths))
        waiting.append(
            {
                vehicle.id: vehicle.waiting_time_seconds
                for vehicle in sorted(simulator.state.vehicles.values(), key=lambda item: item.id)
            }
        )
        if isinstance(controller, QuantumHybridController):
            comparisons.append(dict(controller.comparisons))
        else:
            comparisons.append({})
        _apply_decisions(simulator, selected)
        simulator.step()
    completed = sum(vehicle.completed for vehicle in simulator.state.vehicles.values())
    return ControllerRunResult(
        controller_name=controller_name,
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        simulation_duration_seconds=isolated.duration_seconds,
        time_step_seconds=isolated.time_step_seconds,
        number_of_vehicles=len(simulator.state.vehicles),
        completed_vehicles=completed,
        signal_decisions=tuple(decisions),
        queue_history=tuple(queues),
        waiting_history=tuple(waiting),
        qaoa_comparisons=tuple(comparisons),
    )


def _apply_decisions(
    simulator: TrafficSimulator,
    decisions: tuple[SignalDecision, ...],
) -> None:
    """Apply only changed decisions, preserving elapsed signal time."""

    for decision in decisions:
        signal = simulator.state.intersection_states[decision.intersection_id]
        if (
            signal.current_phase_index != decision.phase_index
            or signal.mode != decision.mode
        ):
            simulator.set_signal_phase(
                decision.intersection_id,
                decision.phase_index,
                decision.mode,
            )