"""Run comparable controller experiments over identical scenarios."""

from dataclasses import dataclass
from typing import Mapping

from traffic_opt.controllers.adaptive import AdaptiveController
from traffic_opt.controllers.base import SignalController, SignalDecision
from traffic_opt.controllers.fixed_time import FixedTimeController
from traffic_opt.controllers.quantum_hybrid import QuantumHybridComparison, QuantumHybridController
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.metrics import SimulationMetrics
from traffic_opt.simulation.state import SimulationState
from traffic_opt.controllers.emergency_corridor import EmergencyCorridorResult

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
    metrics: SimulationMetrics
    corridor_history: tuple[EmergencyCorridorResult, ...]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Raw results for all controllers on one shared scenario."""

    scenario_id: str
    seed: int
    simulation_duration_seconds: int
    time_step_seconds: float
    controller_results: Mapping[str, ControllerRunResult]


@dataclass(frozen=True, slots=True)
class ControllerComparison:
    """Comparable metrics for one controller run."""

    controller_name: str
    average_waiting_time_seconds: float
    maximum_waiting_time_seconds: float
    throughput_vehicles_per_second: float
    completed_vehicles: int
    estimated_fuel_consumption_liters: float
    estimated_co2_emissions_kg: float
    emergency_vehicle_delay_seconds: float


@dataclass(frozen=True, slots=True)
class QAOAObjectiveComparison:
    """One measured QAOA objective alongside its exact classical objective."""

    simulation_step: int
    intersection_id: str
    qaoa_objective_value: float
    classical_objective_value: float
    matches_classical: bool
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class ExperimentAnalysis:
    """Derived, controller-comparable views of an experiment result."""

    controller_metrics: tuple[ControllerComparison, ...]
    qaoa_objectives: tuple[QAOAObjectiveComparison, ...]


def analyze_experiment(result: ExperimentResult) -> ExperimentAnalysis:
    """Build comparable metrics without changing the underlying run result."""

    controller_metrics = tuple(
        ControllerComparison(
            controller_name=name,
            average_waiting_time_seconds=controller_result.metrics.average_waiting_time_seconds,
            maximum_waiting_time_seconds=controller_result.metrics.maximum_waiting_time_seconds,
            throughput_vehicles_per_second=controller_result.metrics.throughput_vehicles_per_second,
            completed_vehicles=controller_result.metrics.total_vehicles_completed,
            estimated_fuel_consumption_liters=controller_result.metrics.estimated_fuel_consumption_liters,
            estimated_co2_emissions_kg=controller_result.metrics.estimated_co2_emissions_kg,
            emergency_vehicle_delay_seconds=controller_result.metrics.average_emergency_vehicle_delay_seconds,
        )
        for name, controller_result in result.controller_results.items()
    )
    qaoa_objectives = tuple(
        QAOAObjectiveComparison(
            simulation_step=step,
            intersection_id=intersection_id,
            qaoa_objective_value=comparison.qaoa_objective_value,
            classical_objective_value=comparison.classical_objective_value,
            matches_classical=comparison.qaoa_matches_classical,
            used_fallback=comparison.used_fallback,
        )
        for controller_result in result.controller_results.values()
        if controller_result.controller_name == "quantum_hybrid"
        for step, comparisons in enumerate(controller_result.qaoa_comparisons)
        for intersection_id, comparison in sorted(comparisons.items())
        if comparison.qaoa_objective_value is not None
    )
    return ExperimentAnalysis(controller_metrics, qaoa_objectives)


def run_experiment(
    scenario: TrafficScenario,
    *,
    controller_config: Mapping[str, Mapping[str, object]] | None = None,
) -> ExperimentResult:
    """Run fixed-time, adaptive, and quantum-hybrid controllers fairly."""

    configuration = {
        name: dict(settings)
        for name, settings in (controller_config or scenario.controller_config).items()
    }
    default_movement_road_map = {
        "north_south": (),
        "east_west": tuple(
            sorted(
                attributes["road_id"]
                for _, _, attributes in scenario.graph.edges(data=True)
            )
        )
    }
    for controller_name in ("adaptive", "quantum_hybrid"):
        configuration.setdefault(controller_name, {}).setdefault(
            "movement_road_map", default_movement_road_map
        )
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
        events=isolated.events,
    )
    simulator.add_vehicles(isolated.initial_vehicles)
    steps = int(isolated.duration_seconds / isolated.time_step_seconds)
    decisions: list[tuple[SignalDecision, ...]] = []
    queues: list[Mapping[str, int]] = []
    waiting: list[Mapping[str, float]] = []
    comparisons: list[Mapping[str, QuantumHybridComparison]] = []
    corridor_history: list[EmergencyCorridorResult] = []
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
        corridor_history.append(simulator.corridor_manager.last_result)
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
        metrics=simulator.metrics,
        corridor_history=tuple(corridor_history),
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