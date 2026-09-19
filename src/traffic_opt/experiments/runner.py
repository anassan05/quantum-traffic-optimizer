"""Run comparable controller experiments over identical scenarios."""

from dataclasses import dataclass
from typing import Mapping

from traffic_opt.controllers.adaptive import AdaptiveController
from traffic_opt.controllers.base import SignalController, SignalDecision
from traffic_opt.controllers.fixed_time import FixedTimeController
from traffic_opt.controllers.quantum_hybrid import QuantumHybridComparison, QuantumHybridController
from traffic_opt.demand import (
    DemandScenarioConfig,
    TrafficDemandGenerator,
    TrafficScenario as DemandTrafficScenario,
    VehicleDemand,
)
from traffic_opt.domain.enums import ControllerType, VehicleType
from traffic_opt.emergency.route_planner import AmbulanceRoutePlanner
from traffic_opt.events import TrafficEventConfig, TrafficEventManager
from traffic_opt.metrics import MetricsCollector, MetricsSnapshot
from traffic_opt.simulation.vehicles import DEFAULT_SPEED_KMH, VehicleState
from traffic_opt.network.intersections import create_default_intersections
from traffic_opt.network.topology import create_default_topology
from traffic_opt.simulation.simulator import TrafficSimulator
from traffic_opt.simulation.state import SimulationState

from .scenarios import TrafficScenario, clone_scenario
from .config import ExperimentConfig
from .results import ComparisonRow, ScenarioExperimentResult, compare_results


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


class ExperimentRunner:
    """Orchestrate existing demand, events, simulation, controllers, and metrics."""

    def __init__(self, topology=None) -> None:
        self.topology = create_default_topology() if topology is None else topology

    def run(self, configuration: ExperimentConfig) -> ScenarioExperimentResult:
        """Run one configured scenario without replacing the simulator."""

        if not self.topology.nodes:
            raise ValueError("experiment topology must contain intersections")
        if not any(self.topology.edges):
            raise ValueError("experiment topology must contain road segments")

        demand_config = configuration.demand_config or DemandScenarioConfig.for_scenario(
            DemandTrafficScenario.NORMAL,
            random_seed=configuration.seed,
            simulation_horizon=configuration.simulation_horizon_seconds,
        )
        event_config = configuration.event_config or TrafficEventConfig(
            random_seed=configuration.seed,
            simulation_horizon=configuration.simulation_horizon_seconds,
            event_count=0,
        )
        demand = TrafficDemandGenerator(
            demand_config,
            self.topology,
            seed=configuration.seed,
        ).generate()
        events = TrafficEventManager(
            self.topology,
            event_config,
            seed=configuration.seed,
        ).generate()
        intersections = create_default_intersections(self.topology)
        vehicles = self._vehicles_for_demand(demand)
        controller = self._create_controller(configuration)
        simulator = TrafficSimulator(
            self.topology,
            intersections,
            time_step_seconds=configuration.time_step_seconds,
            seed=configuration.seed,
        )
        simulator.add_vehicles(vehicles)
        collector = MetricsCollector(
            self.topology,
            spawn_times={item.vehicle_id: item.spawn_time for item in demand},
        )
        collector.collect(simulator.state)
        steps = int(
            configuration.simulation_horizon_seconds / configuration.time_step_seconds
        )
        for _ in range(steps):
            decisions = controller.decide(simulator.state)
            _apply_decisions(simulator, decisions)
            simulator.step()
            collector.collect(simulator.state)

        snapshots = collector.snapshots
        final = snapshots[-1]
        return ScenarioExperimentResult(
            experiment_name=configuration.experiment_name,
            controller_name=ControllerType(configuration.controller).value,
            scenario=demand_config.scenario.value,
            seed=configuration.seed,
            simulation_duration_seconds=configuration.simulation_horizon_seconds,
            demands=demand,
            events=events,
            metrics=collector.aggregate(),
            snapshots=snapshots,
            ambulance_count=final.ambulance_count,
            completed_ambulances=final.completed_ambulances,
            average_ambulance_travel_time_seconds=(
                final.average_ambulance_travel_time_seconds
            ),
            average_ambulance_waiting_time_seconds=(
                final.average_ambulance_waiting_time_seconds
            ),
        )

    def _vehicles_for_demand(self, demand: tuple[VehicleDemand, ...]) -> tuple[VehicleState, ...]:
        planner = AmbulanceRoutePlanner(self.topology)
        vehicles = []
        for item in demand:
            route = planner.plan(
                item.origin_intersection_id,
                item.destination_intersection_id,
            )
            vehicles.append(
                VehicleState(
                    id=item.vehicle_id,
                    vehicle_type=item.vehicle_type,
                    origin_intersection_id=item.origin_intersection_id,
                    destination_intersection_id=item.destination_intersection_id,
                    route_road_ids=route,
                    current_road_id=route[0],
                    speed_kmh=DEFAULT_SPEED_KMH[item.vehicle_type],
                )
            )
        return tuple(vehicles)

    @staticmethod
    def _create_controller(configuration: ExperimentConfig) -> SignalController:
        controller_type = ControllerType(configuration.controller)
        parameters = dict(configuration.controller_config)
        if controller_type is ControllerType.FIXED_TIME:
            return FixedTimeController(**parameters)
        if controller_type is ControllerType.ADAPTIVE_RULE_BASED:
            return AdaptiveController(**parameters)
        if controller_type is ControllerType.QUANTUM_HYBRID:
            return QuantumHybridController(**parameters)
        raise ValueError(f"unsupported controller: {controller_type.value}")