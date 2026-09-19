"""Discrete-time traffic simulation primitives."""

from .events import EffectiveRoadCondition, EventImpactConfig, effective_road_conditions
from .simulator import TrafficSimulator
from .state import IntersectionSignalState, SimulationState
from .vehicles import VehicleState, generate_vehicles

__all__ = [
    "IntersectionSignalState",
    "EffectiveRoadCondition",
    "EventImpactConfig",
    "SimulationState",
    "TrafficSimulator",
    "VehicleState",
    "generate_vehicles",
    "effective_road_conditions",
]