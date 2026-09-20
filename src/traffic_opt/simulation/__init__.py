"""Discrete-time traffic simulation primitives."""

from .events import EventManager
from .metrics import MetricsCollector, SimulationMetrics
from .simulator import TrafficSimulator
from .state import IntersectionSignalState, SimulationState
from .vehicles import VehicleState, create_emergency_vehicle, generate_vehicles

__all__ = [
    "EventManager",
    "MetricsCollector",
    "SimulationMetrics",
    "create_emergency_vehicle",
    "IntersectionSignalState",
    "SimulationState",
    "TrafficSimulator",
    "VehicleState",
    "generate_vehicles",
]