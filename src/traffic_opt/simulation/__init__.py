"""Discrete-time traffic simulation primitives."""

from .simulator import TrafficSimulator
from .state import IntersectionSignalState, SimulationState
from .vehicles import VehicleState, generate_vehicles

__all__ = [
    "IntersectionSignalState",
    "SimulationState",
    "TrafficSimulator",
    "VehicleState",
    "generate_vehicles",
]