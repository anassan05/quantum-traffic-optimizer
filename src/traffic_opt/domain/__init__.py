"""Dependency-free domain models for the traffic optimization platform."""

from .enums import ControllerType, SignalPhase, TrafficEventType, VehicleType
from .models import (
    Intersection,
    RoadSegment,
    SignalPhaseConfig,
    SignalState,
    SimulationState,
    TrafficEvent,
    Vehicle,
)

__all__ = [
    "ControllerType",
    "Intersection",
    "RoadSegment",
    "SignalPhase",
    "SignalPhaseConfig",
    "SignalState",
    "SimulationState",
    "TrafficEvent",
    "TrafficEventType",
    "Vehicle",
    "VehicleType",
]