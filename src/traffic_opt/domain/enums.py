"""Enumerations used by the traffic domain model."""

from enum import Enum


class SignalPhase(str, Enum):
    """High-level signal phases supported by the initial domain model."""

    NORTH_SOUTH = "north_south"
    EAST_WEST = "east_west"
    ALL_RED = "all_red"


class VehicleType(str, Enum):
    """Vehicle categories used for demand and emergency handling."""

    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    AMBULANCE = "ambulance"


class TrafficEventType(str, Enum):
    """Events that can alter simulated road conditions."""

    CONGESTION = "congestion"
    ROAD_CLOSURE = "road_closure"
    ACCIDENT = "accident"
    SPECIAL_DEMAND = "special_demand"


class ControllerType(str, Enum):
    """Traffic controller strategies available to later phases."""

    FIXED_TIME = "fixed_time"
    ADAPTIVE_RULE_BASED = "adaptive_rule_based"
    QUANTUM_HYBRID = "quantum_hybrid"