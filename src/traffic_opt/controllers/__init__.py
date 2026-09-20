"""Traffic signal controller interfaces and classical strategies."""

from .emergency_corridor import (
	EmergencyCorridorManager,
	EmergencyCorridorResult,
	EmergencyCorridorStatus,
)
from .base import SignalController, SignalDecision
from .adaptive import AdaptiveController
from .fixed_time import FixedTimeController
from .quantum_hybrid import QuantumHybridComparison, QuantumHybridController

__all__ = [
	"EmergencyCorridorManager",
	"EmergencyCorridorResult",
	"EmergencyCorridorStatus",
	"AdaptiveController",
	"FixedTimeController",
	"QuantumHybridComparison",
	"QuantumHybridController",
	"SignalController",
	"SignalDecision",
]