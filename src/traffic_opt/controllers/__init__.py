"""Traffic signal controller interfaces and classical strategies."""

from .emergency_corridor import (
	EmergencyCorridorManager,
	EmergencyCorridorResult,
	EmergencyCorridorStatus,
)
from .base import SignalController as SignalControllerProtocol, SignalDecision
from .adaptive import AdaptiveController
from .fixed_time import FixedTimeController
from .quantum_hybrid import QuantumHybridComparison, QuantumHybridController
from .signal_controller import EmergencyPriorityStatus, PriorityResult, SignalController

__all__ = [
	"EmergencyCorridorManager",
	"EmergencyCorridorResult",
	"EmergencyCorridorStatus",
	"EmergencyPriorityStatus",
	"AdaptiveController",
	"FixedTimeController",
	"PriorityResult",
	"QuantumHybridComparison",
	"QuantumHybridController",
	"SignalController",
	"SignalControllerProtocol",
	"SignalDecision",
]