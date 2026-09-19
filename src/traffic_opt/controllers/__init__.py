"""Signal-controller interfaces used by later emergency coordination."""

from .signal_controller import (
    EmergencyPriorityStatus,
    PriorityResult,
    SignalController,
)

__all__ = [
    "EmergencyPriorityStatus",
    "PriorityResult",
    "SignalController",
]