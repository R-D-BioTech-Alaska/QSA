"""Compatibility import for projects written before the canonical ``qsa`` package.

New code should use ``from qsa import QubitRegister``. Existing code importing
``qubit_native`` continues to receive the same class and behavior.
"""

from qsa import (
    GroverSearch,
    OperationPlan,
    Parameter,
    ParameterizedPlan,
    QRegister,
    QubitNativeError,
    QubitRegister,
    SymmetryState,
    __version__,
)

__all__ = [
    "GroverSearch",
    "OperationPlan",
    "Parameter",
    "ParameterizedPlan",
    "QRegister",
    "QubitNativeError",
    "QubitRegister",
    "SymmetryState",
    "__version__",
]
