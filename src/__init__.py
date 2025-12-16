"""
Complexity Physics Framework
============================

A unified framework where spacetime geometry, quantum mechanics, gauge structure,
fermion generations, and arithmetic regularities emerge as stable stationary
points of a constrained complexity functional.

Central hypothesis: δC = 0 where C = R + K + B
    R = Retrodiction complexity (geometric)
    K = Representation complexity (algebraic)
    B = Barrier complexity (topological)

Modules
-------
complexity : Core complexity functional implementation
gauge_theory : Gauge group complexity calculations
fermion_sector : Cl(6) Clifford algebra mass hierarchy
helicity_barrier : Solar wind turbulence model
riemann_zeta : Number theory analysis and zeta function
validation : Empirical validation tools

Example
-------
>>> from src.complexity import TotalComplexity
>>> from src.gauge_theory import GaugeGroupComplexity
>>>
>>> # Calculate Standard Model gauge complexity
>>> sm = GaugeGroupComplexity("SU3xSU2xU1")
>>> print(f"K(SM) = {sm.compute():.2f}")
"""

__version__ = "0.1.0"
__author__ = "Dylan Roy"
__license__ = "MIT"

from .complexity import TotalComplexity, RetrodictionComplexity, BarrierComplexity
from .gauge_theory import GaugeGroupComplexity, RepresentationComplexity
from .fermion_sector import Cl6MassHierarchy, FermionGeneration
from .helicity_barrier import HelicityBarrier, TurbulenceModel
from .riemann_zeta import RiemannZeta, SpectralAnalysis
from .validation import ValidationSuite, EmpiricalData

__all__ = [
    # Complexity
    "TotalComplexity",
    "RetrodictionComplexity",
    "BarrierComplexity",
    # Gauge theory
    "GaugeGroupComplexity",
    "RepresentationComplexity",
    # Fermion sector
    "Cl6MassHierarchy",
    "FermionGeneration",
    # Helicity barrier
    "HelicityBarrier",
    "TurbulenceModel",
    # Riemann zeta
    "RiemannZeta",
    "SpectralAnalysis",
    # Validation
    "ValidationSuite",
    "EmpiricalData",
]
