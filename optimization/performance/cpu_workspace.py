"""Compatibility import for the CPU workspace promoted into production.

Use optimization.reconstruction.CPUReconstructionWorkspace for new code.
"""
from optimization._reconstruction_cpu import CPUReconstructionWorkspace

__all__ = ["CPUReconstructionWorkspace"]
