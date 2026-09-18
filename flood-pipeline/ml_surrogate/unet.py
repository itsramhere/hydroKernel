"""
2D U-Net Neural Network Architecture Specification for Flood Inundation (Phase 3).

Surrogate Model Interface:
- Input: 3 Channels (Normalized DEM, Slope Gradient, 6-hr Precipitation Depth).
- Output: 1 Channel (Continuous Predicted Water Surface Depth in meters).
- Physics-Informed Kernels: Convolutions calibrated to capture gravity-driven flow accumulation
  and topographic depressions.
"""

from typing import Tuple


class UNetArchitectureSpec:
    """Defines layer dimensions, channel mappings, and physical parameterizations."""

    INPUT_CHANNELS = 3    # Ch0: DEM, Ch1: Slope, Ch2: Precip (mm)
    OUTPUT_CHANNELS = 1   # Ch0: Inundation depth h (meters)

    # Physical scaling parameters
    MIN_DAMAGE_DEPTH_M = 0.15   # 15 cm crop/property damage threshold
    SURCHARGE_FACTOR = 0.004    # mm rain to ponded water depth conversion

    @classmethod
    def get_input_shape(cls, height: int = 256, width: int = 256) -> Tuple[int, int, int, int]:
        return (1, cls.INPUT_CHANNELS, height, width)
