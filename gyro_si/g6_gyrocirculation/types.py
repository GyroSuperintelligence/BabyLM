"""Core type definitions for GyroSI system."""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch

# Tensor types
Tensor = Union[np.ndarray, torch.Tensor]
TensorShape = Tuple[int, ...]
TensorValue = Union[int, float]

# Memory types
MemoryType = str
MemoryKey = str
MemoryValue = Union[Tensor, Dict, List, str, int, float]
MemoryDict = Dict[MemoryKey, MemoryValue]

# Stage types
StageName = str
StageLevel = int
StageConfig = Dict[str, Union[str, int, float, bool, List, Dict]]

# Quantization types
PhaseValue = float
QuantizationError = float
QuantizationBin = int

# Algedonic types
SignalType = str
SignalValue = float
SignalConfig = Dict[str, Union[str, float, int]]

# System types
SystemLevel = str
SystemConfig = Dict[str, Union[str, int, float, bool, List, Dict]]

# Lineage types
LineageId = str
LineageTrace = List[Tuple[SystemLevel, StageName, MemoryKey]]
LineageConfig = Dict[str, Union[str, List, Dict]]

# Checksum types
ChecksumValue = str
ChecksumConfig = Dict[str, Union[str, int, bool]]

# Export types
ExportFormat = str
ExportConfig = Dict[str, Union[str, bool, List, Dict]]

# Import types
ImportFormat = str
ImportConfig = Dict[str, Union[str, bool, List, Dict]]

# UI types
UIComponent = str
UIConfig = Dict[str, Union[str, int, float, bool, List, Dict]]
UIEvent = Dict[str, Union[str, int, float, bool, List, Dict]] 