#!/usr/bin/env python3
"""
Thin adapter for GyroHead implementation.

This module provides a PyTorch interface to the GyroHead implementation,
which handles all the physics-based operations internally.
"""

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

class GyroTransformer(torch.nn.Module):
    """Unified Gyro physics transformer delegating all operations to GyroHead."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        from pathlib import Path
        from kernel.gyro_head import GyroHead
        base_mem_path = Path(__file__).parents[3] / "memories"
        self.gyro = GyroHead(base_path=base_mem_path)
    def forward(self, input_ids: torch.Tensor, past_key_values=None) -> torch.Tensor:
        """Forward pass returning logits using GyroHead's forward_pass method."""
        # Delegate to GyroHead's forward_pass which returns logits and KV cache
        logits, _ = self.gyro.forward_pass(input_ids, past_key_values)
        return logits