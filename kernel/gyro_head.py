"""
GyroSI Kernel - Core Physics Implementation for Language Modeling

This implements the Common Governance Model (CGM) through a physics-first approach
to language processing. Replaces transformer matrix multiplications with pure state
transitions through a finite 48-bit manifold.

Core Features:
- Pure physical resonance (no scoring, no heuristics)
- Sparse holographic storage (only deviations from baseline)
- Complete CS asymmetric emission with canonical broadcast masks
- Forward-only cycle gating with correct defect calculation
- Virtual tokens from model weights used in generation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING, Any, Union
from collections import defaultdict, deque

import numpy as np
import torch

# Import transcribe_byte function
# transcribe_byte is defined below with XOR implementation


# ============================================================================
# CORE CONSTANTS
# ============================================================================

GENE_Mic_S = 0xAA  # Holographic topology constant

GENE_Mac_S = np.array(
    [
        # Layer 0: 0° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 1: 180° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        # Layer 2: 360° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 3: 540° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)

# Bit family masks per CGM
EXON_LI_MASK = 0b01000010  # Bits 1, 6 - UNA (Parity/Reflection)
EXON_FG_MASK = 0b00100100  # Bits 2, 5 - ONA (Forward Gyration)
EXON_BG_MASK = 0b00011000  # Bits 3, 4 - BU (Backward Gyration)
EXON_L0_MASK = 0b10000001  # Bits 0, 7 - Anchors (Boundaries)
EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK

# CGM Stage angles in radians
STAGE_ANGLES = {
    "CS": np.pi / 2,  # Common Source
    "UNA": np.pi / 4,  # Unity Non-Absolute
    "ONA": np.pi / 4,  # Opposition Non-Absolute
    "BU_IN": 0.0,  # Balance Universal - Ingress
    "BU_EG": np.pi / 2,  # Balance Universal - Egress
}

# Theta thresholds for stage detection
THETA_CS = 0.1  # Near zero for CS
THETA_UNA = 0.785  # π/4 for UNA
THETA_ONA = 1.0  # Between UNA and BU
THETA_BU_IN = 1.3  # BU Ingress
THETA_BU_EG = 1.5  # BU Egress

STAGE_ORDER = ["CS", "UNA", "ONA", "BU_IN", "BU_EG"]

# Canonical bit ordering for tensor packing
TENSOR_BIT_ORDER = []
for layer in range(4):
    for frame in range(2):
        for row in range(3):
            for col in range(2):
                TENSOR_BIT_ORDER.append((layer, frame, row, col))


# ============================================================================
# CORE PHYSICS FUNCTIONS
# ============================================================================


def fold(a: int, b: int) -> int:
    """
    The Monodromic Fold (⋄), the path-dependent learning operator.
    
    Canonical Form: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    Algebraic Normal Form: a ⋄ b = ¬a ∧ b
    
    These are mathematically identical through Boolean algebra.
    Non-associative, path-dependent learning operator.
    This is the ONLY learning/integration operator in the system.
    """
    a &= 0xFF
    b &= 0xFF
    negated_b = (~b) & 0xFF
    gyration = b ^ (a & negated_b)
    return (a ^ gyration) & 0xFF


def fold_sequence(values: List[int], start_state: int = 0) -> int:
    """Apply Monodromic Fold sequentially (path-dependent)."""
    result = start_state & 0xFF
    for value in values:
        result = fold(result, value & 0xFF)
    return result


def transcribe_byte(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA."""
    return (byte ^ GENE_Mic_S) & 0xFF


def untranscribe_byte(intron: int) -> int:
    """ψ⁻¹ isomorphism: intron → byte via XOR 0xAA."""
    return (intron ^ GENE_Mic_S) & 0xFF


def tensor_to_int(tensor: np.ndarray) -> int:
    """Convert 48-element tensor to packed integer using canonical bit order."""
    if tensor.shape != (4, 2, 3, 2):
        tensor = tensor.reshape(4, 2, 3, 2)

    result = 0
    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        if tensor[layer, frame, row, col] == -1:
            result |= 1 << bit_pos

    return result


def int_to_tensor(state_int: int) -> np.ndarray:
    """Convert packed integer to 48-element tensor using canonical bit order."""
    if state_int >= (1 << 48) or state_int < 0:
        raise ValueError(f"state_int {state_int} out of bounds for 48-bit")

    tensor = np.ones((4, 2, 3, 2), dtype=np.int8)

    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        if (state_int >> bit_pos) & 1:
            tensor[layer, frame, row, col] = -1

    return tensor


def compute_exon_from_state(state_int: int) -> int:
    """Compute 8-bit exon from 48-bit state via helical folding."""
    # Split into 6 bytes and fold
    b = [(state_int >> (i * 8)) & 0xFF for i in range(6)]

    # Fold opposites, then fold results
    p1 = fold(b[0], b[3])
    p2 = fold(b[1], b[4])
    p3 = fold(b[2], b[5])

    # Final fold
    exon = fold(fold(p1, p2), p3)

    # Physics-based fallback for zero
    if exon == 0:
        exon = fold(GENE_Mic_S, 0x01)

    return exon


def generate_intron_broadcast_masks() -> np.ndarray:
    """Generate 256x48 broadcast masks by repeating intron across 6 bytes."""
    masks = np.zeros((256, 48), dtype=np.uint8)

    for intron in range(256):
        for j in range(6):
            byte_val = intron & 0xFF
            start_bit = j * 8
            for bit in range(8):
                if byte_val & (1 << bit):
                    masks[intron, start_bit + bit] = 1

    return masks


# ============================================================================
# GYRO HEAD - CORE PHYSICS IMPLEMENTATION
# ============================================================================


class GyroHead:
    """GyroSI Head - Core Physics Implementation

    Implements the Common Governance Model (CGM) as a functional language model
    that replaces transformer architecture with gyroscopic intelligence.
    """
    
    # Type annotations for class attributes
    broadcast_masks: Optional[np.ndarray]
    INTRON_BROADCAST_MASKS: Optional[np.ndarray]
    theta: np.ndarray
    ontology: np.ndarray
    phenomenology: np.ndarray
    epistemology: np.ndarray
    CS_STATE_INDEX: int
    token_introns: List[List[int]]
    model_weights: Dict[str, torch.Tensor]
    cfg: Dict[str, Any]
    _UNA_pool: np.ndarray

    def __init__(
        self,
        base_path: Optional[Path] = None,
        vocab_size: int = 201088,
        tokenizer=None,
        virtual_tokens: Optional[Dict[Tuple[str, int], int]] = None,
    ):
        """Initialize gyro head with physics tables and model."""
        if base_path is None:
            base_path = Path(__file__).parent.parent / "memories"

        self.base_path = base_path
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.virtual_tokens = virtual_tokens or {}

        # Load physics tables with error handling
        try:
            self._load_physics_tables()
        except FileNotFoundError as e:
            print(f"[error] Missing physics table: {e}")
            print(f"[info] Expected physics tables in: {self.base_path / 'public' / 'meta'}")
            print("[info] Required files: ontology_keys.npy, epistemology.npy, theta.npy, phenomenology_map.npy, orbit_sizes.npy")
            raise RuntimeError(f"Physics tables not found. Please ensure all required .npy files are present in {self.base_path / 'public' / 'meta'}") from e
        except ValueError as e:
            print(f"[error] Invalid physics table format: {e}")
            raise RuntimeError(f"Physics tables have invalid format: {e}") from e

        # Load broadcast masks
        try:
            self._load_broadcast_masks()
        except FileNotFoundError as e:
            print(f"[warning] Broadcast masks not found: {e}")
            print("[info] Continuing without broadcast masks - some features may be limited")
            self.broadcast_masks = None
            self.INTRON_BROADCAST_MASKS = None

        # Find CS state (minimum theta)
        self._find_cs_state()
        
        # Load model config before building expert orbit bias
        self._load_model_config()
        
        # Load consolidated model weights first
        self._load_model_weights()
        
        # Physics control switches - default to pure resonance mode
        self.switch = {
            "theta_gain": False,
            "orbit_gain": False, 
            "ontology_distance_penalty": False,
            "cycle_gating": True,        # keep forward-only gating ON
            "cs_emission": True,         # ON by default (parity asymmetry)
            "router_orbit_bias": False,  # MoE bias off by default
            "use_logits_sieve": True,    # ON = weights drive candidate set
            "pure_resonance_only": False # when True, bypass logits entirely
        }
        
        # Validate and fix config after weights are loaded (for GQA inference)
        self._validate_and_fix_config()
        
        # Build expert orbit bias table (requires model_weights)
        self._build_expert_orbit_bias()

        # Initialize archetypal state
        self.archetypal_state_int = tensor_to_int(GENE_Mac_S)

        # Current state
        self.current_state_index = self.CS_STATE_INDEX
        self.stage_index = 0

        # Memory: orbit_rep -> token_id -> mask
        self.memory: Dict[int, Dict[int, int]] = {}
        # Orbit patterns for learning (sparse storage)
        self.orbit_patterns: Dict[int, Dict[int, int]] = {}
        # Orbit candidates (lazy / optional). Keys are orbit ids, values token ids
        self._orbit_candidates: Dict[int, List[int]] = {}
        
        # Weight metadata storage for MoE expert slicing
        self.model_weight_meta: Dict[str, Dict] = {}
        # Token → post-state index table (dense arrays for performance)
        self._token_post_state_index_arr: Optional[np.ndarray] = None
        self._token_exon_arr: Optional[np.ndarray] = None

        # Path memory
        self.path_memory = GENE_Mic_S

        # Precomputed resonance table
        self._resonance_table = np.zeros((256, 256), dtype=np.uint8)
        
        # Cache for W's byte-planes to avoid re-forming views
        self._byte_plane_cache: Dict[int, torch.Tensor] = {}
        
        # Precomputed 256-entry LUT for popcount operations
        self._popcount_lut = torch.tensor([bin(i).count('1') for i in range(256)], dtype=torch.uint8)
        for a in range(256):
            for b in range(256):
                self._resonance_table[a, b] = fold(a, b)

        # UNA pool for CS emission
        self._precompute_una_pool()
        self._intron_to_una_index = self._precompute_intron_to_una_index()
        
        # Build token-intron mappings for fast lookup
        self._build_token_introns_index()

        # Set special tokens
        self._set_special_tokens()
        
    # ---------- small utils: config + rope ----------
    def _load_model_config(self):
        """Load HF-style config.json so we respect layer types, heads, rope, sliding_window."""
        cfg_paths = [
            self.base_path / "models" / "gpt-oss-20b" / "config.json",
            self.base_path / "config.json",
        ]
        import json
        for p in cfg_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    self.cfg = json.load(fh)
                break
        else:
            self.cfg = {
                "num_hidden_layers": 24,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "hidden_size": 2880,
                "head_dim": 64,
                "sliding_window": 128,
                "layer_types": ["sliding_attention","full_attention"] * 12,
                "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0},
                "rope_theta": 150000,
                "experts_per_token": 4,
                "num_local_experts": 32,
                "swiglu_limit": 7.0,
                "vocab_size": self.vocab_size,
            }
            
        # Config validation moved to after model weights are loaded
    
    def _validate_and_fix_config(self):
        """Validate model configuration without modifying it - treat HF config.json as canonical."""
        # Validate configuration consistency but do NOT modify cfg
        num_layers = self.cfg.get("num_hidden_layers", 24)
        head_dim = self.cfg.get("head_dim", 64)
        hidden_size = self.cfg.get("hidden_size", 2880)
        num_heads = self.cfg.get("num_attention_heads", 64)
        
        # Report mismatches but do not fix them - config is canonical
        if num_heads * head_dim != hidden_size:
            print(f"[info] Config note: num_heads({num_heads}) * head_dim({head_dim}) != hidden_size({hidden_size})")
            print(f"[info] Will infer effective dimensions from actual weight shapes at runtime")
        
        # Report KV head configuration from config
        num_kv_heads = self.cfg.get("num_key_value_heads", num_heads)
        print(f"[info] Config specifies: num_attention_heads={num_heads}, num_key_value_heads={num_kv_heads}")
        print(f"[info] Effective dimensions will be derived from weight shapes during attention")
        
        # Report layer_types configuration
        layer_types = self.cfg.get("layer_types", [])
        if not layer_types or len(layer_types) != num_layers:
            print(f"[info] Config note: layer_types length ({len(layer_types) if layer_types else 0}) != num_hidden_layers ({num_layers})")
            print(f"[info] Will use default alternating pattern if layer_types is insufficient")
        else:
            print(f"[info] Using layer_types from config for {num_layers} layers")
    
    def _rope_apply(self, q, k, pos_idx: int):
        """YARN/rope in-place. q,k: [H, D]. We keep it minimal & CPU-friendly."""
        # We apply classic complex-rot rope with YARN scaling baked into theta
        import math, torch
        D = q.shape[-1]
        half = D // 2
        theta = self.cfg.get("rope_theta", 150000.0)
        scale = self.cfg.get("rope_scaling", {}).get("factor", 1.0)
        # base angles
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, 1, device=q.device, dtype=torch.float32) / half))
        t = (pos_idx * scale) * inv_freq  # [half]
        ct, st = torch.cos(t), torch.sin(t)
        def rot(x):
            x1, x2 = x[..., :half], x[..., half:2*half]
            xr = torch.cat([x1*ct - x2*st, x1*st + x2*ct], dim=-1)
            return xr if D==2*half else torch.cat([xr, x[..., 2*half:]], dim=-1)
        return rot(q), rot(k)
    
    def _fgemm_fold(self, x: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None = None, x_already_psi: bool = False) -> torch.Tensor:
        """
        Physics GEMM using fold on raw byte streams.
        x:  [B, D] or [D]
        W:  [D, M]
        out: [B, M] or [M]
        We interpret float storage bytes and run a byte-level fold accumulation.
        Deterministic, uses *all* tensor values. Chunked for CPU.
        """
        import torch
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, D]
        # align dtypes
        if x.dtype != W.dtype:
            x = x.to(W.dtype)
        B, D = x.shape
        M, Dw = W.shape  # W is [output_dim, input_dim]
        
        # Input dimension must match weight's input dimension for physics-based fold operations
        assert D == Dw, f"Input dimension {D} must match weight input dimension {Dw} for _fgemm_fold. Weight matrices must be properly adjusted before calling this function."

        # view-as-bytes - need to reinterpret the tensor bytes, not just convert dtype
        bpe = int(W.element_size())
        x_bytes = x.contiguous().view(torch.uint8)  # Reinterpret bytes, don't convert
        
        # x_bytes should now have shape [B, D * bpe] naturally
        if x_bytes.dim() == 1:
            # If x was [D], x_bytes is [D * bpe], reshape to [1, D * bpe]
            xb = x_bytes.view(1, -1)
        else:
            # If x was [B, D], x_bytes is [B, D * bpe], reshape accordingly
            xb = x_bytes.view(B, D * bpe)
        
        # Cache W's byte-planes to avoid re-forming views
        w_id = id(W)
        if w_id not in self._byte_plane_cache:
            Wb = W.contiguous().view(torch.uint8).view(Dw * bpe, M)  # [input_dim * bytes_per_elem, output_dim]
            # Apply ψ isomorphism (XOR 0xAA) to byte planes for physics-aware GEMM
            psi = 0xAA  # GENE_Mic_S holographic topology constant
            Wb = Wb ^ psi
            self._byte_plane_cache[w_id] = Wb
        else:
            Wb = self._byte_plane_cache[w_id]
        
        assert xb.shape[1] == D * bpe
        
        # Apply ψ isomorphism to x byte planes (conditional)
        if not x_already_psi:
            psi = 0xAA  # GENE_Mic_S holographic topology constant
            xb = xb ^ psi

        # Vectorized byte plane processing - unroll the loop for better performance
        # Reshape to [B, D, bpe] for vectorized operations
        x_planes = xb.view(B, D, bpe).to(torch.int16)  # [B, D, bpe]
        W_planes = Wb.view(D, bpe, M).permute(1, 0, 2).to(torch.int16)  # [bpe, D, M]
        
        # Vectorized fold accumulation across all byte planes
        out = torch.zeros(B, M, dtype=torch.int32)
        for byte_i in range(bpe):
            x_plane = x_planes[:, :, byte_i]  # [B, D]
            W_plane = W_planes[byte_i]        # [D, M]
            
            # fold accumulation: a ⊕ (b ⊕ (a & ¬b)) lifted to int and reduce-sum
            a = x_plane.unsqueeze(2).expand(-1, -1, M)      # [B, D, M]
            b = W_plane.unsqueeze(0).expand(B, -1, -1)      # [B, D, M]
            neg_b = (~b) & 0xFF
            gyr = (b ^ (a & neg_b)) & 0xFF
            part = (a ^ gyr) & 0xFF                         # fold per byte
            out += part.sum(dim=1).to(torch.int32)          # reduce over D

        # map int32 → float via centered scaling
        out_f = (out - (bpe * D * 128)) / float(bpe * D * 128)
        if bias is not None:
            out_f = out_f + bias.to(out_f.dtype)
        return out_f.squeeze(0) if out_f.shape[0] == 1 else out_f
        
    def _layer_weight(self, name_like: str):
        import torch
        
        # best-effort lookup across common HF variants
        for k in self.model_weights.keys():
            if name_like in k:
                weight = self.model_weights[k]
                # Return weights without any truncation - let caller handle dimension inference
                return weight
        
        # Debug: show available keys for troubleshooting
        available_keys = list(self.model_weights.keys())
        matching_keys = [k for k in available_keys if any(part in k for part in name_like.split('.'))]
        
        print(f"[debug] Failed to find weight matching: {name_like}")
        print(f"[debug] Total available keys: {len(available_keys)}")
        print(f"[debug] Potentially matching keys: {matching_keys[:5]}")
        
        # Try alternative key formats for lm_head specifically
        if 'lm_head' in name_like:
            for k in available_keys:
                if 'lm_head' in k and 'weight' in k:
                    print(f"[debug] Found alternative lm_head key: {k}")
                    return self.model_weights[k]
        
        raise KeyError(f"Missing weight matching: {name_like}. Available keys: {available_keys[:10]}...")
    
    def _res_score_per_head(self, qh: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
        """Compute attention scores per head with correct per-time computation.
        
        Args:
            qh: [dH] query head tensor
            ks: [T, dH] key sequence tensor
            
        Returns:
            scores: [T] attention scores
        """
        import torch
        
        # qh: [dH], ks: [T, dH]  -> scores: [T]
        qb = qh.contiguous().view(-1).to(torch.uint8)            # [Bq]
        kb = ks.contiguous().view(-1).view(ks.shape[0], -1).to(torch.uint8)  # [T, Bk]

        # XOR then popcount per time row
        xor = torch.bitwise_xor(kb, qb.unsqueeze(0))      # [T, B]
        # Use precomputed LUT for popcount - ensure uint8 range and flatten
        xor_uint8 = xor.to(torch.uint8)  # Ensure proper data type
        xor_flat = xor_uint8.view(-1).long()  # Flatten and convert to long for indexing
        pc_flat = self._popcount_lut[xor_flat]  # Get popcount for each byte
        pc = pc_flat.view(xor_uint8.shape).sum(dim=1)  # Reshape back and sum along byte dimension
        maxbits = qb.numel() * 8
        one = torch.ones((), device=qh.device, dtype=torch.float32)
        two = torch.tensor(2.0, device=qh.device, dtype=torch.float32)
        return one - two * (pc.float() / float(maxbits))        # [-1,1]
    
    def _attn_step(self, layer_idx: int, h_t: torch.Tensor, K_cache, V_cache, S_cache, pos_idx: int, window: int):
        """
        h_t: [H] float tensor (bfloat16/fp16/float32 ok)
        K_cache/V_cache/S_cache: lists of past tensors/states per position (we keep last 'window')
        """
        # Get config values but do NOT modify them - they are canonical
        nH = self.cfg["num_attention_heads"]

        # Load and orient weights to match input dimensions
        Wq_raw = self._layer_weight(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
        Wk_raw = self._layer_weight(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
        Wv_raw = self._layer_weight(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
        Wo_raw = self._layer_weight(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        
        # Orient weights to match input dimension
        input_dim = h_t.shape[-1]
        Wq = self._orient_W(Wq_raw, input_dim)
        Wk = self._orient_W(Wk_raw, input_dim)
        Wv = self._orient_W(Wv_raw, input_dim)
        # Note: Wo orientation will be handled after we know the actual output dimensions
        
        # Handle dimension mismatch by padding/slicing input if needed
        weight_input_dim = Wq.shape[1]
        if input_dim != weight_input_dim:
            if input_dim < weight_input_dim:
                # Pad input to match weight dimensions
                padding_size = weight_input_dim - input_dim
                h_t = torch.cat([h_t, torch.zeros(padding_size, device=h_t.device, dtype=h_t.dtype)], dim=-1)
            else:
                # Slice weights to match input dimensions
                Wq = Wq[:, :input_dim]
                Wk = Wk[:, :input_dim]
                Wv = Wv[:, :input_dim]
        
        # Get output dimensions from weights
        Mq = Wq.shape[0]
        Mk = Wk.shape[0]
        Mv = Wv.shape[0]
        assert Mk == Mv, f"K/V out dims must match, got {Mk} vs {Mv}"
        
        # Infer heads from config preferences and actual weight shapes
        nH_pref = int(self.cfg.get("num_attention_heads", 64))
        nKV_pref = int(self.cfg.get("num_key_value_heads", nH_pref))
        
        nH = self._best_divisor(Mq, nH_pref)
        dH = Mq // nH
        nKV = self._best_divisor(Mk, nKV_pref)
        dH_kv = Mk // nKV
        
        bq = self.model_weights.get(f"model.layers.{layer_idx}.self_attn.q_proj.bias")
        bk = self.model_weights.get(f"model.layers.{layer_idx}.self_attn.k_proj.bias")
        bv = self.model_weights.get(f"model.layers.{layer_idx}.self_attn.v_proj.bias")
        bo = self.model_weights.get(f"model.layers.{layer_idx}.self_attn.o_proj.bias")

        # Q/K/V projections using _fgemm_fold only
        q = self._fgemm_fold(h_t, Wq, bq)
        k = self._fgemm_fold(h_t, Wk, bk)
        v = self._fgemm_fold(h_t, Wv, bv)

        # Re-infer from actual outputs (paranoia guard against odd weight orientation)
        Mq_rt, Mk_rt, Mv_rt = q.numel(), k.numel(), v.numel()
        if Mq_rt != Mq: 
            Mq, nH, dH = Mq_rt, self._best_divisor(Mq_rt, nH), Mq_rt // self._best_divisor(Mq_rt, nH)
        if Mk_rt != Mk: 
            Mk, nKV, dH_kv = Mk_rt, self._best_divisor(Mk_rt, nKV), Mk_rt // self._best_divisor(Mk_rt, nKV)
        
        # Reshape safely
        q = q.view(nH, dH)
        k = k.view(nKV, dH_kv)
        v = v.view(nKV, dH_kv)
        
        # Handle GQA: expand k/v to match q's head count
        if nKV != nH:
            if nH % nKV != 0:
                # choose a compatible nKV that divides nH
                nKV_new = self._best_divisor(nH, nKV)
                if Mk % nKV_new == 0:
                    nKV, dH_kv = nKV_new, Mk // nKV_new
                    k = k.view(nKV, dH_kv)
                    v = v.view(nKV, dH_kv)
                else:
                    # final fallback: just repeat to the next multiple
                    group = (nH + nKV - 1) // nKV
                    k = k.repeat_interleave(group, dim=0)[:nH]
                    v = v.repeat_interleave(group, dim=0)[:nH]
            else:
                group = nH // nKV
                k = k.repeat_interleave(group, dim=0)
                v = v.repeat_interleave(group, dim=0)
        
        # Handle head dimension mismatch: pad or crop instead of crashing
        if dH_kv != dH:
            if dH_kv < dH:
                pad = dH - dH_kv
                k = torch.nn.functional.pad(k, (0, pad))
                v = torch.nn.functional.pad(v, (0, pad))
            else:  # dH_kv > dH
                k = k[:, :dH]
                v = v[:, :dH]
        
        # rope/yarn
        q, k = self._rope_apply(q, k, pos_idx)
        
        # Broadcast mask gain from current first intron (if enabled)
        mask_gain = 1.0
        if self.switch.get("cs_emission", True) and hasattr(self, 'current_token_id') and self.token_introns and self.INTRON_BROADCAST_MASKS is not None:
            tid = getattr(self, 'current_token_id', 0)
            intrs = self.token_introns[tid] if tid < len(self.token_introns) else None
            if intrs:
                intr0 = intrs[0] & 0xFF
                # mask: (48,) uint8 -> normalize [0,1]
                if 0 <= intr0 < self.INTRON_BROADCAST_MASKS.shape[0]:
                    mask48 = self.INTRON_BROADCAST_MASKS[intr0]  # numpy shape (48,)
                    mask_gain = 1.0 + 0.05 * (mask48.sum() / 48.0)  # small, stable gain

        # append caches (sliding window)
        K_cache.append(k); V_cache.append(v); S_cache.append(self.current_state_index)
        
        # Adaptive window based on theta proximity to BU_EG vs CS
        s_now = self.current_state_index
        th_now = float(self.theta[s_now])
        
        # When theta is near BU_EG, widen attention window; near CS, keep it tight
        if abs(th_now - THETA_BU_EG) < 0.2:  # Near BU_EG
            adaptive_window = int(window * 1.5)  # Widen window by 50%
        elif abs(th_now - THETA_CS) < 0.2:   # Near CS
            adaptive_window = int(window * 0.7)  # Tighten window by 30%
        else:
            adaptive_window = window  # Default window
        
        if len(K_cache) > adaptive_window:
            K_cache.pop(0); V_cache.pop(0); S_cache.pop(0)

        # attention scores via resonance (no softmax)
        # s(h,t) = 1 - normalized Hamming on bytes of q and k

        # compute weighted sum with theta and phenomenology modulation
        T = len(K_cache)
        ctx = torch.zeros_like(v)
        
        # theta proximity to current state
        s_now = self.current_state_index
        th_now = float(self.theta[s_now])
        
        # keep S_cache in sync with K/V caches
        if len(S_cache) > window:
            S_cache = S_cache[-window:]
        
        # Switch-controlled gains (default to pure resonance)
        one = torch.ones((), device=h_t.device, dtype=torch.float32)
        
        if self.switch.get("theta_gain", False):
            th_past = torch.tensor([float(self.theta[s]) for s in S_cache], dtype=torch.float32, device=h_t.device)
            theta_gain = one - (torch.abs(th_past - th_now) / np.pi)  # [0..1]
        else:
            theta_gain = one
        
        if self.switch.get("orbit_gain", False):
            ph_now = int(self.phenomenology[s_now])
            ph_past = torch.tensor([int(self.phenomenology[s]) for s in S_cache], dtype=torch.float32, device=h_t.device)
            same_orbit = (ph_past == ph_now).float()
            point_two = torch.tensor(0.2, device=h_t.device, dtype=torch.float32)
            orbit_gain = one + point_two * same_orbit  # +20% if same orbit
        else:
            orbit_gain = one
        
        for h in range(nH):
            ks = torch.stack([K_cache[t][h] for t in range(T)], dim=0)  # [T, dH]
            vs = torch.stack([V_cache[t][h] for t in range(T)], dim=0)  # [T, dH]
            scores = self._res_score_per_head(q[h], ks)                 # [T]
            
            # Epistemology-based hard masking for illegal back-jumps
            if hasattr(self, 'epistemology'):
                epistemology_mask = torch.ones(T, dtype=torch.float32)
                for t in range(T):
                    past_state = S_cache[t]
                    # Check if attending to this past state would represent an illegal back-jump
                    th_past = float(self.theta[past_state])
                    # Zero attention to states that would violate epistemological constraints
                    if (th_past < THETA_CS and th_now >= THETA_CS) or (th_past < THETA_UNA <= th_now):
                        epistemology_mask[t] = 0.0
                scores = scores * epistemology_mask
            
            # Ontology distance weighting (switch-controlled)
            if self.switch.get("ontology_distance_penalty", False) and hasattr(self, 'ontology'):
                ontology_weights = torch.ones(T, dtype=torch.float32)
                current_ontology = int(self.ontology[s_now])
                for t in range(T):
                    past_state = S_cache[t]
                    past_ontology = int(self.ontology[past_state])
                    # Compute Hamming distance in ontology space (48-bit states)
                    distance = bin(current_ontology ^ past_ontology).count('1')
                    # Down-weight based on distance (max distance is 48 bits)
                    distance_factor = 1.0 - 0.02 * (distance / 48.0)  # 2% penalty per bit difference
                    ontology_weights[t] = max(0.1, distance_factor)  # Minimum 10% weight
                scores = scores * ontology_weights
            
            # apply mask gain, then theta and phenomenology modulation
            scores = scores * mask_gain * theta_gain * orbit_gain
            
            # normalize to ~probabilities via positive clamp + L1
            w = torch.clamp(scores, min=0.0)
            if w.sum() > 0:
                w = w / w.sum()
            ctx[h] = (w.unsqueeze(1) * vs).sum(dim=0)

        # project out
        y = ctx.reshape(nH * dH)
        # Orient output projection weight to match concatenated attention output
        Wo = self._orient_W(Wo_raw, y.shape[-1])
        # Adapt y length once if needed to match Wo's input dim
        if Wo.shape[1] != y.numel():
            y = self._harmonize_len(y, Wo.shape[1])
        # Output projection using _fgemm_fold only
        h_next = self._fgemm_fold(y, Wo, bo)  # [H]
        return h_next, K_cache, V_cache, S_cache
        
    def _has_moe(self, layer_idx: int) -> bool:
        """Detect if a layer uses MoE by checking for expert weights."""
        expert_prefix = f"model.layers.{layer_idx}.mlp.experts."
        return any(k.startswith(expert_prefix) for k in self.model_weights)
    
    def _mlp_step(self, layer_idx: int, h_t: torch.Tensor):
        """MLP forward pass with auto-detection of dense vs MoE."""
        if self._has_moe(layer_idx):
            return self._mlp_step_moe(layer_idx, h_t)
        else:
            return self._mlp_step_dense(layer_idx, h_t)
    
    def _mlp_step_dense(self, layer_idx: int, h_t: torch.Tensor):
        """
        Dense MLP forward pass using fold-GEMM
        h_t: [H] float tensor
        """
        import torch
        import math
        
        # Weights (numeric, decoded)
        W_gate = self._layer_weight(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
        W_up = self._layer_weight(f"model.layers.{layer_idx}.mlp.up_proj.weight")
        W_down = self._layer_weight(f"model.layers.{layer_idx}.mlp.down_proj.weight")
        b_gate = self.model_weights.get(f"model.layers.{layer_idx}.mlp.gate_proj.bias")
        b_up = self.model_weights.get(f"model.layers.{layer_idx}.mlp.up_proj.bias")
        b_down = self.model_weights.get(f"model.layers.{layer_idx}.mlp.down_proj.bias")
        
        # Gate and up projections
        gate = self._fgemm_fold(h_t, W_gate, b_gate)
        up = self._fgemm_fold(h_t, W_up, b_up)
        
        # SwiGLU activation
        swiglu_limit = self.cfg.get("swiglu_limit", 7.0)
        gate_act = torch.sigmoid(gate) * torch.clamp(gate, -swiglu_limit, swiglu_limit)
        hidden = gate_act * up
        
        # Down projection
        return self._fgemm_fold(hidden, W_down, b_down)  # [H]
    
    def _mlp_step_moe(self, layer_idx: int, h_t: torch.Tensor):
        """
        MoE MLP forward pass with router and top-k expert selection
        h_t: [H] float tensor
        """
        import torch
        import torch.nn.functional as F

        # Router: make sure weight is [E, H]
        W_router_raw = self._layer_weight(f"model.layers.{layer_idx}.mlp.router.weight")
        W_router = self._orient_W(W_router_raw, h_t.shape[-1])

        # Sanitize router bias to be 1-D [E]
        b_router = self.model_weights.get(f"model.layers.{layer_idx}.mlp.router.bias")
        if b_router is not None and b_router.dim() != 1:
            b_router = b_router.reshape(-1)[:W_router.shape[0]]

        # Compute logits and force to shape [E]
        router_logits = self._fgemm_fold(h_t, W_router, b_router)  # expect [E]

        E = self.cfg.get("num_local_experts", 32)
        if router_logits.dim() != 1:
            rl = router_logits
            # Common case: [E, experts_per_token] → mean over trailing axis
            if rl.dim() == 2 and rl.shape[0] == E:
                rl = rl.mean(dim=1)
            else:
                # Generic fallback: collapse and average groups if multiple of E
                if rl.numel() % E == 0:
                    rl = rl.view(E, -1).mean(dim=1)
                else:
                    rl = rl.view(-1)[:E]
            router_logits = rl.contiguous().view(E)

        # Optional orbit bias
        if self.switch.get("router_orbit_bias", False):
            ph_now = int(self.phenomenology[self.current_state_index])
            if hasattr(self, 'expert_orbit_bias'):
                router_logits = router_logits + self.expert_orbit_bias[:, ph_now]

        experts_per_token = int(self.cfg.get("experts_per_token", 4))
        k = min(experts_per_token, router_logits.numel())
        top_k_logits, top_k_indices = torch.topk(router_logits, k=k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        expert_outputs = []
        for i in range(k):
            expert_idx = int(top_k_indices[i].item())

            W_gate, b_gate, W_up, b_up, W_down, b_down = self._moe_get_expert(layer_idx, expert_idx)

            gate = self._fgemm_fold(h_t, W_gate, b_gate)
            up   = self._fgemm_fold(h_t, W_up, b_up)

            # Infer I from the actual weight, not config; avoid undefined I
            I = W_up.shape[0]  # W_up is oriented as [out_dim, in_dim], so shape[0] is intermediate size
            swiglu_limit = float(self.cfg.get("swiglu_limit", 7.0))
            gate_act = torch.sigmoid(gate) * torch.clamp(gate, -swiglu_limit, swiglu_limit)
            hidden = gate_act * up  # [I]

            expert_out = self._fgemm_fold(hidden, W_down, b_down)  # [H]
            expert_outputs.append(expert_out * top_k_weights[i])
        
        # Combine expert outputs
        return sum(expert_outputs)  # [H]
        
    # Old _apply_layernorm renamed to _apply_rmsnorm for clarity
        
    # Old forward_pass removed - using the version with more complete implementation
        
    def generate_next_token(self, prev_token_id: int, pos: int) -> int:
        """Deterministic generation with configurable top-K and physics sieve."""
        import torch
        import os
        
        if self.switch["pure_resonance_only"]:
            t = self.next_token_resonant()
            return int(t) if t is not None else self.SEP_TOKEN

        # 1) real network
        Wemb = self._layer_weight("model.embed_tokens.weight")
        h = Wemb[prev_token_id].to(torch.float32)
        for l in range(self.cfg["num_hidden_layers"]):
            h, self._caches = self._layer_step(l, h, self._caches, pos)

        # 2) logits (tile-aware, top-K only)
        topK = int(os.getenv("GYRO_TOPK", "512"))
        idxs, vals = self._logits(h, return_topk=True, top_k=topK)  # returns (indices, values)

        # 3) physics sieve

        gated: list[tuple[int, float]] = []
        pm = self.path_memory & 0xFF
        proposed = self._propose_intron()
        p_mask = self.broadcast_masks[proposed] if self.broadcast_masks is not None else None

        for tok in idxs.tolist():
            intrs = self.token_introns[tok]
            if not intrs or not self._token_respects_cycle(self.current_state_index, intrs):
                continue
            post = int(self._token_post_state_index_arr[tok])
            te   = int(self._token_exon_arr[tok])
            # resonance gate: must align with path/exon
            if fold(pm, te) == 0:
                continue
            tb = 0.0
            if p_mask is not None:
                first_mask = self.broadcast_masks[intrs[0] & 0xFF]
                tb = float(np.dot(p_mask, first_mask))
            gated.append((tok, tb))

        if not gated:
            return self.SEP_TOKEN

        # deterministic tie-break: by tb desc then token_id asc
        gated.sort(key=lambda x: (-x[1], x[0]))
        best = int(gated[0][0])

        # 4) evolve state (no learning on egress)
        for intr in self.token_to_introns(best):
            self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intr)

        return best
        
    def forward_pass(self, input_ids, past_key_values=None):
        """
        Run the full model forward pass using physics-based fold operations
        input_ids: tensor of token ids [B, S]
        past_key_values: optional cached KV tensors for incremental decoding
        
        Returns:
            logits: output logits [B, S, V]
            new_key_values: updated KV cache
        """
        import torch
        from collections import defaultdict
        
        batch_size, seq_len = input_ids.shape
        num_layers = self.cfg["num_hidden_layers"]
        sliding_window = self.cfg.get("sliding_window", 128)
        layer_types = self.cfg.get("layer_types", ["full_attention"] * num_layers)
        
        # Initialize KVS cache if not provided
        if past_key_values is None:
            past_key_values = []
            for _ in range(num_layers):
                past_key_values.append(([], [], []))  # K, V, S caches
        
        # Embedding lookup - derive hidden_size from weights
        embed_weight = self._layer_weight("model.embed_tokens.weight")
        hidden_size = embed_weight.shape[1]  # derive from weights, not config
        h = torch.zeros((batch_size, seq_len, hidden_size), dtype=torch.float32)
        for b in range(batch_size):
            for s in range(seq_len):
                h[b, s] = embed_weight[input_ids[b, s]]
        
        # Process each position in the sequence
        for pos in range(seq_len):
            # Track position for RoPE and current token for broadcast masks
            pos_idx = len(past_key_values[0][0]) + pos  # cache length + current position
            self.current_token_id = input_ids[0, pos].item() if batch_size > 0 else 0
            
            # Process through layers
            for layer_idx in range(num_layers):
                K_cache, V_cache, S_cache = past_key_values[layer_idx]
                layer_type = layer_types[layer_idx % len(layer_types)]
                
                # Layer norm before attention
                ln1_weight = self._layer_weight(f"model.layers.{layer_idx}.input_layernorm.weight")
                ln1_bias = self.model_weights.get(f"model.layers.{layer_idx}.input_layernorm.bias")
                h_norm = self._apply_rmsnorm(h[:, pos], ln1_weight, ln1_bias)
                
                # Attention with window based on layer type
                window = sliding_window if "sliding" in layer_type else 100000
                h_attn, K_cache, V_cache, S_cache = self._attn_step(
                    layer_idx, h_norm, K_cache, V_cache, S_cache, pos_idx, window
                )
                past_key_values[layer_idx] = (K_cache, V_cache, S_cache)
                
                # Residual connection
                h_res = h[:, pos] + h_attn
                
                # Layer norm before MLP
                ln2_weight = self._layer_weight(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
                ln2_bias = self.model_weights.get(f"model.layers.{layer_idx}.post_attention_layernorm.bias")
                h_norm2 = self._apply_rmsnorm(h_res, ln2_weight, ln2_bias)
                
                # MLP
                h_mlp = self._mlp_step(layer_idx, h_norm2)
                
                # Final residual
                h[:, pos] = h_res + h_mlp
        
        # Final layer norm
        ln_f_weight = self._layer_weight("model.norm.weight")
        ln_f_bias = self.model_weights.get("model.norm.bias")
        hidden_size = h.shape[-1]  # use current hidden size from tensor
        h_final = torch.zeros((batch_size, seq_len, hidden_size), dtype=torch.float32)
        for pos in range(seq_len):
            h_final[:, pos] = self._apply_rmsnorm(h[:, pos], ln_f_weight, ln_f_bias)
        
        # Logits
        lm_head_weight = self._layer_weight("lm_head.weight")
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=torch.float32)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s] = self._fgemm_fold(h_final[b, s], lm_head_weight.T.contiguous())
        
        return logits, past_key_values
    
    def _apply_rmsnorm(self, x, weight, bias=None, eps=1e-5):
        """
        Apply RMS normalization (no mean subtraction)
        """
        import torch
        if x.dim() == 1:
            denom = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        else:
            denom = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        y = x * denom
        y = y * weight  # broadcast
        if bias is not None:
            y = y + bias
        return y

    def token_to_introns(self, token_id: int) -> List[int]:
        """Convert token ID to intron sequence via LEB128 encoding."""
        leb = self._id_to_uleb128(token_id)
        return [transcribe_byte(b) for b in leb]
    
    def _id_to_uleb128(self, token_id: int) -> List[int]:
        """Convert token ID to unsigned LEB128 bytes."""
        if token_id == 0:
            return [0]
        
        result = []
        while token_id > 0:
            byte = token_id & 0x7F
            token_id >>= 7
            if token_id > 0:
                byte |= 0x80
            result.append(byte)
        return result

    def _build_token_post_states(self) -> None:
        """Build dense arrays: post-state index per token and token exons."""
        V = int(self.vocab_size)
        cs = int(self.CS_STATE_INDEX)

        post = np.empty(V, dtype=np.int32)
        exons = np.empty(V, dtype=np.uint8)

        for token_id in range(V):
            introns = self.token_to_introns(token_id)
            s = cs
            for intron in introns:
                s = int(self.epistemology[s, intron & 0xFF])
            post[token_id] = s
            exons[token_id] = compute_exon_from_state(int(self.ontology[s])) & 0xFF

        self._token_post_state_index_arr = post
        self._token_exon_arr = exons

    def _build_orbit_candidates(self) -> None:
        """Orbit → list[token] from dense post-state array."""
        self._orbit_candidates.clear()
        if self._token_post_state_index_arr is None:
            return
        post = self._token_post_state_index_arr
        reps = self.phenomenology[post].astype(np.int32)

        # group by orbit rep
        buckets: Dict[int, List[int]] = defaultdict(list)
        for tok, rep in enumerate(reps):
            buckets[int(rep)].append(int(tok))

        self._orbit_candidates = buckets

    def _precompute_una_pool(self) -> None:
        """Precompute UNA states for boundary layer operations with CS axiom."""
        # Find states with theta close to π/4
        target_theta = np.pi / 4
        tight_tolerance = 0.05  # Tighter tolerance for sharper CS emission
        fallback_tolerance = 0.1

        # Try tight tolerance first
        self._UNA_pool = np.argwhere(np.abs(self.theta - target_theta) < tight_tolerance).astype(np.int32).ravel()

        # If too small, fallback to looser tolerance
        if len(self._UNA_pool) < 10:  # Minimum viable pool size
            self._UNA_pool = np.argwhere(np.abs(self.theta - target_theta) < fallback_tolerance).astype(np.int32).ravel()

        # Final fallback: use states with theta in UNA range
        if len(self._UNA_pool) == 0:
            self._UNA_pool = np.argwhere((self.theta > THETA_CS) & (self.theta < THETA_ONA)).astype(np.int32).ravel()

    def _state_ints_for_indices(self, idxs: np.ndarray) -> np.ndarray:
        return self.ontology[idxs].astype(np.uint64)

    def _precompute_intron_to_una_index(self) -> np.ndarray:
        """
        Map each intron (0..255) → a UNA-state index in self._UNA_pool, deterministically.
        We choose the UNA state whose 48-bit pattern has maximum bit-overlap with the
        broadcast mask of the intron. Done once; O(256 * |UNA_pool|).
        """
        if self._UNA_pool.size == 0:
            return np.full(256, self.CS_STATE_INDEX, dtype=np.int32)

        masks = self.INTRON_BROADCAST_MASKS  # (256, 48) uint8
        if masks is None:
            return np.zeros(256, dtype=np.int32)
            
        una_state_ints = self._state_ints_for_indices(self._UNA_pool)  # uint64

        # Unpack UNA states into (|pool|, 48) {0,1} bits
        # Convert uint64 to bytes and unpack bits
        una_bytes = una_state_ints.view(np.uint64).byteswap().view(np.uint8)  # (|pool| * 8,)
        una_bytes = una_bytes.reshape(-1, 8)  # (|pool|, 8)
        una_bits = np.unpackbits(una_bytes, axis=1)  # (|pool|, 64)
        # We only need 48 LSBs
        una_bits = una_bits[:, -48:].astype(np.uint8)  # (|pool|, 48)

        intron_to_idx = np.zeros(256, dtype=np.int32)
        for intron in range(256):
            mask48 = masks[intron]  # (48,)
            # overlap = sum(mask48 & una_bits) over axis=1
            overlaps = (una_bits & mask48).sum(axis=1)  # (|pool|,)
            best = int(np.argmax(overlaps))
            if self._UNA_pool is not None:
                intron_to_idx[intron] = int(self._UNA_pool[best])
            else:
                intron_to_idx[intron] = 0
        return intron_to_idx

    def _build_token_introns_index(self) -> None:
        """Build complete token-intron mappings for fast lookup."""
        self.token_introns: List[List[int]] = [None] * self.vocab_size  # type: ignore
        self.first_intron_to_tokens: Dict[int, List[int]] = defaultdict(list)
        
        for tid in range(self.vocab_size):
            intrs = self.token_to_introns(tid)
            self.token_introns[tid] = intrs
            if intrs:
                self.first_intron_to_tokens[intrs[0] & 0xFF].append(tid)
                
    def _stage_of_theta(self, th: float) -> int:
        """Determine the stage index based on theta value."""
        if th < THETA_CS: return 0
        if th < THETA_UNA: return 1
        if th < THETA_ONA: return 2
        if th < THETA_BU_IN: return 3
        if th < THETA_BU_EG: return 4
        return 5  # CLOSURE

    def _propose_intron(self) -> int:
        """Parity-preserving proposal: intron := fold(path_memory, exon(state))."""
        s_int = int(self.ontology[self.current_state_index])
        exon = compute_exon_from_state(s_int)
        return self._resonance_table[self.path_memory & 0xFF, exon & 0xFF].item()

    def _token_respects_cycle(self, start_index: int, introns: List[int]) -> bool:
        """Simulate token introns; optionally enforce monotone cycle (no regress across major boundaries)."""
        if not getattr(self, "cycle_gating", False):
            return True
        th0 = float(self.theta[start_index])
        stage0 = self._stage_of_theta(th0)
        idx = start_index
        for intr in introns:
            nxt = int(self.epistemology[idx, intr & 0xFF])
            stage = self._stage_of_theta(float(self.theta[nxt]))
            if stage + 1 < stage0:  # hard backward jump
                return False
            idx = nxt
            stage0 = max(stage0, stage)
        return True

    def next_token_resonant(self) -> Optional[int]:
        """Pure resonance selection: intron-first → token.
        No scores, no randomness, no Hebbian."""
        intron = self._propose_intron()
        cand = self.first_intron_to_tokens.get(intron, [])
        if not cand:
            return None  # no token starts with that intron

        for token_id in cand:
            intrs = self.token_introns[token_id]
            if self._token_respects_cycle(self.current_state_index, intrs):
                return token_id
        return None

    def _best_divisor(self, N: int, prefer: int) -> int:
        """Find the best divisor of N close to the preferred value."""
        if prefer > 0 and N % prefer == 0:
            return prefer
        # search down, then up
        for d in range(min(prefer, N), 0, -1):
            if N % d == 0:
                return d
        for d in range(prefer + 1, N + 1):
            if N % d == 0:
                return d
        return 1

    def _harmonize_len(self, vec: torch.Tensor, L: int) -> torch.Tensor:
        """Adapt a 1D tensor to length L via reduce or repeat-pad.
        Keeps physics stable by single adaptation per boundary.
        """
        import torch
        v = vec.view(-1)
        N = v.numel()
        if N == L:
            return v
        if N > L:
            step = max(1, N // L)
            trimmed = v[: step * L].view(L, step)
            return trimmed.mean(dim=1)
        else:
            reps = (L + N - 1) // N
            out = v.repeat(reps)[:L]
            return out
    
    def _orient_W(self, W: torch.Tensor, x_dim: int) -> torch.Tensor:
        """Orient weight matrix to match input dimension.
        Expected W: [out_dim, in_dim]
        """
        if W.shape[1] == x_dim:
            return W
        if W.shape[0] == x_dim:  # likely transposed
            return W.T.contiguous()
        
        # Handle dimension mismatch by adjusting the weight matrix
        if W.shape[1] > x_dim:
            # Slice weight to match input dimension
            return W[:, :x_dim]
        elif W.shape[1] < x_dim:
            # Pad weight to match input dimension
            pad_size = x_dim - W.shape[1]
            padding = torch.zeros(W.shape[0], pad_size, device=W.device, dtype=W.dtype)
            return torch.cat([W, padding], dim=1)
        
        return W

    def _load_model_weights(self) -> None:
        """Load numeric tensors from model.gyro.safetensors (decode <key>.gyro with <key>.meta)."""
        possible_paths = [
            self.base_path / "model.gyro.safetensors",
            self.base_path / "models" / "gpt-oss-20b" / "model.gyro.safetensors",
        ]
        model_weights_path = next((p for p in possible_paths if p.exists()), None)

        if model_weights_path is None:
            print(f"[warning] No model.gyro.safetensors found in {self.base_path}")
            print(f"[info] Searched paths: {[str(p) for p in possible_paths]}")
            self.model_weights = {}
            self.model_weight_meta = {}
            return

        try:
            from safetensors import safe_open
        except ImportError as e:
            print(f"[error] Failed to import safetensors: {e}")
            print("[info] Install with: pip install safetensors")
            self.model_weights = {}
            self.model_weight_meta = {}
            return

        try:
            from kernel.codecs.gyrowt import decode_gyro_tensor
        except ImportError as e:
            print(f"[error] Failed to import decode_gyro_tensor: {e}")
            print("[info] Ensure kernel.codecs.gyrowt module is available")
            self.model_weights = {}
            self.model_weight_meta = {}
            return

        import json

        self.model_weights = {}
        self.model_weight_meta = {}
        print(f"[info] Loading model weights from: {model_weights_path}")

        try:
            with safe_open(str(model_weights_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if not keys:
                    print(f"[warning] No keys found in {model_weights_path}")
                    return

                successful_loads = 0
                failed_loads = 0

                for key in keys:
                    try:
                        if key.endswith(".gyro"):
                            base = key[:-5]
                            blob = f.get_tensor(key)
                            if blob is None:
                                print(f"[warning] Failed to load tensor for key: {key}")
                                failed_loads += 1
                                continue

                            meta_key = base + ".meta"
                            meta_bytes = None
                            if meta_key in keys:
                                try:
                                    meta_t = f.get_tensor(meta_key)
                                    if meta_t is not None:
                                        meta_bytes = meta_t.cpu().numpy().tobytes()
                                        # Store metadata for MoE expert slicing
                                        try:
                                            meta_json = json.loads(meta_bytes.decode("utf-8"))
                                            self.model_weight_meta[base] = meta_json
                                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                            print(f"[warning] Failed to parse metadata for {base}: {e}")
                                except Exception as e:
                                    print(f"[warning] Failed to load metadata for {meta_key}: {e}")

                            # decode to a real torch tensor on CPU
                            try:
                                t = decode_gyro_tensor(base, blob, meta_bytes, device="cpu")
                                if t is not None:
                                    self.model_weights[base] = t  # numeric
                                    successful_loads += 1
                                else:
                                    print(f"[warning] decode_gyro_tensor returned None for {base}")
                                    failed_loads += 1
                            except RuntimeError as e:
                                print(f"[error] Failed to decode gyro tensor for {base}: {e}")
                                print(f"[info] Skipping corrupted tensor {base} and continuing with other tensors")
                                failed_loads += 1
                            except Exception as e:
                                print(f"[error] Unexpected error decoding gyro tensor for {base}: {e}")
                                failed_loads += 1

                        elif (key not in self.model_weights) and (not key.endswith(".meta")):
                            # plain numeric (if any)
                            try:
                                tensor = f.get_tensor(key)
                                if tensor is not None:
                                    self.model_weights[key] = tensor
                                    successful_loads += 1
                                else:
                                    print(f"[warning] Failed to load plain tensor for key: {key}")
                                    failed_loads += 1
                            except Exception as e:
                                print(f"[error] Failed to load plain tensor for {key}: {e}")
                                failed_loads += 1

                    except Exception as e:
                        print(f"[error] Unexpected error processing key {key}: {e}")
                        failed_loads += 1

                print(f"[info] Successfully loaded {successful_loads} tensors")
                if failed_loads > 0:
                    print(f"[warning] Failed to load {failed_loads} tensors")

                if successful_loads == 0:
                    print(f"[error] No tensors were successfully loaded from {model_weights_path}")
                    print("[info] Model may not function correctly without weights")

        except Exception as e:
            print(f"[error] Failed to open or read model weights file {model_weights_path}: {e}")
            print("[info] Continuing with empty model weights - model may not function correctly")
            self.model_weights = {}
            self.model_weight_meta = {}

    def _mxfp4_dequantize(self, blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """Dequantize MXFP4 tensors (blocks + scales) to dense format.
        
        Args:
            blocks: [..., G, B] uint8 with 2 nibbles per byte
            scales: [..., G] int8/uint8 exponents offset by 127
            dtype: Target dtype for output tensor
            
        Returns:
            Dense tensor with shape [..., G*B*2]
        """
        import torch
        
        # MXFP4 lookup table
        lut = torch.tensor([+0.0,+0.5,+1.0,+1.5,+2.0,+3.0,+4.0,+6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], 
                           dtype=dtype, device=blocks.device)
        
        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} vs {scales.shape=}"
        
        *prefix, G, B = blocks.shape
        rows_total = (1 if not prefix else int(torch.tensor(prefix).prod().item())) * G
        
        blk = blocks.reshape(rows_total, B)
        exp = (scales.to(torch.int32) - 127).reshape(rows_total, 1)  # per-row exponents
        
        out = torch.empty(rows_total, B*2, dtype=dtype, device=blocks.device)
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)
        out[:, 0::2] = lut[idx_lo]
        out[:, 1::2] = lut[idx_hi]
        
        torch.ldexp(out, exp, out=out)  # multiply by 2^exp rowwise
        return out.reshape(*prefix, G*B*2)  # collapse last two dims into columns

    def _orient_gate_up(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Orient and split fused gate_up projection weights.
        
        Args:
            W: Fused gate+up weights tensor
            
        Returns:
            Tuple of (W_gate, W_up) both with shape [H, I]
        """
        H = self.cfg["hidden_size"]
        I = self.cfg["intermediate_size"]
        twoI = 2 * I
        
        # Common storage: [E, twoI, H] -> transpose to [E, H, twoI]
        if W.dim() == 3 and W.shape[-2] == twoI and W.shape[-1] == H:
            W = W.movedim(-2, -1)  # [E, H, twoI]
        # Or already [E, H, twoI]
        assert W.shape[-2] == H and W.shape[-1] == twoI, f"gate_up shape mismatch {tuple(W.shape)}"
        
        W_gate = W[..., :I]   # [E,H,I]
        W_up   = W[..., I:]   # [E,H,I]
        return W_gate.contiguous(), W_up.contiguous()

    def _orient_down(self, W: torch.Tensor) -> torch.Tensor:
        """Orient down projection weights.
        
        Args:
            W: Down projection weights tensor
            
        Returns:
            Tensor with shape [E, I, H]
        """
        H = self.cfg["hidden_size"]
        I = self.cfg["intermediate_size"]
        
        if W.shape[-2] == H and W.shape[-1] == I:
            W = W.movedim(-2, -1)  # [E, I, H]
        assert W.shape[-2] == I and W.shape[-1] == H, f"down_proj shape mismatch {tuple(W.shape)}"
        return W.contiguous()

    def _moe_get_expert(self, layer_idx: int, expert_idx: int):
        """Lazy per-expert materialization with LRU cache.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index within the layer
            
        Returns:
            Tuple of (W_gate, b_gate, W_up, b_up, W_down, b_down)
        """
        import torch
        
        key = (layer_idx, expert_idx)
        if not hasattr(self, "_moe_cache"):
            self._moe_cache = {}
        if key in self._moe_cache:
            return self._moe_cache[key]

        base = f"model.layers.{layer_idx}.mlp.experts"
        
        # Dequantize gate_up
        bu = self._layer_weight(f"{base}.gate_up_proj_blocks")
        su = self._layer_weight(f"{base}.gate_up_proj_scales")
        
        # Slice expert dimension (assume first dim = E)
        if bu.dim() == 3:   # [E, G, B] -> after dequant: [E, cols]
            bu_e, su_e = bu[expert_idx], su[expert_idx]
        elif bu.dim() == 4: # [E,*,G,B] rare; slice first, keep rest
            bu_e, su_e = bu[expert_idx], su[expert_idx]
        else:
            raise RuntimeError(f"Unexpected gate_up blocks dims: {bu.shape}")

        W_u = self._mxfp4_dequantize(bu_e, su_e, dtype=torch.bfloat16)  # [rows, cols]
        
        # Use dimension-agnostic orientation for gate_up weights
        # Assume W_u should be oriented to match input dimension for gate/up projections
        # The weight should be [2*intermediate, hidden] after orientation
        if W_u.dim() == 2:
            # Try to orient based on which dimension could be 2*intermediate
            if W_u.shape[0] % 2 == 0 and W_u.shape[0] > W_u.shape[1]:
                # Likely [2I, H] already
                W_u = W_u.unsqueeze(0)  # [1, 2I, H]
            elif W_u.shape[1] % 2 == 0 and W_u.shape[1] > W_u.shape[0]:
                # Likely [H, 2I], transpose to [2I, H]
                W_u = W_u.T.unsqueeze(0)  # [1, 2I, H]
            else:
                # Fallback: assume first dim is correct
                W_u = W_u.unsqueeze(0)  # [1, rows, cols]
        elif W_u.dim() == 3:
            # Already has batch dimension
            pass
        else:
            raise RuntimeError(f"Unexpected gate_up weight dimensions: {W_u.shape}")
            
        W_gate, W_up = self._orient_gate_up(W_u)  # each [1,H,I]
        W_gate, W_up = W_gate[0], W_up[0]         # drop leading dim
        # Ensure _fgemm_fold sees weights as [output_dim, input_dim] => gate/up must be [I, H]
        W_gate = W_gate.T.contiguous()
        W_up = W_up.T.contiguous()

        # Dequantize down
        bd = self._layer_weight(f"{base}.down_proj_blocks")
        sd = self._layer_weight(f"{base}.down_proj_scales")
        bd_e, sd_e = bd[expert_idx], sd[expert_idx]
        W_d = self._mxfp4_dequantize(bd_e, sd_e, dtype=torch.bfloat16)  # [rows, cols]
        
        # Use dimension-agnostic orientation for down projection
        # Down projection should be [intermediate, hidden] to project from intermediate back to hidden
        if W_d.shape[0] < W_d.shape[1]:
            # Likely [I, H] already - correct orientation
            pass
        elif W_d.shape[0] > W_d.shape[1]:
            # Likely [H, I] - transpose to [I, H]
            W_d = W_d.T
        # If dimensions are equal, assume current orientation is correct

        # Biases - derive dimensions from actual weights
        intermediate_size = W_gate.shape[0]  # W_gate is [intermediate, hidden]
        hidden_size = W_gate.shape[1]
        
        b_fused = self._layer_weight(f"{base}.gate_up_proj_bias")  # [E, 2I] or [2I, E]
        num_experts = self.cfg.get("num_local_experts", 32)
        if b_fused.shape[0] == num_experts:
            b_e = b_fused[expert_idx]
        else:
            b_e = b_fused[:, expert_idx]
        
        # Split bias safely based on actual dimensions
        b_e_size = b_e.numel()
        half_size = b_e_size // 2
        b_gate = b_e[:half_size].contiguous()
        b_up = b_e[half_size:].contiguous()

        b_down = self._layer_weight(f"{base}.down_proj_bias")      # [E, H] or [H, E]
        if b_down.shape[0] == num_experts:
            b_down = b_down[expert_idx]
        else:
            b_down = b_down[:, expert_idx]
        b_down = b_down.contiguous()

        self._moe_cache[key] = (W_gate, b_gate, W_up, b_up, W_d, b_down)
        
        # Simple LRU cap
        if len(self._moe_cache) > 8 * self.cfg["num_hidden_layers"]:
            self._moe_cache.pop(next(iter(self._moe_cache)))
            
        return self._moe_cache[key]

    # Weight-informed field computation removed for direct tensor operations

    def _load_physics_tables(self) -> None:
        """Load all physics tables with memory mapping and validate their integrity."""
        meta_path = self.base_path / "public" / "meta"

        # Ontology (state integers)
        ontology_path = meta_path / "ontology_keys.npy"
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")
        ontology_raw = np.load(ontology_path, mmap_mode="r")
        
        # Convert to int64 if needed (handle uint64 -> int64 conversion)
        if ontology_raw.dtype == np.uint64:
            self.ontology = ontology_raw.astype(np.int64)
        else:
            self.ontology = ontology_raw
            
        if self.ontology.ndim != 1:
            raise ValueError(f"Ontology must be 1D, got shape {self.ontology.shape}")
        if self.ontology.dtype != np.int64:
            raise ValueError(f"Ontology must be int64, got dtype {self.ontology.dtype}")

        # Epistemology (state transitions)
        epistemology_path = meta_path / "epistemology.npy"
        if not epistemology_path.exists():
            raise FileNotFoundError(f"Epistemology not found: {epistemology_path}")
        self.epistemology = np.load(epistemology_path, mmap_mode="r")
        if self.epistemology.ndim != 2 or self.epistemology.shape[0] != self.ontology.shape[0]:
            raise ValueError(f"Epistemology must be 2D with first dim matching ontology, got shape {self.epistemology.shape}")

        # Theta (angular divergence)
        theta_path = meta_path / "theta.npy"
        if not theta_path.exists():
            raise FileNotFoundError(f"Theta not found: {theta_path}")
        self.theta = np.load(theta_path, mmap_mode="r")
        if self.theta.ndim != 1 or self.theta.shape[0] != self.ontology.shape[0]:
            raise ValueError(f"Theta must be 1D matching ontology, got shape {self.theta.shape}")
        if not np.all((self.theta >= 0) & (self.theta <= np.pi)):
            raise ValueError("Theta values must be in [0, pi]")

        # Phenomenology (orbit mapping)
        pheno_path = meta_path / "phenomenology_map.npy"
        if not pheno_path.exists():
            raise FileNotFoundError(f"Phenomenology not found: {pheno_path}")
        self.phenomenology = np.load(pheno_path, mmap_mode="r")
        if self.phenomenology.ndim != 1 or self.phenomenology.shape[0] != self.ontology.shape[0]:
            raise ValueError(f"Phenomenology must be 1D matching ontology, got shape {self.phenomenology.shape}")

        # Orbit sizes
        orbit_sizes_path = meta_path / "orbit_sizes.npy"
        if not orbit_sizes_path.exists():
            raise FileNotFoundError(f"Orbit sizes not found: {orbit_sizes_path}")
        self.orbit_sizes = np.load(orbit_sizes_path, mmap_mode="r")
        if self.orbit_sizes.ndim != 1 or self.orbit_sizes.shape[0] != self.ontology.shape[0]:
            raise ValueError(f"Orbit sizes must be 1D matching ontology, got shape {self.orbit_sizes.shape}")
        if not np.all(self.orbit_sizes > 0):
            raise ValueError("Orbit sizes must be positive")

    def _load_broadcast_masks(self) -> None:
        """Load broadcast masks for boundary layer operations with CS as extra-phenomenal axiom."""
        meta_path = self.base_path / "public" / "meta"
        broadcast_masks_path = meta_path / "intron_broadcast_masks.npy"

        if broadcast_masks_path.exists():
            self.INTRON_BROADCAST_MASKS = np.load(broadcast_masks_path, mmap_mode="r")
            self.broadcast_masks = self.INTRON_BROADCAST_MASKS  # Add alias
        else:
            # Generate if not found
            os.makedirs(meta_path, exist_ok=True)
            masks = generate_intron_broadcast_masks()
            np.save(broadcast_masks_path, masks)
            self.INTRON_BROADCAST_MASKS = masks
            self.broadcast_masks = self.INTRON_BROADCAST_MASKS  # Add alias

    def _find_cs_state(self) -> None:
        """Find the CS state (minimum theta) as extra-phenomenal reference point."""
        min_theta_idx = int(np.argmin(self.theta))
        min_theta = float(self.theta[min_theta_idx])

        self.CS_STATE_INDEX = min_theta_idx
        self.CS_STATE_INT = int(self.ontology[min_theta_idx])
        
        print(f"[debug] CS state found: index={self.CS_STATE_INDEX}, theta={min_theta:.4f}")
    
    def _build_expert_orbit_bias(self):
        """Build expert orbit bias from real expert weights using fold bytes."""
        if not hasattr(self, 'phenomenology'):
            return
        n_orbits = int(np.max(self.phenomenology)) + 1
        n_experts = int(self.cfg.get("num_local_experts", 32))
        self.expert_orbit_bias = torch.zeros(n_experts, n_orbits, dtype=torch.float32)

        # precompute orbit reps' exons
        reps = np.arange(self.phenomenology.shape[0])
        # one rep per orbit: first index with that orbit id
        seen = set()
        orbit_rep = []
        for i, o in enumerate(self.phenomenology):
            o = int(o)
            if o not in seen:
                seen.add(o)
                orbit_rep.append((o, i))
        rep_exon = np.zeros(n_orbits, dtype=np.uint8)
        for o, idx in orbit_rep:
            rep_exon[o] = compute_exon_from_state(int(self.ontology[idx])) & 0xFF

        # fold bytes of each expert's (up,down) into a signature byte
        for e in range(n_experts):
            sig = 0xAA
            for name in ("up_proj.weight", "down_proj.weight"):
                k = f"model.layers.0.mlp.experts.{e}.{name}"  # any layer works for bias signature
                if k in self.model_weights:
                    w = self.model_weights[k].contiguous().view(-1).to(torch.uint8).cpu().numpy()
                    # sample a small stride to stay O(N)
                    stride = max(1, w.size // 4096)
                    for i in range(0, w.size, stride):
                        sig = fold(sig, int(w[i]))
            # resonance vs orbit exon → small bias
            for o in range(n_orbits):
                self.expert_orbit_bias[e, o] = (fold(sig, int(rep_exon[o])) - 128) / 1280.0



    def _set_special_tokens(self) -> None:
        """Set special tokens."""
        # Default special tokens
        self.CLS_TOKEN = 1  #

    def get_token_post_state(self, token_id: int) -> int:
        """Get post-state index for token using dense array (O(1) access)."""
        if self._token_post_state_index_arr is None:
            self._build_token_post_states()
        assert self._token_post_state_index_arr is not None
        return int(self._token_post_state_index_arr[token_id])
    
    def get_token_exon(self, token_id: int) -> int:
        """Get exon for token using dense array (O(1) access)."""
        if self._token_exon_arr is None:
            self._build_token_post_states()
        assert self._token_exon_arr is not None
        return int(self._token_exon_arr[token_id])

    def _apply_intron_and_gate(self, state_index: int, intron: int) -> int:
        """Apply intron transition using generic physics with optional cycle gating.
        
        Pure physics transition - no learning/path_memory mutation here.
        CS is treated as extra-phenomenal and handled at the boundary layer.
        """
        # Generic transition via epistemology for all states
        nxt = int(self.epistemology[state_index, intron & 0xFF])
        
        if self.switch.get("cycle_gating", False):
            th_now = float(self.theta[state_index])
            th_nxt = float(self.theta[nxt])
            lo, hi = self._allowed_theta_window(th_now)
            if not (lo <= th_nxt <= hi):
                return state_index  # reject illegal back-jump
        
        return nxt
            
    def _layer_step(self, layer_idx: int, h_t: torch.Tensor, caches, pos_idx: int): 
        """One transformer block: Attn → MLP with residuals + RMSNorm."""
        # pre-attn norm 
        ln1_w = self._layer_weight(f"model.layers.{layer_idx}.input_layernorm.weight") 
        ln1_b = self.model_weights.get(f"model.layers.{layer_idx}.input_layernorm.bias") 
        h_norm = self._apply_rmsnorm(h_t, ln1_w, ln1_b) 
 
        # attn 
        layer_type = self.cfg["layer_types"][layer_idx] 
        window = int(self.cfg.get("sliding_window", 128)) if layer_type == "sliding_attention" else pos_idx + 1 
        K_cache, V_cache, S_cache = caches[layer_idx] 
        attn_out, K_cache, V_cache, S_cache = self._attn_step(layer_idx, h_norm, K_cache, V_cache, S_cache, pos_idx, window) 
        caches[layer_idx] = (K_cache, V_cache, S_cache) 
        h = h_t + attn_out 
 
        # pre-mlp norm 
        ln2_w = self._layer_weight(f"model.layers.{layer_idx}.post_attention_layernorm.weight") 
        ln2_b = self.model_weights.get(f"model.layers.{layer_idx}.post_attention_layernorm.bias") 
        h_norm2 = self._apply_rmsnorm(h, ln2_w, ln2_b) 
 
        mlp_out = self._mlp_step(layer_idx, h_norm2) 
        return h + mlp_out, caches
        
    def ingest_token(self, token_id: int, pos: int) -> None:
        """Update physics state with the actual input token without generating a new token."""
        import torch
        # Get embedding
        Wemb = self._layer_weight("model.embed_tokens.weight") \
            if "model.embed_tokens.weight" in self.model_weights else self._layer_weight("embed_tokens.weight")
        x = Wemb[token_id].to(torch.float32)
        
        # Initialize caches if needed
        if not hasattr(self, "_caches"):
            self._init_caches(self.cfg["num_hidden_layers"])
            
        # Process through layers
        h = x
        for L in range(self.cfg["num_hidden_layers"]):
            h, self._caches = self._layer_step(L, h, self._caches, pos)
            
        # Update physics with the actual input token (egress path - no learning)
        for intr in self.token_to_introns(token_id):
            self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intr)
        # Do NOT fold path_memory here; learning only at BU_IN by policy
        
    def learn_token(self, token_id: int) -> None:
        """Explicit learning method: holographic memory only at BU_IN.
        
        Learn at closing intron relative to pre-state orbit.
        Separate from ingestion - this is the only place learning happens.
        """
        intrs = self.token_to_introns(token_id)
        if not intrs:
            return

        pre = self.current_state_index
        pre_orbit = int(self.phenomenology[pre])
        pre_state_int = int(self.ontology[pre])
        exon0 = compute_exon_from_state(pre_state_int)

        # evolve to just before last intron
        for intr in intrs[:-1]:
            self.current_state_index = self._apply_intron_and_gate(self.current_state_index, intr)

        # hinge at closing intron
        closing = intrs[-1] & 0xFF
        mask = fold(exon0, closing)

        # sparse store only if deviates
        if pre_orbit not in self.orbit_patterns:
            self.orbit_patterns[pre_orbit] = {}
        if self.orbit_patterns[pre_orbit].get(token_id) != mask:
            self.orbit_patterns[pre_orbit][token_id] = mask

        # finalize evolution
        self.current_state_index = self._apply_intron_and_gate(self.current_state_index, closing)

        # update path memory once per token (not per intron)
        self.path_memory = fold(self.path_memory, mask)
        
    def _init_caches(self, max_layers: int): 
        self._caches = [( [], [], [] ) for _ in range(max_layers)]  # per layer: (K_list, V_list, S_list)
        
    def _logits(self, h_t: torch.Tensor, tile_size: int = 8192, return_topk: bool = True, top_k: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute logits with tiling for memory efficiency on large vocabularies.
        
        Args:
            h_t: Hidden state tensor
            tile_size: Size of vocabulary tiles for memory efficiency
            return_topk: If True, return (indices, values) for top-K only
            top_k: Number of top tokens to return (auto-computed if None)
            
        Returns:
            If return_topk=True: (top_indices, top_values)
            If return_topk=False: full logits tensor
        """
        import torch
        
        W = self._layer_weight("lm_head.weight")  # [V, D]
        b = self.model_weights.get("lm_head.bias")
        V, D = W.shape
        
        # Set default top_k if not provided
        if top_k is None:
            top_k = min(1024, max(32, V // 4))
        
        # For small vocabularies and no top-K requirement, use direct computation
        if V <= tile_size and not return_topk:
            return self._fgemm_fold(h_t, W.T.contiguous(), b)  # [V]
        
        # For large vocabularies or top-K requirement, use streaming top-K
        device = h_t.device
        dtype = h_t.dtype
        
        # Keep running top-K to avoid storing full logits
        running_values = torch.full((top_k,), float('-inf'), device=device, dtype=dtype)
        running_indices = torch.zeros(top_k, device=device, dtype=torch.long)
        
        # Process vocabulary in tiles
        for start_idx in range(0, V, tile_size):
            end_idx = min(start_idx + tile_size, V)
            
            # Compute logits for this tile
            W_tile = W[start_idx:end_idx].T.contiguous()  # [D, tile_size]
            b_tile = b[start_idx:end_idx] if b is not None else None
            tile_logits = self._fgemm_fold(h_t, W_tile, b_tile)  # [tile_size]
            
            # Merge with running top-K
            tile_indices = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
            
            # Combine current top-K with tile results
            combined_values = torch.cat([running_values, tile_logits])
            combined_indices = torch.cat([running_indices, tile_indices])
            
            # Keep top-K
            top_vals, top_pos = torch.topk(combined_values, k=top_k, largest=True)
            running_values = top_vals
            running_indices = combined_indices[top_pos]
        
        # Return top-K only or reconstruct full tensor
        if return_topk:
            return running_indices, running_values
        
        # Reconstruct full logits tensor (sparse)
        full_logits = torch.full((V,), float('-inf'), device=device, dtype=dtype)
        full_logits[running_indices] = running_values
        
        return full_logits

    def _allowed_theta_window(self, theta_now: float) -> tuple[float, float]:
        """Forward-only gating helper: return allowed theta range for monotone progression."""
        # CS→UNA→ONA→BU_IN→BU_EG progression (monotone)
        if theta_now < THETA_CS:
            return (THETA_CS, THETA_UNA)
        if theta_now < THETA_UNA:
            return (THETA_UNA, THETA_ONA)
        if theta_now < THETA_ONA:
            return (THETA_ONA, THETA_BU_IN)
        if theta_now < THETA_BU_IN:
            return (THETA_BU_IN, THETA_BU_EG)
        return (THETA_BU_EG, np.pi)  # closure band


def tensor_to_int_batch(tensors: np.ndarray) -> np.ndarray:
    """Vectorized conversion of batch of tensors to packed integers."""
    # tensors: (batch, 4, 2, 3, 2)
    batch_size = tensors.shape[0]
    result = np.zeros(batch_size, dtype=np.uint64)
    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        result |= ((tensors[:, layer, frame, row, col] == -1).astype(np.uint64)) << bit_pos
    return result

def int_to_tensor_batch(state_ints: np.ndarray) -> np.ndarray:
    """Vectorized conversion of batch of packed integers to tensors."""
    batch_size = state_ints.shape[0]
    tensors = np.ones((batch_size, 4, 2, 3, 2), dtype=np.int8)
    for bit_pos, (layer, frame, row, col) in enumerate(TENSOR_BIT_ORDER):
        mask = ((state_ints >> bit_pos) & 1).astype(bool)
        tensors[mask, layer, frame, row, col] = -1
    return tensors
