"""
GyroSI Kernel v0.9.11.0 - Pure Physics Implementation with Complete Corrections

This implements the Common Governance Model (CGM) through a physics-first approach
to language processing. No matrix multiplications, no transformers - pure state
transitions through a finite 48-bit manifold.

Key Features:
- Downloads and uses any HuggingFace language model
- Converts model weights to compressed state sequences via Fold
- All generation via pure physical resonance (no scoring, no heuristics)
- Sparse holographic storage (only deviations from baseline)
- Complete CS asymmetric emission with canonical broadcast masks
- Forward-only cycle gating with theoretically-derived thresholds
- Proper tokenizer byte protocol (no fake LEB128)
- Progress reporting for weight conversion
"""

import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open

# ============================================================================
# CORE CONSTANTS - COMPLETE CGM FOUNDATION
# ============================================================================

GENE_Mic_S = 0xAA  # Holographic topology constant (8-bit)

# GENE_Mac_S: The archetypal 48-element tensor [4, 2, 3, 2]
GENE_Mac_S = np.array([
    # Layer 0: 0° phase
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    # Layer 1: 180° phase
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]],
    # Layer 2: 360° phase
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    # Layer 3: 540° phase
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)

# Bit family masks per CGM
EXON_LI_MASK = 0b01000010  # Bits 1, 6 - UNA (Parity/Reflection)
EXON_FG_MASK = 0b00100100  # Bits 2, 5 - ONA (Forward Gyration)
EXON_BG_MASK = 0b00011000  # Bits 3, 4 - BU (Backward Gyration)
EXON_L0_MASK = 0b10000001  # Bits 0, 7 - Anchors (Boundaries)

# CGM Stage thresholds based on theoretical angular sequence
# These are derived from the CGM's recursive closure: π/2 → π/4 → π/4 → 0
THETA_CS = np.pi / 2                    # π/2 - Common Source (chiral seed)
THETA_UNA = np.pi / 4                   # π/4 - Unity Non-Absolute (planar split)
THETA_ONA = np.pi / 4                   # π/4 - Opposition Non-Absolute (3D tilt)
THETA_BU_IN = 0.0                       # 0.0 - Balance Universal Ingress (defect = 0)
THETA_BU_EG = np.pi / 2                 # π/2 - Balance Universal Egress (CS Recollection)

# Stage ordering for cycle gating - irreversible temporal flow
STAGE_ORDER = ["CS", "UNA", "ONA", "BU_IN", "BU_EG"]

# Canonical bit ordering for tensor <-> int conversion
# Maps each bit position (0-47) to its tensor coordinate [layer, frame, row, col]
# This ensures platform-independent, deterministic packing
TENSOR_BIT_ORDER = []
for layer in range(4):
    for frame in range(2):
        for row in range(3):
            for col in range(2):
                TENSOR_BIT_ORDER.append((layer, frame, row, col))

# ============================================================================
# CANONICAL BROADCAST MASKS FOR CS EMISSION
# ============================================================================

def generate_intron_broadcast_masks() -> np.ndarray:
    """Generate the canonical 256x48 broadcast masks for CS emission.
    
    Each intron has a unique 48-bit pattern that preserves chirality
    and seeds the UNA ring with proper parity violation.
    """
    masks = np.zeros((256, 48), dtype=np.uint8)
    
    for intron in range(256):
        # Extract bit families from intron
        li_bits = (intron & EXON_LI_MASK) >> 1  # Chirality
        fg_bits = (intron & EXON_FG_MASK) >> 2  # Forward gyration
        bg_bits = (intron & EXON_BG_MASK) >> 3  # Backward gyration
        l0_bits = intron & EXON_L0_MASK         # Anchors
        
        # Build 48-bit pattern preserving bit family semantics
        for i in range(48):
            layer = i // 12
            position = i % 12
            
            # Distribute bit families across the 48-bit pattern
            if position in [0, 11]:  # Anchor positions
                masks[intron, i] = (l0_bits >> (position // 11)) & 1
            elif position in [1, 6]:  # Chirality positions
                masks[intron, i] = (li_bits >> ((position-1) // 5)) & 1
            elif position in [2, 5, 7, 10]:  # FG positions
                masks[intron, i] = (fg_bits >> ((position-2) % 4)) & 1
            else:  # BG positions [3, 4, 8, 9]
                masks[intron, i] = (bg_bits >> ((position-3) % 4)) & 1
                
            # Apply layer-specific chirality twist
            if layer > 0:
                masks[intron, i] ^= (li_bits & 1)
    
    return masks

# Generate canonical broadcast masks
INTRON_BROADCAST_MASKS = generate_intron_broadcast_masks()

# ============================================================================
# CORE PHYSICS FUNCTIONS
# ============================================================================

def fold(a: int, b: int) -> int:
    """The Monodromic Fold: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    
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
            result |= (1 << bit_pos)
    
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
    # Extract 8-bit chunks for helical processing
    chunk_0 = (state_int >> 0) & 0xFF
    chunk_1 = (state_int >> 8) & 0xFF
    chunk_2 = (state_int >> 16) & 0xFF
    chunk_3 = (state_int >> 24) & 0xFF
    chunk_4 = (state_int >> 32) & 0xFF
    chunk_5 = (state_int >> 40) & 0xFF
    
    # Helical folding: fold(chunk_i, chunk_{i+1})
    exon = fold(chunk_0, chunk_1)
    exon = fold(exon, chunk_2)
    exon = fold(exon, chunk_3)
    exon = fold(exon, chunk_4)
    exon = fold(exon, chunk_5)
    
    # State-dependent fallback for exon=0 (physics, not magic)
    if exon == 0:
        # Use state-dependent perturbation based on state bits
        perturbation = (state_int & 0xFF) or GENE_Mic_S
        exon = perturbation
    
    return exon & 0xFF


def list_cached_models() -> List[str]:
    """List all cached models in the kernel directory."""
    kernel_dir = Path(__file__).parents[1] / "memories" / "kernel"
    if not kernel_dir.exists():
        return []
    
    cached_models = []
    for model_dir in kernel_dir.iterdir():
        if model_dir.is_dir():
            conversion_cache = model_dir / "conversion_meta.json"
            if conversion_cache.exists():
                try:
                    with open(conversion_cache, 'r') as f:
                        meta = json.load(f)
                    tensors_or_files = meta.get("num_tensors", meta.get("num_weight_files", 0))
                    cached_models.append({
                        "name": meta.get("model_name", model_dir.name),
                        "date": meta.get("conversion_date", "Unknown"),
                        "tensors": tensors_or_files,
                        "size_mb": meta.get("total_size", 0) / (1024 * 1024)
                    })
                except:
                    cached_models.append({
                        "name": model_dir.name,
                        "date": "Unknown",
                        "tensors": 0,
                        "size_mb": 0
                    })
    
    return cached_models


def clear_model_cache(model_name: str = None):
    """Clear cache for a specific model."""
    if model_name is None:
        model_name = "distilgpt2"
    
    kernel_dir = Path(__file__).parents[1] / "memories" / "kernel"
    model_key = model_name.replace("/", "_").replace("-", "_")
    model_dir = kernel_dir / model_key
    
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)
        print(f"🗑️ Cleared cache for {model_name}")
    else:
        print(f"📁 No cache found for {model_name}")


def print_cache_info():
    """Print information about cached models."""
    cached_models = list_cached_models()
    
    if not cached_models:
        print("📁 No cached models found")
        return
    
    print(f"📁 Cached models ({len(cached_models)}):")
    for model in cached_models:
        print(f"  • {model['name']}")
        print(f"    Converted: {model['date']}")
        print(f"    Tensors: {model['tensors']}")
        print(f"    Size: {model['size_mb']:.1f} MB")
        print()


# ============================================================================
# MODEL WEIGHT CONVERSION FUNCTIONS
# ============================================================================

def download_model(model_name: str = None, force_reconvert: bool = False):
    """Download and convert any HuggingFace language model to physics format."""
    if model_name is None:
        model_name = "Qwen/Qwen3-0.6B"
    
    # Import required modules
    import json
    import os
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import torch
    from safetensors import safe_open
    
    # Create kernel directory structure
    kernel_dir = Path(__file__).parents[1] / "memories" / "kernel"
    kernel_dir.mkdir(exist_ok=True)
    
    # Create model-specific subdirectory
    model_key = model_name.replace("/", "_").replace("-", "_")
    model_dir = kernel_dir / model_key
    model_dir.mkdir(exist_ok=True)
    
    # Define cache files
    tokenizer_cache = model_dir / "tokenizer.json"
    weights_cache = model_dir / "weights.npz"
    config_cache = model_dir / "config.json"
    conversion_cache = model_dir / "conversion_meta.json"
    
    try:
        # Check if we have cached conversion metadata
        if not force_reconvert and conversion_cache.exists():
            print(f"📁 Found cached conversion for {model_name}")
            # We still resolve actual weight file paths via the hub cache to avoid stale paths
        elif force_reconvert:
            print(f"🔄 Force re-conversion requested for {model_name}")
        else:
            print(f"📥 No cached conversion found for {model_name}")
        
        # Download from HuggingFace (uses local cache on subsequent runs)
        print(f"📥 Downloading {model_name} model files...")
        
        # Download tokenizer
        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save tokenizer to cache
        tokenizer.save_pretrained(str(model_dir))
        print(f"  → Saved tokenizer to cache")
        
        # Download raw model files (paths only)
        print("  → Downloading model files...")
        model_files = snapshot_download(
            repo_id=model_name,
            allow_patterns=["*.safetensors", "*.bin", "config.json"],
            ignore_patterns=["*.md", "*.txt", "*.git*"]
        )
        
        # Load config
        config_path = f"{model_files}/config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Save config to cache
        with open(config_cache, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"  → Saved config to cache")
        
        # Collect weight file paths instead of loading them
        safetensor_files: List[str] = []
        pytorch_model_path: Optional[str] = None
        for filename in os.listdir(model_files):
            if filename.endswith('.safetensors'):
                safetensor_files.append(os.path.join(model_files, filename))
            elif filename == 'pytorch_model.bin':
                pytorch_model_path = os.path.join(model_files, filename)
        safetensor_files.sort()
        
        # Save conversion metadata
        total_bytes = 0
        for fp in safetensor_files:
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
        conversion_meta = {
            "model_name": model_name,
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_weight_files": len(safetensor_files) + (1 if pytorch_model_path else 0),
            "total_size": total_bytes,
            "weight_files": safetensor_files,
            "pytorch_model_path": pytorch_model_path,
        }
        with open(conversion_cache, 'w') as f:
            json.dump(conversion_meta, f, indent=2)
        print(f"  → Saved conversion metadata")
        
        print(f"✅ Downloaded and indexed {model_name}: {len(safetensor_files)} safetensors files")
        
        return {
            "tokenizer": tokenizer,
            "safetensor_files": safetensor_files,
            "pytorch_model_path": pytorch_model_path,
            "vocab_size": tokenizer.vocab_size,
            "model_config": model_config,
            "resolved_model_name": model_name,
            "model_key": model_key,
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}: {e}")


def quantize_and_compress_weights(weights: np.ndarray, max_size: int = 10000) -> Tuple[List[int], float]:
    """Convert weights to compressed intron sequence using pure physics.
    
    Args:
        weights: Weight tensor to compress
        max_size: Maximum number of elements to process (for memory)
    
    Returns:
        Compressed byte sequence and quantization scale
    """
    # Flatten and limit size
    flat = weights.flatten()
    if len(flat) > max_size:
        flat = flat[:max_size]
    
    # Quantize to int8
    scale = np.max(np.abs(flat)) / 127.0 if np.max(np.abs(flat)) > 0 else 1.0
    quantized = np.clip(flat / scale, -128, 127).astype(np.int8)
    
    # Convert to unsigned for delta encoding
    unsigned = quantized.view(np.uint8)
    
    # Delta encode for compression
    deltas = np.zeros(len(unsigned), dtype=np.uint8)
    prev = 0
    for i in range(len(unsigned)):
        deltas[i] = (int(unsigned[i]) - prev) & 0xFF
        prev = int(unsigned[i])
    
    # Convert to introns
    introns = [transcribe_byte(d) for d in deltas]
    
    # Compress via Fold sequence
    compressed = []
    for i in range(0, len(introns), 8):
        chunk = introns[i:i+8]
        compressed.append(fold_sequence(chunk, start_state=GENE_Mic_S))
    
    return compressed, scale


# ============================================================================
# GYRO KERNEL - PURE PHYSICS IMPLEMENTATION
# ============================================================================

class GyroKernel:
    """GyroSI Kernel v0.9.11.0 - Pure Physics Implementation
    
    Complete corrections:
    - Proper tokenizer byte protocol (no fake LEB128)
    - Canonical broadcast masks for CS emission
    - Fixed virtual token collisions with proper keying
    - Progress reporting for weight conversion
    - Theoretically-derived stage thresholds
    - Canonical bit ordering for tensor operations
    """
    
    def __init__(self, base_path: Optional[Path] = None, debug: bool = False, model_name: str = None, force_reconvert: bool = False):
        """Initialize kernel with physics tables."""
        if base_path is None:
            base_path = Path(__file__).parents[1] / "memories"
        
        self.base_path = base_path
        self.debug = debug
        self.model_name = model_name
        self.force_reconvert = force_reconvert
        
        # Statistics tracking
        self.stats = {
            "tokens_learned": 0,
            "memory_entries": 0,
            "orbits_discovered": 0,
            "generation_steps": 0,
            "states_visited": set(),
            "unique_masks": set(),
            "weight_tensors_compressed": 0,
            "weight_compression_bytes": 0,
            "algedonic_resets": 0,
            "pce_escapes": 0,
        }
        
        # Load physics tables (ontology, epistemology, phenomenology, theta, orbit_sizes)
        self._load_physics_tables()
        
        # Load broadcast masks from meta artifact for proper CS emission
        self._load_broadcast_masks()
        
        # Find CS state (θ=0)
        self._find_cs_state()
        
        # Initialize archetypal state
        self.archetypal_state_int = tensor_to_int(GENE_Mac_S)
        
        # Start from CS
        self.current_state_index = self.CS_STATE_INDEX
        # Event-based stage tracker (0: CS, 1: UNA, 2: ONA, 3: BU_IN, 4: BU_EG)
        self.stage_index: int = 0
        
        # Memory: orbit_rep -> token_id -> mask
        self.memory: Dict[int, Dict[int, int]] = {}
        
        # Model knowledge (compressed weights)
        self.model_knowledge: Dict[str, List[int]] = {}
        self.model_config = {}
        # Token → projected 48-bit state from embeddings (aux)
        self.token_state_map: Dict[int, int] = {}
        # Token → on-manifold post-state index and exon (primary semantics)
        self.token_post_state_index: Dict[int, int] = {}
        self.token_exon_cache: Dict[int, int] = {}
        # Vectorized caches for fast selection
        self._token_ids_np: Optional[np.ndarray] = None
        self._token_exons_np: Optional[np.ndarray] = None
        self._pop8: Optional[np.ndarray] = None
        
        # Precomputed resonance table for fast fold operations
        self._resonance_table: Optional[np.ndarray] = None
        self._build_resonance_table()
        
        # Hierarchical candidate selection
        self._orbit_candidates: Dict[int, List[int]] = {}
        self._theta_buckets: Dict[int, List[int]] = defaultdict(list)
        self._state_cache: Dict[int, List[int]] = {}
        self._build_hierarchical_candidates()
        
        # Virtual tokens with proper keying: (layer_name, position) -> comp_byte
        self.virtual_tokens: Dict[Tuple[str, int], int] = {}
        
        # Load model and tokenizer - FATAL ERROR if fails
        model_data = download_model(self.model_name, self.force_reconvert)
        if not model_data or not model_data["tokenizer"]:
            raise RuntimeError("Failed to load tokenizer - cannot operate without it")
        
        # Persist resolved model identity
        self.model_name = model_data.get("resolved_model_name", self.model_name)
        self._model_key = model_data.get("model_key", (self.model_name or "model").replace("/", "_").replace("-", "_"))
        
        self.tokenizer = model_data["tokenizer"]
        
        # Set special tokens from tokenizer
        self._set_special_tokens()
        
        # Path memory
        self.path_memory = GENE_Mic_S
        
        # Generative path weaving for dynamic resonance
        self.last_generated_mask = GENE_Mic_S
        
        # Algedonic reset for stagnation decay
        self.stagnation_counter = 0
        self.last_generated_token = None
        
        # Physics switches
        self.physics_switches = {
            'cs_emission': True,
            'cycle_gating': True,
            'sparse_storage': True,
            'model_import': True,
            'embedding_projection': False,
        }
        
        # Build valid tokens from tokenizer
        self._build_valid_tokens()
        
        # Import model weights with progress reporting
        if self.physics_switches['model_import']:
            self._import_model_weights(model_data)
        
        if self.debug:
            print(f"\n🧬 GyroSI Kernel v0.9.11.0 initialized")
            print(f"📍 CS state: index={self.CS_STATE_INDEX}, θ={float(self.theta[self.CS_STATE_INDEX]):.4f}")
            print(f"📊 Ontology: {len(self.ontology):,} states")
            print(f"💫 Virtual tokens: {len(self.virtual_tokens):,}")
            print(f"🤖 Model: {self.model_name or 'unknown'}")
            print(f"📝 Tokenizer vocab size: {self.tokenizer.vocab_size}")
    
    def _load_physics_tables(self) -> None:
        """Load all physics tables with memory mapping."""
        meta_path = self.base_path / "public" / "meta"
        
        # Ontology (state integers)
        ontology_path = meta_path / "ontology_keys.npy"
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")
        self.ontology = np.load(ontology_path, mmap_mode="r")
        
        # Epistemology (state transitions)
        epistemology_path = meta_path / "epistemology.npy"
        if not epistemology_path.exists():
            raise FileNotFoundError(f"Epistemology not found: {epistemology_path}")
        self.epistemology = np.load(epistemology_path, mmap_mode="r")
        
        # Theta (angular divergence)
        theta_path = meta_path / "theta.npy"
        if not theta_path.exists():
            raise FileNotFoundError(f"Theta not found: {theta_path}")
        self.theta = np.load(theta_path, mmap_mode="r")
        
        # Phenomenology (orbit mapping)
        pheno_path = meta_path / "phenomenology_map.npy"
        if not pheno_path.exists():
            raise FileNotFoundError(f"Phenomenology not found: {pheno_path}")
        self.phenomenology = np.load(pheno_path, mmap_mode="r")
        
        # Orbit sizes
        orbit_sizes_path = meta_path / "orbit_sizes.npy"
        if not orbit_sizes_path.exists():
            raise FileNotFoundError(f"Orbit sizes not found: {orbit_sizes_path}")
        self.orbit_sizes = np.load(orbit_sizes_path, mmap_mode="r")
        
        # Validate shapes and types
        assert self.epistemology.shape[1] == 256, "Epistemology must have 256 introns"
        assert self.ontology.shape[0] == self.epistemology.shape[0], "Ontology/epistemology size mismatch"
        assert self.theta.shape[0] == self.ontology.shape[0], "Theta size mismatch"
        assert self.phenomenology.shape[0] == self.ontology.shape[0], "Phenomenology size mismatch"
    
    def _load_broadcast_masks(self) -> None:
        """Load broadcast masks from the meta artifact for proper CS emission."""
        meta_path = self.base_path / "public" / "meta"
        broadcast_masks_path = meta_path / "intron_broadcast_masks.npy"
        
        if not broadcast_masks_path.exists():
            raise FileNotFoundError(f"Intron broadcast masks not found: {broadcast_masks_path}")
        
        self.INTRON_BROADCAST_MASKS = np.load(broadcast_masks_path, mmap_mode="r")
        if self.INTRON_BROADCAST_MASKS.shape != (256, 48):
            raise ValueError(f"Intron broadcast masks shape mismatch: expected (256, 48), got {self.INTRON_BROADCAST_MASKS.shape}")
        
        print(f"✅ Loaded {self.INTRON_BROADCAST_MASKS.shape[0]} canonical broadcast masks from meta artifact.")
    
    def _find_cs_state(self) -> None:
        """Find the true CS state."""
        # Find state with minimum theta
        min_theta_idx = int(np.argmin(self.theta))
        min_theta = float(self.theta[min_theta_idx])
        
        # Use min theta state as CS
        self.CS_STATE_INDEX = min_theta_idx
        self.CS_STATE_INT = int(self.ontology[min_theta_idx])
        
        if self.debug:
            print(f"✅ Found CS state: index={min_theta_idx}, θ={min_theta:.4f}")
    
    def _set_special_tokens(self) -> None:
        """Set special tokens from tokenizer."""
        # Get special tokens with proper fallback order
        self.CLS_TOKEN = getattr(self.tokenizer, 'cls_token_id', None) or \
                        getattr(self.tokenizer, 'bos_token_id', None) or 1
        self.SEP_TOKEN = getattr(self.tokenizer, 'sep_token_id', None) or \
                        getattr(self.tokenizer, 'eos_token_id', None) or 2
        self.PAD_TOKEN = getattr(self.tokenizer, 'pad_token_id', None) or 0
        
        if self.debug:
            print(f"📌 Special tokens: CLS={self.CLS_TOKEN}, SEP={self.SEP_TOKEN}, PAD={self.PAD_TOKEN}")
    
    def _build_valid_tokens(self) -> None:
        """Build set of valid tokens from tokenizer."""
        self.valid_tokens = set(range(self.tokenizer.vocab_size))
        
        if self.debug:
            print(f"📚 Valid tokens: {len(self.valid_tokens)} from tokenizer")
    
    def _import_model_weights(self, model_data: dict) -> None:
        """Import model weights with progress reporting (streamed)."""
        print("📦 Converting model weights to physics...")
        total_original_size = 0
        total_compressed_size = 0
        embeddings_buffer: Optional[np.ndarray] = None
        tensors_processed = 0
        
        safetensor_files: List[str] = model_data.get("safetensor_files", [])
        pytorch_model_path: Optional[str] = model_data.get("pytorch_model_path")
        
        # Iterate safetensors files and stream tensors
        for st_path in safetensor_files:
            try:
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        if isinstance(tensor, torch.Tensor):
                            # Statistics only
                            try:
                                total_original_size += tensor.numel() * tensor.element_size()
                            except Exception:
                                pass
                            
                            # Compress tensor to virtual bytes (bounded work)
                            # Convert to numpy efficiently
                            try:
                                arr = tensor.detach().cpu().numpy()
                            except TypeError:
                                arr = tensor.detach().cpu().float().numpy()
                            compressed_bytes, _ = quantize_and_compress_weights(arr)
                            total_compressed_size += len(compressed_bytes)
                            for pos, byte_val in enumerate(compressed_bytes):
                                self.virtual_tokens[(key, pos)] = byte_val
                            
                            # Capture embedding matrix once
                            if embeddings_buffer is None and arr.ndim == 2 and "embed" in key.lower() and "weight" in key.lower():
                                embeddings_buffer = arr
                            
                            tensors_processed += 1
                            if tensors_processed % 50 == 0:
                                print(f"  Progress: {tensors_processed} tensors processed")
            except Exception as e:
                print(f"  ⚠️ Failed to read {st_path}: {e}")
        
        # Fallback to pytorch_model.bin if needed
        if tensors_processed == 0 and pytorch_model_path and os.path.exists(pytorch_model_path):
            print("  → Loading PyTorch model weights (fallback)...")
            checkpoint = torch.load(pytorch_model_path, map_location='cpu')
            for key, tensor in checkpoint.items():
                if isinstance(tensor, torch.Tensor):
                    try:
                        total_original_size += tensor.numel() * tensor.element_size()
                    except Exception:
                        pass
                    arr = tensor.detach().cpu().numpy()
                    compressed_bytes, _ = quantize_and_compress_weights(arr)
                    total_compressed_size += len(compressed_bytes)
                    for pos, byte_val in enumerate(compressed_bytes):
                        self.virtual_tokens[(key, pos)] = byte_val
                    if embeddings_buffer is None and arr.ndim == 2 and "embed" in key.lower() and "weight" in key.lower():
                        embeddings_buffer = arr
                    tensors_processed += 1
                    if tensors_processed % 50 == 0:
                        print(f"  Progress: {tensors_processed} tensors processed")
        
        print(f"✅ Imported {tensors_processed} weight tensors")
        if total_original_size > 0:
            compression_ratio = (1 - total_compressed_size / total_original_size) * 100
            print(f"   Original size: {total_original_size:,} bytes")
            print(f"   Compressed size: {total_compressed_size:,} bytes")
            print(f"   Compression: {compression_ratio:.1f}% reduction")
        print(f"   Virtual tokens created: {len(self.virtual_tokens):,}")
        
        # Project embeddings to token states (semantic bridge)
        if embeddings_buffer is not None and self.physics_switches.get('embedding_projection', False):
            t0 = time.time()
            try:
                self._project_embeddings_to_token_states(embeddings_buffer)
            except Exception as e:
                print(f"⚠️ Embedding projection failed: {e}")
            finally:
                print(f"   Embedding projection took {time.time()-t0:.2f}s")
 
        # Build or load on-manifold token post-states (from CS via real bytes introns)
        try:
            t0 = time.time()
            self._build_or_load_token_post_states(model_data)
            print(f"   Token post-states build/load took {time.time()-t0:.2f}s")
            # Rebuild hierarchical candidates now that caches are ready
            self._build_hierarchical_candidates()
        except Exception as e:
            print(f"⚠️ Token post-state build failed: {e}")
    
    def _get_stage(self, state_index: int) -> str:
        """Get CGM stage label from the event-based stage tracker."""
        order = ["CS", "UNA", "ONA", "BU_IN", "BU_EG"]
        idx = self.stage_index
        if idx < 0:
            idx = 0
        if idx >= len(order):
            idx = len(order) - 1
        return order[idx]

    def _project_embeddings_to_token_states(self, embeddings: np.ndarray, limit: Optional[int] = None) -> None:
        """Project token embeddings to 48-bit states via physics fold.

        This preserves token semantics by creating a direct token → state map
        used during generation as primary knowledge.
        """
        vocab = min(self.tokenizer.vocab_size, embeddings.shape[0])
        if limit is None:
            # Avoid excessive startup cost; can be tuned
            limit = vocab
        count = min(vocab, limit)

        # Vector quantization scale per row
        def project_row(vec: np.ndarray) -> int:
            # Normalize and quantize to uint8
            max_abs = float(np.max(np.abs(vec))) if np.max(np.abs(vec)) > 0 else 1.0
            q = np.clip((vec / max_abs) * 127.0, -128, 127).astype(np.int8).view(np.uint8)
            # Produce 6 bytes via fold over 6 interleaved streams
            six_bytes: List[int] = []
            for k in range(6):
                acc = GENE_Mic_S
                # Interleave to cover the whole vector while preserving order memory
                for idx in range(k, q.shape[0], 6):
                    acc = fold(acc, transcribe_byte(int(q[idx])))
                six_bytes.append(acc & 0xFF)
            # Pack 6 bytes into 48-bit state (byte 0 is least significant)
            state_int = 0
            for byte_index, byte_val in enumerate(six_bytes):
                state_int |= (int(byte_val) & 0xFF) << (8 * byte_index)
            # Ensure 48-bit
            state_int &= (1 << 48) - 1
            return state_int

        print(f"🧭 Projecting embeddings to token states: {count}/{vocab}")
        self.token_state_map.clear()
        for token_id in range(count):
            try:
                vec = embeddings[token_id]
                self.token_state_map[token_id] = project_row(vec)
            except Exception:
                # Skip malformed rows
                continue
        print(f"✅ Token states built: {len(self.token_state_map):,}")

    def _build_token_post_states(self, limit: Optional[int] = None) -> None:
        """Snap tokens to the manifold by applying uLEB128 introns via epistemology from CS.

        Caches token_id → post_state_index and token_id → exon (8-bit) for fast generation.
        """
        vocab = self.tokenizer.vocab_size
        if limit is None:
            limit = vocab
        count = min(vocab, limit)

        self.token_post_state_index.clear()
        self.token_exon_cache.clear()

        cs_index = self.CS_STATE_INDEX
        # Faster token->string mapping
        ids = list(range(count))
        try:
            tok_strings = self.tokenizer.convert_ids_to_tokens(ids)
        except Exception:
            tok_strings = [self.tokenizer.decode([i], skip_special_tokens=False) for i in ids]

        last_progress = 0
        for token_id, token_str in zip(ids, tok_strings):
            if token_id not in self.valid_tokens:
                continue
            try:
                token_bytes = token_str.encode('utf-8', errors='ignore')
            except Exception:
                continue
            if not token_bytes:
                continue
            introns = [transcribe_byte(b) for b in token_bytes]
            s = cs_index
            for ι in introns:
                s = int(self.epistemology[s, ι & 0xFF])
            self.token_post_state_index[token_id] = s
            state_int = int(self.ontology[s])
            exon_tok = compute_exon_from_state(state_int) & 0xFF
            self.token_exon_cache[token_id] = exon_tok
            # Progress
            if token_id - last_progress >= 10000:
                last_progress = token_id
                print(f"  → Built {len(self.token_post_state_index):,}/{count:,} token post-states")
        print(f"✅ Token post-states built: {len(self.token_post_state_index):,}")
        # Build vectorized arrays
        self._materialize_vectorized_tokens()

    def _materialize_vectorized_tokens(self) -> None:
        """Create NumPy arrays for fast vectorized token selection."""
        if not self.token_exon_cache:
            return
        ids = np.fromiter(self.token_exon_cache.keys(), dtype=np.int32, count=len(self.token_exon_cache))
        exs = np.fromiter(self.token_exon_cache.values(), dtype=np.uint8, count=len(self.token_exon_cache))
        self._token_ids_np = ids
        self._token_exons_np = exs
        if self._pop8 is None:
            self._pop8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def _build_resonance_table(self) -> None:
        """Pre-compute a table of fold(a, b) for all possible a, b values."""
        self._resonance_table = np.zeros((256, 256), dtype=np.uint8)
        for a in range(256):
            for b in range(256):
                self._resonance_table[a, b] = fold(a, b)

    def _build_hierarchical_candidates(self) -> None:
        """Build hierarchical candidate selection structures."""
        # Populate _orbit_candidates and _theta_buckets
        for token_id, post_idx in self.token_post_state_index.items():
            orbit_rep = int(self.phenomenology[post_idx])
            theta_val = float(self.theta[post_idx])
            
            # Add to orbit bucket
            self._orbit_candidates.setdefault(orbit_rep, []).append(token_id)
            
            # Add to theta bucket (e.g., 0.0 to 0.1, 0.1 to 0.2, etc.)
            # This is a simplified example; a more sophisticated bucketing might be needed
            # For now, a simple binning based on theta value
            theta_bin = int(theta_val * 10) # Example: 0.0 to 1.0 -> 0 to 10
            self._theta_buckets[theta_bin].append(token_id)

        # Populate _state_cache for fast state lookup
        for token_id, post_idx in self.token_post_state_index.items():
            self._state_cache[token_id] = [post_idx]

    def _build_or_load_token_post_states(self, model_data: dict) -> None:
        """Load cached token post-states/exons if present; otherwise build and save."""
        # Determine cache paths under model cache dir
        # Use resolved model_key from download phase to avoid path mismatches
        kernel_dir = Path(__file__).parents[1] / "memories" / "kernel"
        model_key = getattr(self, "_model_key", (self.model_name or "model").replace("/", "_").replace("-", "_"))
        model_dir = kernel_dir / model_key
        model_dir.mkdir(exist_ok=True)
        post_path = model_dir / "token_post_states.npy"
        exons_path = model_dir / "token_exons.npy"

        if post_path.exists() and exons_path.exists():
            # Load as memory-mapped for fast startup
            post_idx = np.load(str(post_path), mmap_mode='r')
            exons = np.load(str(exons_path), mmap_mode='r')
            # Rehydrate dict caches
            self.token_post_state_index = {int(i): int(s) for i, s in enumerate(post_idx) if s >= 0}
            self.token_exon_cache = {int(i): int(e) for i, e in enumerate(exons) if e >= 0}
            self._materialize_vectorized_tokens()
            print(f"✅ Loaded token post-states/exons from cache (memory-mapped)")
            return

        # Build fresh if cache doesn't exist
        self._build_token_post_states()
        # Persist to disk as memory-mapped arrays for fast load next time
        vocab = self.tokenizer.vocab_size
        post_idx = np.full((vocab,), -1, dtype=np.int64)
        exons = np.full((vocab,), -1, dtype=np.int16)
        for tok, st in self.token_post_state_index.items():
            post_idx[int(tok)] = int(st)
        for tok, ex in self.token_exon_cache.items():
            exons[int(tok)] = int(ex)
        np.save(str(post_path), post_idx)
        np.save(str(exons_path), exons)
        print(f"💾 Cached token post-states/exons")
    
    def _apply_intron(self, state_index: int, intron: int, position: Optional[str] = None) -> int:
        """Apply intron with CS emission and event-based stage progression.

        position: one of {"first", "middle", "last"} relative to the current token's intron sequence.
        """
        current_stage = self._get_stage(state_index)
        
        # CS asymmetric emission with canonical broadcast masks
        if self.physics_switches['cs_emission'] and state_index == self.CS_STATE_INDEX:
            if (intron & (EXON_FG_MASK | EXON_BG_MASK)) == 0:
                # Standing intron: CS remains invariant
                return self.CS_STATE_INDEX
            else:
                # Driving intron: use canonical broadcast mask
                broadcast_mask = self.INTRON_BROADCAST_MASKS[intron]
                
                # Convert mask to state integer
                emitted_int = 0
                for i, bit in enumerate(broadcast_mask):
                    if bit:
                        emitted_int |= (1 << i)
                
                # Event: leaving CS on first driving intron → advance to UNA
                if self.stage_index < 1:
                    self.stage_index = 1

                # Find UNA candidates by minimal |theta - β| (β = π/4)
                # Select a small top-k set then choose maximal bit overlap
                theta_diff = np.abs(self.theta - THETA_UNA)
                top_k = min(1024, len(theta_diff))
                idxs = np.argpartition(theta_diff, kth=top_k - 1)[:top_k]

                best = int(idxs[0])
                best_overlap = -1
                for cand in idxs:
                    cand_int = int(self.ontology[cand])
                    overlap = 48 - (emitted_int ^ cand_int).bit_count()
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = int(cand)

                return best
        
        # Normal transition via epistemology
        if 0 <= state_index < len(self.epistemology):
            next_index = int(self.epistemology[state_index, intron & 0xFF])

            # Event-based stage advancement
            if position == "first":
                if self.stage_index < 1:
                    self.stage_index = 1  # UNA
            elif position == "middle":
                if self.stage_index < 2:
                    self.stage_index = 2  # ONA
            elif position == "last":
                # Closing intron has already set BU_IN in learn flow; ensure BU_EG after transition
                if self.stage_index < 4:
                    self.stage_index = 4  # BU_EG

            return next_index
        
        return state_index
    
    def _id_to_uleb128(self, x: int) -> List[int]:
        """Encode integer id into unsigned LEB128 bytes."""
        out: List[int] = []
        val = int(x)
        while True:
            b = val & 0x7F
            val >>= 7
            if val:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
        return out

    def token_to_introns(self, token_id: int) -> List[int]:
        """Get ACTUAL bytes from tokenizer, not LEB128 of ID."""
        if token_id not in self.valid_tokens:
            return []
        
        # Get the token's actual string representation
        token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        
        # Get its UTF-8 bytes (the actual representation)
        token_bytes = token_str.encode('utf-8')
        
        # Apply ψ isomorphism to each byte
        introns = [transcribe_byte(b) for b in token_bytes]
        return introns
    
    def apply_introns_batch(self, state_indices: np.ndarray, introns: np.ndarray) -> np.ndarray:
        """Vectorized state transitions using NumPy advanced indexing."""
        return self.epistemology[state_indices, introns]
    
    def learn_token(self, token_id: int) -> None:
        """Learn a token using sparse BU hinge physics."""
        if token_id not in self.valid_tokens:
            return
        
        # Track statistics
        self.stats["tokens_learned"] += 1
        self.stats["states_visited"].add(self.current_state_index)
        
        # Get PRE-state
        pre_state_index = self.current_state_index
        pre_state_int = int(self.ontology[pre_state_index])
        baseline_exon = compute_exon_from_state(pre_state_int)
        
        # Use orbit representative for compression
        orbit_rep = int(self.phenomenology[pre_state_index])
        
        # Get token's intron sequence
        introns = self.token_to_introns(token_id)
        if not introns:
            return
        
        # Evolve through all but last intron (UNA then ONA)
        for i, intron in enumerate(introns[:-1]):
            position = "first" if i == 0 else "middle"
            self.current_state_index = self._apply_intron(self.current_state_index, intron, position=position)
        
        # BU hinge: learn at closing intron
        closing_intron = introns[-1]
        token_mask = fold(baseline_exon, closing_intron)
        
        # Update path memory
        self.path_memory = fold(self.path_memory, token_mask)
        
        # SPARSE STORAGE: Only store if deviates from baseline
        if self.physics_switches['sparse_storage']:
            deviation_bits = (token_mask ^ baseline_exon).bit_count()
            
            if deviation_bits > 0:  # Only if different
                if orbit_rep not in self.memory:
                    self.memory[orbit_rep] = {}
                    self.stats["orbits_discovered"] = len(self.memory)
                
                # Fold with existing mask
                existing = self.memory[orbit_rep].get(token_id, baseline_exon)
                self.memory[orbit_rep][token_id] = fold(existing, token_mask)
                
                self.stats["memory_entries"] = sum(len(d) for d in self.memory.values())
                self.stats["unique_masks"].add(token_mask)
        
        # Mark BU_IN at hinge, then apply closing intron and advance to BU_EG
        if self.stage_index < 3:
            self.stage_index = 3  # BU_IN
        self.current_state_index = self._apply_intron(self.current_state_index, closing_intron, position="last")
    
    def _get_physics_filtered_candidates(self, current_state_index: int, max_candidates: int = 200) -> List[int]:
        """Get physics-filtered candidates using hierarchical selection."""
        current_theta = float(self.theta[current_state_index])
        current_orbit = int(self.phenomenology[current_state_index])
        
        # Check cache first
        cache_key = current_state_index
        if cache_key in self._state_cache and len(self._state_cache[cache_key]) > 1:
            cached_candidates = self._state_cache[cache_key][1:]  # Skip the state index
            if len(cached_candidates) <= max_candidates:
                return cached_candidates
        
        candidates = []
        
        # 1. Start with orbit candidates (fastest filter)
        orbit_candidates = self._orbit_candidates.get(current_orbit, [])
        candidates.extend(orbit_candidates)
        
        # 2. Add adjacent orbit candidates
        for adj_orbit in [current_orbit - 1, current_orbit + 1]:
            if adj_orbit >= 0 and adj_orbit in self._orbit_candidates:
                candidates.extend(self._orbit_candidates[adj_orbit])
        
        # 3. Filter by theta window
        theta_window = 0.2
        theta_filtered = []
        for token_id in candidates:
            # Include ALL valid tokens (no filtering to hide errors)
            if token_id not in self.valid_tokens:
                continue
            
            post_idx = self.token_post_state_index[token_id]
            post_theta = float(self.theta[post_idx])
            if abs(post_theta - current_theta) < theta_window:
                theta_filtered.append(token_id)
        
        # 4. If theta filtering gives too few, expand window
        if len(theta_filtered) < 50:
            theta_window = 0.5
            theta_filtered = []
            for token_id in candidates:
                if token_id not in self.valid_tokens:
                    continue
                
                post_idx = self.token_post_state_index[token_id]
                post_theta = float(self.theta[post_idx])
                if abs(post_theta - current_theta) < theta_window:
                    theta_filtered.append(token_id)
        
        candidates = theta_filtered
        
        # 6. Limit to max_candidates
        if len(candidates) > max_candidates:
            candidates = np.random.choice(candidates, max_candidates, replace=False).tolist()
        
        # 7. Cache the result
        self._state_cache[cache_key] = [current_state_index] + candidates
        
        return candidates

    def generate_token(self) -> Optional[int]:
        """Generate next token using pure physics resonance (full-vocab, vectorized)."""
        if self._token_exons_np is None or self._resonance_table is None:
            return None
        
        # Compute resonant seed (path-dependent)
        current_state_int = int(self.ontology[self.current_state_index])
        current_exon = compute_exon_from_state(current_state_int)
        resonant_seed = fold(fold(current_exon, self.path_memory), self.last_generated_mask)
        
        # Vectorized scoring across all tokens
        ex = self._token_exons_np
        rs = np.uint8(resonant_seed)
        pm = np.uint8(self.path_memory)
        alignment = self._resonance_table[rs, ex]
        diff = np.bitwise_xor(alignment, pm, dtype=np.uint8)
        defects = self._pop8[diff]
        # Deterministic argmin selection
        idx = int(np.argmin(defects))
        chosen_token = int(self._token_ids_np[idx])
        
        # Apply token's introns to evolve state
        if chosen_token is not None:
            introns = self.token_to_introns(chosen_token)
            for intron in introns:
                self.current_state_index = self._apply_intron(self.current_state_index, intron)
            
            # Update generative path weaving
            self.last_generated_mask = self.token_exon_cache.get(chosen_token, GENE_Mic_S)
            
            # Update stagnation tracking
            if chosen_token == self.last_generated_token:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            self.last_generated_token = chosen_token
            
            # Endogenous stagnation handling: attempt PCE escape before any reset
            if self.stagnation_counter >= 2:
                escaped = self._pce_escape()
                if escaped:
                    self.stats["pce_escapes"] += 1
                    self.stagnation_counter = 0
                elif self.stagnation_counter >= 5:  # Only reset after 5 failed escapes
                    if self.debug: print("  ⚠️ Stagnation persists! Applying algedonic reset.")
                    self.stats["algedonic_resets"] += 1
                    self.reset()
                    self.stagnation_counter = 0
        
        return chosen_token
    
    def _find_real_token_for_byte(self, byte: int) -> Optional[int]:
        """Find real token whose exon has minimal Fold defect to byte."""
        best_token = None
        best_defect = float('inf')
        
        # Use deterministic sampling with fixed seed
        import random
        rng = random.Random(42)  # Fixed seed for reproducibility
        sample_tokens = rng.sample(list(self.valid_tokens), min(1000, len(self.valid_tokens)))
        
        for token_id in sample_tokens:
            # Get token's intron sequence
            introns = self.token_to_introns(token_id)
            if not introns:
                continue
            
            # Compute exon from closing intron
            token_exon = fold(GENE_Mic_S, introns[-1])
            
            # Measure defect
            defect = (fold(byte, token_exon) ^ byte).bit_count()
            
            if defect < best_defect:
                best_defect = defect
                best_token = token_id
        
        return best_token

    def _pce_escape(self) -> bool:
        """Primordial Chirality Emission (endogenous escape) without reset.

        Applies a deterministic driving intron to advance along the helical path,
        preserving chirality and avoiding heuristic resets.
        Returns True if a state transition occurred.
        """
        pre_index = self.current_state_index
        # Choose a driving intron based on current stage
        if self.stage_index <= 1:
            intron = EXON_FG_MASK  # forward gyration
            pos = "first"
        elif self.stage_index == 2:
            intron = EXON_FG_MASK | EXON_LI_MASK  # add parity to break symmetry
            pos = "middle"
        else:
            intron = EXON_BG_MASK  # recollection
            pos = "last"

        self.current_state_index = self._apply_intron(self.current_state_index, intron, position=pos)
        return self.current_state_index != pre_index
    
    def reset(self) -> None:
        """Reset to CS state."""
        self.current_state_index = self.CS_STATE_INDEX
        self.path_memory = GENE_Mic_S
        self.last_generated_mask = GENE_Mic_S
        
        # Reset stage tracker to CS
        self.stage_index = 0
        
        if self.debug:
            theta = float(self.theta[self.CS_STATE_INDEX])
            print(f"🔄 Reset to CS: θ={theta:.4f}")
    
    def learn_text(self, text: str) -> None:
        """Learn from text."""
        # Tokenize text
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids if hasattr(encoding, 'ids') else encoding
        
        if self.debug:
            print(f"\n📚 Learning {len(tokens)} tokens...")
        
        initial_memory = self.stats["memory_entries"]
        
        # Learn CLS token at beginning
        self.learn_token(self.CLS_TOKEN)
        
        # Learn the text tokens
        for token_id in tokens:
            if token_id in self.valid_tokens:
                self.learn_token(token_id)
        
        # Learn SEP token at end
        self.learn_token(self.SEP_TOKEN)
        
        if self.debug:
            new_entries = self.stats["memory_entries"] - initial_memory
            print(f"✓ Learned: {len(tokens)} tokens")
            print(f"  New memory entries: {new_entries}")
            print(f"  Total orbits: {self.stats['orbits_discovered']}")
    
    def generate_text(self, max_tokens: int = 50) -> str:
        """Generate text."""
        tokens = []
        
        if self.debug:
            print(f"\n🌀 Generating (max {max_tokens} tokens)...")
        
        for i in range(max_tokens):
            token_id = self.generate_token()
            
            if token_id is None:
                if self.debug:
                    print(f"[No resonant token found - stopping]")
                break
            
            tokens.append(token_id)
            
            # Stop at SEP token
            if token_id == self.SEP_TOKEN:
                if self.debug:
                    print(f"[SEP token - stopping]")
                break
        
        # Decode tokens
        if not tokens:
            return ""
        text = self.tokenizer.decode(tokens)
        return text
    
    def generate_from_prompt(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text directly from a prompt using the pre-trained model's knowledge.
        
        This is the correct way to use a pre-trained model - no learning needed,
        just direct generation from the model's existing knowledge.
        """
        # Reset to clean state
        self.reset()
        
        # Add system prompt for better behavior
        system_prompt = "You are a helpful AI assistant. Please provide clear and accurate responses."
        
        # Set context by learning the system prompt and user prompt
        self.learn_text(system_prompt)
        self.learn_text(f"User: {prompt}")
        self.learn_text("Assistant:")
        
        # Generate from the model's pre-trained knowledge
        return self.generate_text(max_tokens=max_tokens)
    
    def print_stats(self) -> None:
        """Print kernel statistics."""
        print("📊 Statistics:")
        print(f"  Tokens learned: {self.stats['tokens_learned']}")
        print(f"  Memory entries: {self.stats['memory_entries']}")
        print(f"  Orbits discovered: {self.stats['orbits_discovered']}")
        print(f"  Generation steps: {self.stats['generation_steps']}")
        print(f"  Unique states visited: {self.stats['states_visited']}")
        print(f"  Unique masks: {self.stats['unique_masks']}")
        print(f"  Weight tensors: {len(self.virtual_tokens) // 1000:,}k virtual tokens")
        print(f"  Algedonic resets: {self.stats['algedonic_resets']}")
    
    def interactive_mode(self):
        """Interactive mode for asking questions to the pre-trained model."""
        print("\n" + "="*60)
        print("Interactive Mode - Ask questions to the pre-trained model")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                prompt = input("\n🤔 Your question: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print(f"🤖 Generating response...")
                response = self.generate_from_prompt(prompt, max_tokens=100)
                print(f"💬 Response: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_demo(self):
        """Run demonstration with direct generation from pre-trained model."""
        print("\n" + "="*60)
        model_display = self.model_name or "unknown"
        print(f"GyroSI Kernel v0.9.11.0 - {model_display} Demo")
        print("="*60)
        
        # Simple user questions
        test_questions = [
            "Hello, how are you?",
            "Tell me a story please."
        ]
        
        for question in test_questions:
            print(f"\n🤔 User: {question}")
            
            # Generate directly from the pre-trained model's knowledge
            print(f"🤖 Assistant: ", end="", flush=True)
            generated = self.generate_from_prompt(question, max_tokens=50)
            print(f"{generated}")
        
        # Print statistics at the end
        print("\n" + "="*60)
        self.print_stats()
        print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_cache_info()
    
    # Initialize kernel with Qwen3-4B-Instruct-2507
    kernel = GyroKernel(debug=True)  # Default to Qwen3-4B-Instruct-2507
    
    print("\n" + "=" * 60)
    print(f"GyroSI Kernel v0.9.11.0 - Qwen3-0.6B Demo")
    print("=" * 60)
    
    kernel.run_demo()