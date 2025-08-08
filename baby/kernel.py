"""
GyroSI Kernel - Full Physics Model

This implements the complete physics model using real epistemology:
1. 48-bit states with deterministic epistemology transitions
2. LEB128 tokenization with ψ isomorphism (XOR 0xAA)
3. Monodromic Fold for learning
4. Resonance-based generation (no scoring)

Uses the real meta files: epistemology.npy, ontology_keys.npy, tokenizer.json
"""

import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# Core Constants
GENE_Mic_S = 0xAA  # Holographic topology constant

# GENE_Mac_S: 48-bit archetypal tensor [4, 2, 3, 2]
GENE_Mac_S = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
    [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
    [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
], dtype=np.int8)

# Exon masks for gyration operations
EXON_LI_MASK = 0b01000010  # UNA bits (Parity/Reflection)
EXON_FG_MASK = 0b00100100  # ONA bits (Forward Gyration)
EXON_BG_MASK = 0b00011000  # BU-Eg bits (Backward Gyration)


def tensor_to_int(tensor: np.ndarray) -> int:
    """Convert 48-bit tensor to integer state."""
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError(f"Expected tensor shape (4, 2, 3, 2), got {tensor.shape}")
    
    # Flatten and convert: +1 -> 0, -1 -> 1
    bits = (tensor.flatten(order="C") == -1).astype(np.uint8)
    packed = np.packbits(bits, bitorder="big")
    return int.from_bytes(packed.tobytes(), "big")


def int_to_tensor(state_int: int) -> np.ndarray:
    """Convert integer state to 48-bit tensor."""
    if state_int >= (1 << 48) or state_int < 0:
        raise ValueError(f"state_int {state_int} out of bounds for 48-bit")
    
    state_bytes = state_int.to_bytes(6, "big")
    bits = np.unpackbits(np.frombuffer(state_bytes, dtype=np.uint8), bitorder="big")
    tensor_flat = (1 - 2 * bits).astype(np.int8)
    return tensor_flat.reshape(4, 2, 3, 2)


def apply_gyration(state_int: int, intron: int) -> int:
    """Apply gyroscopic transformation to state using intron instruction."""
    intron &= 0xFF
    
    # Build transformation mask
    mask = 0
    if intron & EXON_LI_MASK:  # LI: flip all bits
        mask ^= (1 << 48) - 1
    if intron & EXON_FG_MASK:  # FG: flip layers 0 & 2
        for layer in (0, 2):
            for frame in range(2):
                for row in range(3):
                    for col in range(2):
                        bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                        mask ^= 1 << bit_index
    if intron & EXON_BG_MASK:  # BG: flip layers 1 & 3
        for layer in (1, 3):
            for frame in range(2):
                for row in range(3):
                    for col in range(2):
                        bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                        mask ^= 1 << bit_index
    
    # Apply transformation
    temp_state = state_int ^ mask
    
    # Thomas gyration (path-dependent memory)
    intron_pattern = 0
    for i in range(6):
        intron_pattern |= intron << (8 * i)
    intron_pattern &= (1 << 48) - 1
    
    gyration = temp_state & intron_pattern
    final_state = temp_state ^ gyration
    
    return final_state & ((1 << 48) - 1)


def fold(a: int, b: int) -> int:
    """The Monodromic Fold: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))"""
    a &= 0xFF
    b &= 0xFF
    gyration_of_b = b ^ (a & (~b & 0xFF))
    return (a ^ gyration_of_b) & 0xFF


def fold_sequence(introns: List[int], start_state: int = 0) -> int:
    """Apply Monodromic Fold to sequence of introns."""
    result = start_state & 0xFF
    for intron in introns:
        result = fold(result, intron & 0xFF)
    return result


def transcribe_byte(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA."""
    return (byte ^ GENE_Mic_S) & 0xFF


def untranscribe_byte(intron: int) -> int:
    """ψ⁻¹ isomorphism: intron → byte via XOR 0xAA."""
    return (intron ^ GENE_Mic_S) & 0xFF


def token_id_to_leb128(token_id: int) -> List[int]:
    """Convert token ID to LEB128 bytes."""
    if token_id < 0:
        raise ValueError("Token ID must be non-negative")
    
    bytes_list = []
    while True:
        byte = token_id & 0x7F
        token_id >>= 7
        if token_id == 0:
            bytes_list.append(byte)
            break
        else:
            bytes_list.append(byte | 0x80)
    return bytes_list


def leb128_to_token_id(leb_bytes: List[int]) -> int:
    """Convert LEB128 bytes to token ID."""
    result = 0
    shift = 0
    
    for byte in leb_bytes:
        if shift > 28:
            raise ValueError("Token ID too large")
        result |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            break
    
    return result


def token_to_introns(token_id: int) -> List[int]:
    """Convert token ID to intron sequence via ψ isomorphism."""
    leb_bytes = token_id_to_leb128(token_id)
    return [transcribe_byte(b) for b in leb_bytes]


def introns_to_token(introns: List[int]) -> int:
    """Convert intron sequence to token ID via ψ⁻¹ isomorphism."""
    leb_bytes = [untranscribe_byte(i) for i in introns]
    return leb128_to_token_id(leb_bytes)


class GyroKernel:
    """Full GyroSI physics model with epistemology."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize with real epistemology and tokenizer."""
        if base_path is None:
            base_path = Path(__file__).parents[1] / "memories"
        
        self.base_path = base_path
        
        # Load ontology and epistemology
        self._load_physics_tables()
        
        # Initialize state to archetypal
        archetypal_int = tensor_to_int(GENE_Mac_S)
        self.current_state_index = self._get_state_index(archetypal_int)
        
        # Learning storage
        self.learned_masks: Dict[int, int] = {}  # token_id -> 8-bit mask
        self.learned_sequence: List[int] = []  # Store learned token sequence
        self._generation_index = 0
        
        # Physics memory: track last 6 states (diameter of state-graph)
        self.state_history: List[int] = [self.current_state_index]
        self.max_history = 6
        
        # Token cycling: avoid getting stuck on same token
        self.last_generated_tokens: List[int] = []
        self.max_repeat_tokens = 3
        
        # Physics switches (for testing and ablation)
        self.enable_cycle_gating = True
        self.enable_theta_window = True
        self.enable_orbit_constraints = True
        self.enable_cs_asymmetric = True
        self.enable_special_token_stages = True
        self.enable_mask_interference = True
        self.enable_short_memory = True
        
        # Physics parameters
        self.theta_window_size = 0.2  # Δθ for neighborhood
        self.orbit_tolerance = 5      # Orbit family tolerance
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
    def _load_physics_tables(self) -> None:
        """Load epistemology, ontology, and theta from disk."""
        meta_path = self.base_path / "public" / "meta"
        
        # Load ontology (state integers)
        ontology_path = meta_path / "ontology_keys.npy"
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")
        self.ontology = np.load(ontology_path, mmap_mode="r")
        
        # Load epistemology (state transition table)
        epistemology_path = meta_path / "epistemology.npy"
        if not epistemology_path.exists():
            raise FileNotFoundError(f"Epistemology not found: {epistemology_path}")
        self.epistemology = np.load(epistemology_path, mmap_mode="r")
        
        # Load theta (angular divergence) if available
        theta_path = meta_path / "theta.npy"
        try:
            self.theta = np.load(theta_path, mmap_mode="r")
            print(f"📊 Loaded theta: {len(self.theta):,} values")
        except FileNotFoundError:
            print("⚠️ Theta table not found - using approximation")
            self.theta = None
        
        print(f"📊 Loaded ontology: {len(self.ontology):,} states")
        print(f"📊 Loaded epistemology: {self.epistemology.shape}")
    
    def _load_tokenizer(self):
        """Load the real BERT tokenizer."""
        try:
            from tokenizers import Tokenizer
            tokenizer_path = self.base_path / "public" / "tokenizers" / "bert-base-uncased" / "tokenizer.json"
            if tokenizer_path.exists():
                print(f"📝 Loaded tokenizer: {tokenizer_path}")
                return Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            print("⚠️ tokenizers not available, using fallback")
        return None
    
    def _get_state_index(self, state_int: int) -> int:
        """Get ontology index for a state integer."""
        idx = np.searchsorted(self.ontology, state_int)
        if idx == len(self.ontology) or self.ontology[idx] != state_int:
            raise ValueError(f"State {state_int} not in ontology")
        return int(idx)
    
    def _get_state_int(self, state_index: int) -> int:
        """Get state integer from ontology index."""
        if state_index < 0 or state_index >= len(self.ontology):
            raise ValueError(f"Index {state_index} out of bounds")
        return int(self.ontology[state_index])
    
    def process_token(self, token_id: int) -> None:
        """Process a token through epistemology and learn its pattern."""
        introns = token_to_introns(token_id)
        
        # Apply each intron through epistemology (deterministic transitions)
        for intron in introns:
            next_index = self.epistemology[self.current_state_index, intron & 0xFF]
            self.current_state_index = int(next_index)
            
            # Track state history (6-step memory)
            self.state_history.append(self.current_state_index)
            if len(self.state_history) > self.max_history:
                self.state_history.pop(0)
        
        # Learn the token's holographic signature
        mask = fold_sequence(introns, start_state=0)
        old_mask = self.learned_masks.get(token_id, 0)
        self.learned_masks[token_id] = fold(old_mask, mask)
        
        # Store in learned sequence for parrot mode
        self.learned_sequence.append(token_id)
    
    def generate_token(self) -> int:
        """Generate next token based on endogenous resonance - no scoring."""
        current_stage = self._get_cycle_stage(self.current_state_index)
        
        # Physics-based token selection with admissibility checks
        admissible_tokens = []
        
        # Get theta neighborhood for mask envelope
        if self.enable_theta_window:
            neighborhood = self._get_theta_neighborhood(self.current_state_index)
            local_envelope = self._get_local_mask_envelope(neighborhood)
        else:
            local_envelope = 0xFF  # All bits set if theta window disabled
        
        # Check each learned token for physics admissibility
        for token_id in self.learned_masks.keys():
            # Stage-based token restrictions
            if not self._is_token_stage_allowed(token_id, current_stage):
                continue
                
            # Check if token creates valid stage transition
            if self.enable_cycle_gating:
                try:
                    next_state_index = self._simulate_token_transition(token_id)
                    next_stage = self._get_cycle_stage(next_state_index)
                    if not self._is_stage_transition_allowed(current_stage, next_stage):
                        continue
                except:
                    continue  # Skip on simulation error
            
            # Check mask interference (neural firing condition)
            if self.enable_mask_interference and local_envelope != 0:
                current_state_int = self._get_state_int(self.current_state_index)
                exon_product = self._state_to_exon_product(current_state_int)
                resonance = exon_product & local_envelope
                if resonance == 0:
                    continue  # No interference, no firing
            
            # Check short memory (avoid recent states)
            if self.enable_short_memory:
                if self._would_revisit_recent_state(token_id):
                    continue
            
            # Token passed all checks - it's admissible
            admissible_tokens.append(token_id)
        
        # Select from admissible tokens (deterministic tie-breaking)
        if admissible_tokens:
            # Sort by token ID for deterministic ordering
            admissible_tokens.sort()
            selected_token = admissible_tokens[0]  # First admissible
            
            # Track this token
            self.last_generated_tokens.append(selected_token)
            if len(self.last_generated_tokens) > 10:
                self.last_generated_tokens.pop(0)
            return selected_token
        
        # No admissible learned tokens: use Common Source logic
        token_id = self._generate_from_common_source()
        self.last_generated_tokens.append(token_id)
        if len(self.last_generated_tokens) > 10:
            self.last_generated_tokens.pop(0)
        return token_id
    
    def _token_resonates_from_state(self, token_id: int, state_index: int) -> bool:
        """Check if token creates resonance from current state via epistemology."""
        introns = token_to_introns(token_id)
        
        # Primary resonance: Can we reach a learned state pattern?
        test_state = state_index
        try:
            for intron in introns:
                next_state = self.epistemology[test_state, intron & 0xFF]
                test_state = int(next_state)
                
                # Check if this intermediate state has been learned before
                for learned_token in self.learned_masks.keys():
                    learned_introns = token_to_introns(learned_token)
                    if len(learned_introns) > 0:
                        # Check if current state matches a known intermediate state
                        check_state = state_index
                        for i, learned_intron in enumerate(learned_introns):
                            if i < len(introns) and introns[i] == learned_intron:
                                check_state = self.epistemology[check_state, learned_intron & 0xFF]
                                if int(check_state) == test_state:
                                    return True
            
            # Alternative resonance: Does this token create meaningful structure?
            final_state_int = self._get_state_int(test_state)
            final_exon = self._state_to_exon_product(final_state_int)
            
            # Resonance if we end up at a structurally similar state
            current_state_int = self._get_state_int(state_index)
            current_exon = self._state_to_exon_product(current_state_int)
            
            # Check for harmonic resonance (patterns in exon products)
            if final_exon != 0 and current_exon != 0:
                resonance_overlap = final_exon & current_exon
                return resonance_overlap != 0
            
            # Fallback: Check if we recognize this token directly
            return token_id in self.learned_masks
            
        except (IndexError, ValueError):
            return False
    
    def _get_visited_states(self) -> set:
        """Get set of states we've visited (limited memory - last 6 steps)."""
        return set(self.state_history)
    
    def _get_theta_for_state(self, state_index: int) -> float:
        """Get theta (angular divergence) for a state."""
        try:
            if hasattr(self, 'theta') and state_index < len(self.theta):
                return float(self.theta[state_index])
            # Fallback: approximate from state structure
            state_int = self._get_state_int(state_index)
            popcount = bin(state_int).count('1')
            return min(popcount * 0.1, 1.5)  # Rough approximation
        except:
            return 0.0
    
    def _get_cycle_stage(self, state_index: int) -> str:
        """Determine cycle stage from theta value following CGM theory."""
        theta = self._get_theta_for_state(state_index)
        
        # CGM cycle stages based on theta thresholds
        if theta < 0.1:
            return "CS"      # Common Source (θ ≈ 0)
        elif theta < 0.5:
            return "UNA"     # Unity Non-Absolute (θ = π/4 ≈ 0.785)
        elif theta < 1.0:
            return "ONA"     # Opposition Non-Absolute (θ = π/4 ≈ 0.785) 
        elif theta < 1.4:
            return "BU_IN"   # Balance Universal - Ingress
        elif theta < 1.8:
            return "BU_EG"   # Balance Universal - Egress
        else:
            return "CLOSURE" # Full cycle completion
    
    def _is_stage_transition_allowed(self, current_stage: str, next_stage: str) -> bool:
        """Check if transition between cycle stages is allowed (forward-only)."""
        stage_order = ["CS", "UNA", "ONA", "BU_IN", "BU_EG", "CLOSURE"]
        
        if current_stage not in stage_order or next_stage not in stage_order:
            return True  # Allow unknown stages
        
        current_idx = stage_order.index(current_stage)
        next_idx = stage_order.index(next_stage)
        
        # Allow forward progression or staying in same stage
        # Also allow CLOSURE -> CS (cycle restart)
        if next_idx >= current_idx:
            return True
        if current_stage == "CLOSURE" and next_stage == "CS":
            return True
        
        return False
    
    def _simulate_token_transition(self, token_id: int) -> int:
        """Simulate applying a token and return the resulting state index."""
        introns = token_to_introns(token_id)
        
        # Simulate the token's path through epistemology
        test_state = self.current_state_index
        try:
            for intron in introns:
                next_state = self.epistemology[test_state, intron & 0xFF]
                test_state = int(next_state)
            return test_state
        except (IndexError, ValueError):
            return self.current_state_index  # Return current if simulation fails
    
    def _would_revisit_recent_state(self, token_id: int) -> bool:
        """Check if token would lead to a recently visited state."""
        try:
            next_state = self._simulate_token_transition(token_id)
            return next_state in self.state_history
        except:
            return False
    
    def _get_theta_neighborhood(self, state_index: int) -> List[int]:
        """Get states within theta window of current state."""
        if not self.enable_theta_window:
            return [state_index]  # Just current state
            
        current_theta = self._get_theta_for_state(state_index)
        neighborhood = []
        
        # Sample a subset of states to check (for performance)
        total_states = len(self.ontology) if hasattr(self, 'ontology') else 10000
        sample_size = min(1000, total_states // 10)  # Sample 10% or max 1000
        
        import random
        sample_indices = random.sample(range(total_states), sample_size)
        
        for other_idx in sample_indices:
            other_theta = self._get_theta_for_state(other_idx)
            if abs(other_theta - current_theta) <= self.theta_window_size:
                neighborhood.append(other_idx)
        
        return neighborhood if neighborhood else [state_index]
    
    def _get_local_mask_envelope(self, state_indices: List[int]) -> int:
        """Get local envelope via OR of masks from neighborhood states."""
        envelope = 0x00
        
        for state_idx in state_indices:
            # Look for learned masks at this state
            for token_id, mask in self.learned_masks.items():
                # Check if this token was learned at or near this state
                # (simplified - in full system would check phenotype store)
                try:
                    # Simple heuristic: if token introns can reach this state
                    introns = token_to_introns(token_id)
                    test_state = self.current_state_index
                    for intron in introns:
                        next_state = self.epistemology[test_state, intron & 0xFF]
                        test_state = int(next_state)
                        if test_state == state_idx:
                            envelope |= mask
                            break
                except:
                    continue
        
        return envelope
    
    def _is_token_stage_allowed(self, token_id: int, current_stage: str) -> bool:
        """Check if token is allowed at current cycle stage."""
        if not self.enable_special_token_stages:
            return True
            
        # Special token restrictions based on cycle stage
        if token_id == 101:  # [CLS]
            return current_stage == "CS"  # Only at Common Source
        elif token_id == 102:  # [SEP] 
            return current_stage in ["BU_EG", "CLOSURE"]  # Only at closure stages
        
        return True  # Other tokens allowed at any stage
    
    def _generate_from_common_source(self) -> int:
        """Generate token using Common Source (CS) physics."""
        current_state_int = self._get_state_int(self.current_state_index)
        
        # Check if we're at or near Common Source (CS)
        if self._is_near_common_source(current_state_int):
            # At CS: generate based on asymmetric fixed point behavior
            return self._cs_asymmetric_emission()
        
        # Away from CS: use exon product to derive token
        exon_product = self._state_to_exon_product(current_state_int)
        return self._exon_to_token(exon_product)
    
    def _is_near_common_source(self, state_int: int) -> bool:
        """Check if state is at or near Common Source."""
        # CS characteristics: minimal structure, low population count
        popcount = bin(state_int).count('1')
        return popcount <= 2  # Very low structure
    
    def _cs_asymmetric_emission(self) -> int:
        """Common Source asymmetric fixed point: emit based on driving introns."""
        if not self.enable_cs_asymmetric:
            # Fallback to exon-based generation
            exon_product = self._state_to_exon_product(self._get_state_int(self.current_state_index))
            return self._exon_to_token(exon_product)
            
        # CS should preferentially emit [CLS] to begin sequences
        # This implements the "driving introns" concept
        current_state_int = self._get_state_int(self.current_state_index)
        
        # Detect standing vs driving based on state structure
        pattern = current_state_int & 0xFF
        
        # Standing introns (no change): emit padding or stay  
        if pattern == 0x00:
            return 0  # Padding token
            
        # Driving introns (initiate UNA): emit [CLS] to start sequence
        if bin(pattern).count('1') <= 2:
            return 101  # [CLS] token
            
        # Complex pattern: emit content token to advance
        base_token = 1000 + (pattern % 28000)
        return base_token
    
    def _state_to_exon_product(self, state_int: int) -> int:
        """Project 48-bit state to 8-bit exon product."""
        # XOR-fold the 6 bytes of the state
        b0 = (state_int >> 0) & 0xFF
        b1 = (state_int >> 8) & 0xFF
        b2 = (state_int >> 16) & 0xFF
        b3 = (state_int >> 24) & 0xFF
        b4 = (state_int >> 32) & 0xFF
        b5 = (state_int >> 40) & 0xFF
        return b0 ^ b1 ^ b2 ^ b3 ^ b4 ^ b5
    
    def _exon_to_token(self, exon_product: int) -> int:
        """Convert exon product to a valid token ID."""
        # Simple mapping: use exon product as base for small token IDs
        return exon_product % 1000  # Ensure reasonable token ID range
    
    def reset(self) -> None:
        """Reset to archetypal state."""
        archetypal_int = tensor_to_int(GENE_Mac_S)
        self.current_state_index = self._get_state_index(archetypal_int)
        self._generation_index = 0
        self.state_history = [self.current_state_index]
        self.last_generated_tokens = []
    
    def get_state_info(self) -> Dict[str, int]:
        """Get current state information."""
        current_state_int = self._get_state_int(self.current_state_index)
        return {
            'state_index': self.current_state_index,
            'state_int': current_state_int,
            'learned_tokens': len(self.learned_masks),
            'exon_product': self._state_to_exon_product(current_state_int)
        }


    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs using real tokenizer."""
        if self.tokenizer is None:
            # Fallback: simple word-based tokenization
            words = text.lower().split()
            return [hash(word) % 1000 for word in words]
        
        return self.tokenizer.encode(text).ids
    
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text using real tokenizer."""
        if self.tokenizer is None:
            # Fallback: just show token IDs
            return " ".join(f"[{tid}]" for tid in token_ids)
        
        # Filter out special tokens for cleaner output
        filtered_tokens = [t for t in token_ids if t not in [101, 102, 0]]  # Remove [CLS], [SEP], [PAD]
        return self.tokenizer.decode(filtered_tokens)
    
    def learn_text(self, text: str) -> None:
        """Learn from text input."""
        tokens = self.text_to_tokens(text)
        for token_id in tokens:
            self.process_token(token_id)
    
    def generate_text(self, max_tokens: int = 50, use_parrot: bool = False, debug: bool = False) -> str:
        """Generate text using physics or parrot mode."""
        tokens = []
        
        if debug:
            print(f"🎯 Generate text: max_tokens={max_tokens}, use_parrot={use_parrot}")
            print(f"   Learned tokens: {list(self.learned_masks.keys())}")
            print(f"   Learned sequence: {self.learned_sequence if use_parrot else 'N/A'}")
        
        for i in range(max_tokens):
            if use_parrot:
                # Parrot mode: reproduce learned sequence
                if hasattr(self, '_generation_index') and self._generation_index < len(self.learned_sequence):
                    token_id = self.learned_sequence[self._generation_index]
                    self._generation_index += 1
                else:
                    break  # End of learned sequence
            else:
                # Physics mode: generate based on current state
                token_id = self.generate_token()
            
            tokens.append(token_id)
            
            if debug:
                token_text = self.tokens_to_text([token_id])
                print(f"   Step {i+1}: {token_id} -> '{token_text}'")
            
            # Update state (except in parrot mode where we don't want to change state)
            if not use_parrot:
                self.process_token(token_id)
            
            # Stop at end-of-sequence tokens
            if token_id in [102, 0]:  # [SEP] or padding
                break
        
        return self.tokens_to_text(tokens)


def demo_full_model():
    """Demonstrate full GyroSI model with real epistemology."""
    print("🧠 GyroSI Full Model Demo")
    print("=" * 30)
    
    try:
        kernel = GyroKernel()
    except FileNotFoundError as e:
        print(f"❌ Cannot load model files: {e}")
        print("💡 Run: python -m baby.information ontology --output memories/public/meta/ontology_keys.npy")
        print("💡 Then: python -m baby.information epistemology --keys memories/public/meta/ontology_keys.npy --output memories/public/meta/epistemology.npy")
        return
    
    # Test text
    text = "algorithm is a sequence of instructions"
    print(f"📝 Input: {text}")
    print(f"⚙️ Initial state: {kernel.get_state_info()}")
    
    # Phase 1: Learning (ONCE)
    print("\n📚 LEARNING PHASE")
    print("-" * 20)
    kernel.learn_text(text)
    print(f"⚙️ After learning: {kernel.get_state_info()}")
    
    # Phase 2: Parrot mode (exact reproduction)
    print("\n🦜 PARROT MODE")
    print("-" * 15)
    kernel.reset()
    parrot_response = kernel.generate_text(max_tokens=20, use_parrot=True, debug=True)
    print(f"🤖 Parrot: {parrot_response}")
    
    # Phase 3: Resonance mode (physics-based)
    print("\n🧬 RESONANCE MODE")
    print("-" * 17)
    kernel.reset()
    
    # Show physics switches
    print("🔧 Physics switches:")
    for attr in dir(kernel):
        if attr.startswith('enable_'):
            print(f"   {attr}: {getattr(kernel, attr)}")
    
    # Generate with detailed physics debugging
    print("\n🎯 Physics generation step by step:")
    for step in range(5):
        current_stage = kernel._get_cycle_stage(kernel.current_state_index)
        current_theta = kernel._get_theta_for_state(kernel.current_state_index)
        
        print(f"\n   Step {step + 1}:")
        print(f"      Current state: {kernel.current_state_index}")
        print(f"      Cycle stage: {current_stage}")
        print(f"      Theta: {current_theta:.3f}")
        print(f"      State history: {kernel.state_history}")
        
        token = kernel.generate_token()
        token_text = kernel.tokens_to_text([token])
        
        print(f"      Generated: {token} ('{token_text}')")
        
        # Update state for next iteration
        kernel.process_token(token)
    
    # Also generate normal text
    kernel.reset()
    physics_response = kernel.generate_text(max_tokens=10, use_parrot=False, debug=False)
    print(f"\n🤖 Physics text: {physics_response}")
    
    print(f"\n📊 Model learned {len(kernel.learned_masks)} unique tokens")
    print("✅ Full model demo completed")


if __name__ == "__main__":
    demo_full_model()
