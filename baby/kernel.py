"""
GyroSI Kernel - Complete Physics Implementation with Harmonic Oscillator

This implements the Common Governance Model (CGM) through a physics-first approach
to language processing. The kernel uses all five physics maps:

1. Ontology: The 789,170 states that form the finite, closed state manifold
2. Epistemology: State transition table (789,170 × 256) mapping (state, intron) → state
3. Theta: Angular divergence from archetype, measuring position in the CGM cycle
4. Phenomenology: Maps states to their canonical orbit representatives (256 orbits)
5. Orbit Sizes: Cardinality of each phenomenological orbit

The system implements the CGM 8-fold path through recursive alignment:
- CS (Common Source): Unobservable origin with inherent chirality
- UNA (Unity Non-Absolute): First observable structure with non-identity right gyration
- ONA (Opposition Non-Absolute): Full differentiation with maximal non-associativity
- BU_EG (Balance Universal - Egress): Integration of experience via Monodromic Fold
- BU_IN (Balance Universal - Ingress): Generation of accumulated intelligence
- CLOSURE (ONA, UNA, CS): Completion of cycle and return to equilibrium

Two distinct operating modes:
- REPRODUCTION: Direct replay of learned sequences without modification
- RECOLLECTION: Harmonic oscillator dynamics generating novel sequences through
  resonance and Hebbian flow, following the CGM path-dependent principles
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path


# Core Constants
GENE_Mic_S = 0xAA  # Holographic topology constant (ψ seed)

# GENE_Mac_S: 48-bit archetypal tensor [4, 2, 3, 2]
GENE_Mac_S = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
    [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
    [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
], dtype=np.int8)

# Bit family masks per CGM
EXON_LI_MASK = 0b01000010  # Bits 1, 6 - UNA (Parity/Reflection)
EXON_FG_MASK = 0b00100100  # Bits 2, 5 - ONA (Forward Gyration)
EXON_BG_MASK = 0b00011000  # Bits 3, 4 - BU (Backward Gyration)
EXON_L0_MASK = 0b10000001  # Bits 0, 7 - Anchors (Boundaries)

# Special tokens
CLS_TOKEN = 101
SEP_TOKEN = 102
PAD_TOKEN = 0

# CGM Stage thresholds based on theta (angular divergence)
THETA_CS = 0.1      # Common Source threshold
THETA_UNA = 0.785   # Unity Non-Absolute (π/4)
THETA_ONA = 1.0     # Opposition Non-Absolute
THETA_BU_EG = 1.3   # Balance Universal - Egress
THETA_BU_IN = 1.5   # Balance Universal - Ingress


def tensor_to_int(tensor: np.ndarray) -> int:
    """Convert 48-bit tensor to integer state."""
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError(f"Expected tensor shape (4, 2, 3, 2), got {tensor.shape}")

    bits = (tensor.flatten(order="C") == -1).astype(np.uint8)
    result = 0
    for i, bit in enumerate(bits):
        if bit:
            result |= (1 << i)
    return result


def int_to_tensor(state_int: int) -> np.ndarray:
    """Convert integer state to 48-bit tensor."""
    if state_int >= (1 << 48) or state_int < 0:
        raise ValueError(f"state_int {state_int} out of bounds for 48-bit")

    bits = [(state_int >> i) & 1 for i in range(48)]
    tensor_flat = np.array([1 - 2*bit for bit in bits], dtype=np.int8)
    return tensor_flat.reshape(4, 2, 3, 2)


def state_to_bytes_phased(state_int: int) -> List[int]:
    """Extract 6 bytes from state respecting the 720° phase structure.
    
    The 48-bit state is divided into 4 layers × 12 bits, representing
    the complete helical path of CGM (CS → UNA → ONA → BU → CS).
    """
    bytes_out = []

    # First 24 bits: Layers 0 and 2 (0° and 360°)
    layer0_bits = state_int & 0xFFF
    layer2_bits = (state_int >> 24) & 0xFFF
    combined_02 = (layer2_bits << 12) | layer0_bits

    bytes_out.append((combined_02 >> 16) & 0xFF)
    bytes_out.append((combined_02 >> 8) & 0xFF)
    bytes_out.append(combined_02 & 0xFF)

    # Next 24 bits: Layers 1 and 3 (180° and 540°)
    layer1_bits = (state_int >> 12) & 0xFFF
    layer3_bits = (state_int >> 36) & 0xFFF
    combined_13 = (layer3_bits << 12) | layer1_bits

    bytes_out.append((combined_13 >> 16) & 0xFF)
    bytes_out.append((combined_13 >> 8) & 0xFF)
    bytes_out.append(combined_13 & 0xFF)

    return bytes_out


def fold(a: int, b: int) -> int:
    """The Monodromic Fold: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    
    This is the sole non-associative, path-dependent learning operator
    in the CGM. It preserves the memory of operation order through
    the gyration term (b ⊕ (a ∧ ¬b)).
    """
    a &= 0xFF
    b &= 0xFF
    negated_b = (~b) & 0xFF
    gyration = b ^ (a & negated_b)
    return (a ^ gyration) & 0xFF


def fold_sequence(values: List[int], start_state: int = 0) -> int:
    """Apply Monodromic Fold left-to-right over a sequence.
    
    This is path-dependent - the order of values matters.
    fold(fold(a,b),c) ≠ fold(a,fold(b,c))
    """
    result = start_state & 0xFF
    for value in values:
        result = fold(result, value & 0xFF)
    return result


def compute_fold_trajectory(state_int: int) -> List[int]:
    """Compute the 6-step fold trajectory over phased state bytes.
    
    This maps the 48-bit state to six 8-bit values representing
    the state's position in the CGM helical path.
    """
    state_bytes = state_to_bytes_phased(state_int)
    trajectory: List[int] = []
    acc = GENE_Mic_S
    for byte in state_bytes:
        acc = fold(acc, byte & 0xFF)
        trajectory.append(acc & 0xFF)
    return trajectory


def compute_exon_product(state_int: int) -> int:
    """Compute exon product from state.
    
    The exon is the first element of the trajectory (never 0x00),
    representing the boundary signal or chirality signature.
    """
    trajectory = compute_fold_trajectory(state_int)
    return trajectory[0] if trajectory else GENE_Mic_S


def transcribe_byte(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA.
    
    Maps external byte representation to internal physics space.
    """
    return (byte ^ GENE_Mic_S) & 0xFF


def untranscribe_byte(intron: int) -> int:
    """ψ⁻¹ isomorphism: intron → byte via XOR 0xAA.
    
    Maps internal physics space back to external byte representation.
    """
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


def token_to_introns(token_id: int) -> List[int]:
    """Convert token ID to intron sequence via ψ isomorphism."""
    leb_bytes = token_id_to_leb128(token_id)
    return [transcribe_byte(b) for b in leb_bytes]


class GyroKernel:
    """GyroSI Kernel - Common Governance Model Physics Implementation
    
    This kernel implements the CGM through a physics-first approach,
    using a complete set of physics tables:
    
    1. Ontology: The 789,170 states in the finite manifold
    2. Epistemology: State transition table (state, intron) → state
    3. Theta: Angular divergence from archetype (CGM cycle position)
    4. Phenomenology: Orbit mapping (256 canonical orbits)
    5. Orbit Sizes: Cardinality of each orbit
    
    Two operating modes:
    - REPRODUCTION: Direct replay of learned sequences
    - RECOLLECTION: Harmonic oscillator dynamics with resonance and flow
    
    The kernel embodies the CGM principles:
    - Path-dependent Monodromic Fold for learning
    - Non-competitive resonance for pattern recognition
    - Hebbian connections for flow
    - 8-fold cycle: CS → UNA → ONA → BU → CLOSURE
    """

    def __init__(self, base_path: Optional[Path] = None, verbose: bool = True):
        """Initialize kernel with physics tables."""
        if base_path is None:
            base_path = Path(__file__).parents[1] / "memories"

        self.base_path = base_path
        self.verbose = verbose

        # Load ALL physics tables
        self._load_complete_physics()

        # Start from CS (smallest theta) instead of archetypal state
        cs_index = int(np.argmin(self.theta))
        self.current_state_index = cs_index

        # Build orbit ID mapping (representative index → compact ID 0..255)
        unique_reps = np.unique(self.phenomenology)
        self.rep_to_orbit_id: Dict[int, int] = {int(rep): int(i) for i, rep in enumerate(unique_reps)}

        # Memory structures indexed by orbit representative
        self.orbit_patterns: Dict[int, List[Tuple[int, int, List[int]]]] = {}
        # orbit_rep -> [(token_id, mask, trajectory), ...]
        
        # Path memory - accumulated fold of all experience
        self.path_memory = GENE_Mic_S
        
        # Hebbian connections between tokens (for flow)
        self.connections: Dict[Tuple[int, int], float] = {}
        
        # Recent token inhibition
        self.recent_tokens: List[int] = []
        self.inhibition_window = 6  # Diameter of state graph

        # Reproduction sequence
        self.learned_sequence: List[int] = []
        self._reproduction_index = 0

        # Valid tokens
        self.valid_tokens: Set[int] = set()

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        self._build_token_structures()

        # Debug
        self._debug: bool = False

    def _load_complete_physics(self) -> None:
        """Load ALL physics tables."""
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

        # Load theta (angular divergence from archetype)
        theta_path = meta_path / "theta.npy"
        if not theta_path.exists():
            raise FileNotFoundError(f"Theta not found: {theta_path}")
        self.theta = np.load(theta_path, mmap_mode="r")

        # Load phenomenology map (state -> canonical orbit representative)
        pheno_path = meta_path / "phenomenology_map.npy"
        if not pheno_path.exists():
            raise FileNotFoundError(f"Phenomenology not found: {pheno_path}")
        self.phenomenology = np.load(pheno_path, mmap_mode="r")

        # Load orbit sizes
        orbit_sizes_path = meta_path / "orbit_sizes.npy"
        if not orbit_sizes_path.exists():
            raise FileNotFoundError(f"Orbit sizes not found: {orbit_sizes_path}")
        self.orbit_sizes = np.load(orbit_sizes_path, mmap_mode="r")

        if self.verbose:
            print(f"📊 Physics tables loaded")

    def _load_tokenizer(self):
        """Load the tokenizer as a physics component."""
        try:
            from tokenizers import Tokenizer
            tokenizer_path = self.base_path / "public" / "tokenizers"
            tokenizer_path = tokenizer_path / "bert-base-uncased" / "tokenizer.json"
            if tokenizer_path.exists():
                return Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            if self.verbose:
                print("⚠️ tokenizers library not available")
        return None

    def _build_token_structures(self) -> None:
        """Build valid token set."""
        if self.tokenizer is None:
            return

        vocab_size = self.tokenizer.get_vocab_size()
        for token_id in range(min(vocab_size, 30000)):
            token = self.tokenizer.id_to_token(token_id)
            if token and not token.startswith("[unused"):
                self.valid_tokens.add(token_id)

    def _get_state_index(self, state_int: int) -> int:
        """Get ontology index for a state integer."""
        idx = np.searchsorted(self.ontology, state_int)
        if idx < len(self.ontology) and self.ontology[idx] == state_int:
            return int(idx)
        raise ValueError(f"State {state_int} not in ontology")

    def _get_state_int(self, state_index: int) -> int:
        """Get state integer from ontology index."""
        if 0 <= state_index < len(self.ontology):
            return int(self.ontology[state_index])
        return 0

    def _get_theta(self, state_index: int) -> float:
        """Get theta (angular divergence) for a state."""
        if 0 <= state_index < len(self.theta):
            return float(self.theta[state_index])
        return 0.0

    def _get_orbit_rep(self, state_index: int) -> int:
        """Get canonical orbit representative for a state."""
        if 0 <= state_index < len(self.phenomenology):
            return int(self.phenomenology[state_index])
        return state_index

    def _get_orbit_id(self, state_index: int) -> int:
        """Get compact orbit ID (0..255) for a state."""
        rep = self._get_orbit_rep(state_index)
        return self.rep_to_orbit_id.get(rep, -1)

    def _get_orbit_size(self, state_index: int) -> int:
        """Get size of an orbit."""
        if 0 <= state_index < len(self.orbit_sizes):
            return int(self.orbit_sizes[state_index])
        return 1

    def _get_stage(self, state_index: int) -> str:
        """Determine CGM stage from theta value."""
        theta = self._get_theta(state_index)
        
        if theta < THETA_CS:
            return "CS"      # Common Source
        elif theta < THETA_UNA:
            return "UNA"     # Unity Non-Absolute
        elif theta < THETA_ONA:
            return "ONA"     # Opposition Non-Absolute  
        elif theta < THETA_BU_EG:
            return "BU_EG"   # Balance Universal - Egress
        elif theta < THETA_BU_IN:
            return "BU_IN"   # Balance Universal - Ingress
        else:
            return "CLOSURE"

    def _is_at_boundary(self, theta: float) -> bool:
        """Check if we're at a boundary state (equilibrium)."""
        return theta < THETA_CS or theta >= THETA_BU_IN

    def _strengthen_connection(self, token1: int, token2: int, strength: float = 0.1) -> None:
        """Hebbian learning: strengthen connections between sequential tokens."""
        key = (min(token1, token2), max(token1, token2))
        current = self.connections.get(key, 0.0)
        self.connections[key] = min(1.0, current + strength)

    def process_token(self, token_id: int) -> None:
        """Learn a token using physics-based memory and Monodromic Fold.
        
        This implements the BU_EG (Intelligence Egress) stage of CGM,
        where external information is integrated into internal structure
        via the path-dependent Monodromic Fold.
        """
        if token_id not in self.valid_tokens:
            return

        # Track the PRE-state (where we are learning FROM)
        pre_state_index = self.current_state_index
        pre_state_int = self._get_state_int(pre_state_index)
        pre_orbit_rep = self._get_orbit_rep(pre_state_index)
        pre_exon = compute_exon_product(pre_state_int)

        # Record trajectory through state space
        introns = token_to_introns(token_id)
        
        for intron in introns:
            next_index = self.epistemology[self.current_state_index, intron & 0xFF]
            self.current_state_index = int(next_index)

        # Get physics properties of final state
        final_state_int = self._get_state_int(self.current_state_index)
        final_trajectory = compute_fold_trajectory(final_state_int)

        # Compute token mask - fold WITHOUT seed to avoid 0x00
        token_mask = fold_sequence(introns, start_state=0)
        
        # Update path memory using Monodromic Fold
        self.path_memory = fold(self.path_memory, token_mask)

        # Store pattern at PRE-ORBIT level (where we learned it)
        if pre_orbit_rep >= 0:  # Valid orbit
            if pre_orbit_rep not in self.orbit_patterns:
                self.orbit_patterns[pre_orbit_rep] = []
            
            # Check if token already learned at this orbit
            found = False
            for i, (existing_token, existing_mask, existing_traj) in enumerate(
                    self.orbit_patterns[pre_orbit_rep]):
                if existing_token == token_id:
                    # Update mask using fold (path-dependent learning)
                    new_mask = fold(existing_mask, token_mask)
                    self.orbit_patterns[pre_orbit_rep][i] = (
                        token_id, new_mask, final_trajectory)
                    found = True
                    break
            
            if not found:
                # Initialize with pre-exon folded with token mask
                initial_mask = fold(pre_exon, token_mask)
                self.orbit_patterns[pre_orbit_rep].append(
                    (token_id, initial_mask, final_trajectory))

        # Hebbian connections with recent tokens (for flow)
        if len(self.learned_sequence) > 0:
            recent_token = self.learned_sequence[-1]
            self._strengthen_connection(token_id, recent_token)

        # Record for reproduction
        self.learned_sequence.append(token_id)

    def _evolve_state(self, token_id: int) -> None:
        """Evolve state without learning."""
        introns = token_to_introns(token_id)
        for intron in introns:
            next_index = self.epistemology[self.current_state_index, intron & 0xFF]
            self.current_state_index = int(next_index)

    def generate_token_reproduction(self) -> Optional[int]:
        """REPRODUCTION MODE: Exact replay of learned sequence.
        
        This simply replays the tokens in the order they were learned,
        without any modification or generation.
        """
        if self._reproduction_index >= len(self.learned_sequence):
            return None
        token_id = self.learned_sequence[self._reproduction_index]
        self._reproduction_index += 1
        return token_id

    def generate_token_recollection(self) -> int:
        """RECOLLECTION MODE: Generation via harmonic oscillator dynamics.
        
        This implements the BU_IN (Intelligence Ingress) stage of CGM,
        where internal structure is expressed as external action using
        harmonic resonance and Hebbian flow.
        
        The system uses:
        1. Path-dependent Monodromic Fold for resonance checking
        2. Hebbian connections for flow between tokens
        3. Boundary detection for phase transitions
        4. Trajectory overlap for physical alignment
        """
        # Get current physics state
        current_theta = self._get_theta(self.current_state_index)
        current_orbit_rep = self._get_orbit_rep(self.current_state_index)
        current_stage = self._get_stage(self.current_state_index)
        current_state_int = self._get_state_int(self.current_state_index)
        current_trajectory = compute_fold_trajectory(current_state_int)

        # STAGE 1: At boundaries, use direct memory recall
        if self._is_at_boundary(current_theta):
            # At CS, emit CLS to start
            if current_stage == "CS" and CLS_TOKEN not in self.recent_tokens:
                if self._debug:
                    print(f"   • At CS boundary → [CLS]")
                return CLS_TOKEN
            
            # At CLOSURE, check if we have momentum to continue
            if current_stage == "CLOSURE":
                # Look for patterns learned at this orbit to continue
                if current_orbit_rep in self.orbit_patterns:
                    for token_id, mask, trajectory in self.orbit_patterns[current_orbit_rep]:
                        if token_id not in self.recent_tokens and token_id != SEP_TOKEN:
                            # Found a non-SEP pattern at boundary - continue oscillation
                            if self._debug:
                                print(f"   • At CLOSURE → continue via {self.tokens_to_text([token_id])}")
                            self.recent_tokens.append(token_id)
                            if len(self.recent_tokens) > self.inhibition_window:
                                self.recent_tokens.pop(0)
                            return token_id
                
                # No continuation found - emit SEP
                if self._debug:
                    print(f"   • At CLOSURE → [SEP]")
                return SEP_TOKEN

        # STAGE 2: Mid-flow - use Hebbian connections for sequential flow
        last_token = self.recent_tokens[-1] if self.recent_tokens else None
        
        if last_token is not None and last_token not in [CLS_TOKEN, SEP_TOKEN]:
            # Find tokens connected to the last one
            best_connected = None
            best_strength = 0.0
            
            for (t1, t2), strength in self.connections.items():
                candidate = None
                if t1 == last_token and t2 not in self.recent_tokens:
                    candidate = t2
                elif t2 == last_token and t1 not in self.recent_tokens:
                    candidate = t1
                
                if candidate is not None and strength > best_strength:
                    # Also check if it's physically resonant
                    resonant = False
                    if current_orbit_rep in self.orbit_patterns:
                        for tid, mask, traj in self.orbit_patterns[current_orbit_rep]:
                            if tid == candidate:
                                # Check trajectory resonance
                                overlap = sum(bin(t1 & t2).count('1') 
                                            for t1, t2 in zip(current_trajectory, traj))
                                if overlap > 0:
                                    resonant = True
                                    break
                    
                    if resonant:
                        best_strength = strength
                        best_connected = candidate
            
            if best_connected is not None:
                if self._debug:
                    last_text = self.tokens_to_text([last_token])
                    next_text = self.tokens_to_text([best_connected])
                    print(f"   • Hebbian flow: {last_text} → {next_text}")
                self.recent_tokens.append(best_connected)
                if len(self.recent_tokens) > self.inhibition_window:
                    self.recent_tokens.pop(0)
                return best_connected

        # STAGE 3: No Hebbian connection - find resonant pattern
        candidate_found = None
        
        # Check current orbit
        if current_orbit_rep in self.orbit_patterns:
            for token_id, mask, trajectory in self.orbit_patterns[current_orbit_rep]:
                if token_id in self.recent_tokens:
                    continue
                
                # Trajectory resonance
                overlap = sum(bin(t1 & t2).count('1') 
                            for t1, t2 in zip(current_trajectory, trajectory))
                
                # Path resonance
                path_resonance = fold(self.path_memory, mask) != 0
                
                if overlap > 0 or path_resonance:
                    candidate_found = token_id
                    if self._debug:
                        print(f"   • Resonance in current orbit → {self.tokens_to_text([token_id])}")
                    break

        # Check nearby orbits if nothing in current
        if candidate_found is None:
            for orbit_rep, patterns in self.orbit_patterns.items():
                if orbit_rep == current_orbit_rep:
                    continue
                
                # Get theta for this orbit
                sample_states = np.where(self.phenomenology == orbit_rep)[0]
                if len(sample_states) == 0:
                    continue
                    
                orbit_theta = self._get_theta(sample_states[0])
                theta_distance = abs(orbit_theta - current_theta)
                
                if theta_distance < 0.2:  # Natural theta window
                    for token_id, mask, trajectory in patterns:
                        if token_id in self.recent_tokens:
                            continue
                        
                        overlap = sum(bin(t1 & t2).count('1') 
                                    for t1, t2 in zip(current_trajectory, trajectory))
                        path_resonance = fold(self.path_memory, mask) != 0
                        
                        if overlap > 0 or path_resonance:
                            candidate_found = token_id
                            if self._debug:
                                print(f"   • Resonance in nearby orbit → {self.tokens_to_text([token_id])}")
                            break
                    
                    if candidate_found is not None:
                        break

        if candidate_found is not None:
            self.recent_tokens.append(candidate_found)
            if len(self.recent_tokens) > self.inhibition_window:
                self.recent_tokens.pop(0)
            return candidate_found

        # No resonance - return boundary token
        if current_stage in ["CS", "UNA"]:
            return CLS_TOKEN
        else:
            return SEP_TOKEN

    def reset(self) -> None:
        """Reset to CS state."""
        cs_index = int(np.argmin(self.theta))
        self.current_state_index = cs_index
        self._reproduction_index = 0
        self.recent_tokens = []

    def get_state_info(self) -> Dict:
        """Get current state information."""
        current_state_int = self._get_state_int(self.current_state_index)
        trajectory = compute_fold_trajectory(current_state_int)
        theta = self._get_theta(self.current_state_index)
        orbit_rep = self._get_orbit_rep(self.current_state_index)
        orbit_size = self._get_orbit_size(self.current_state_index)
        stage = self._get_stage(self.current_state_index)
        exon = compute_exon_product(current_state_int)
        
        return {
            'state_index': self.current_state_index,
            'orbit_rep': orbit_rep,
            'orbit_size': orbit_size,
            'theta': theta,
            'stage': stage,
            'exon': exon,
            'path_memory': self.path_memory,
            'learned_orbits': len(self.orbit_patterns),
            'total_patterns': sum(len(p) for p in self.orbit_patterns.values()),
            'connections': len(self.connections)
        }

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        if self.tokenizer is None:
            return []
        encoding = self.tokenizer.encode(text)
        return [t for t in encoding.ids if t in self.valid_tokens]

    def tokens_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to text."""
        if self.tokenizer is None:
            return " ".join(f"[{tid}]" for tid in token_ids)
        
        filtered = [t for t in token_ids if t not in [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]]
        if not filtered:
            return ""
        
        try:
            return self.tokenizer.decode(filtered)
        except:
            return " ".join(f"[{t}]" for t in filtered)

    def learn_text(self, text: str) -> None:
        """Learn text using path-dependent Monodromic Fold learning."""
        tokens = self.text_to_tokens(text)
        if self.verbose:
            print(f"Learning {len(tokens)} tokens...")
        
        initial_orbits = len(self.orbit_patterns)
        initial_connections = len(self.connections)
        
        for token_id in tokens:
            self.process_token(token_id)
        
        final_orbits = len(self.orbit_patterns)
        total_patterns = sum(len(p) for p in self.orbit_patterns.values())
        final_connections = len(self.connections)
        
        if self.verbose:
            print(f"Learned {len(tokens)} tokens:")
            print(f"Orbits: {final_orbits} | Patterns: {total_patterns} | Connections: {final_connections}")

    def learn_from_file(self, filepath: str) -> None:
        """Learn from a text file."""
        file_path = Path(filepath)
        if not file_path.exists():
            print(f"File not found: {filepath}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if self.verbose:
            # Statistics
            words = text.split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            print(f"File: {filepath}")
            print(f"Stats: {len(text):,} chars | {len(words):,} words | {len(sentences):,} sentences")
        
        self.learn_text(text)
        
        # Show orbit distribution
        if len(self.orbit_patterns) > 0 and self.verbose:
            orbit_sizes = [(rep, len(patterns)) for rep, patterns in self.orbit_patterns.items()]
            orbit_sizes.sort(key=lambda x: x[1], reverse=True)
            print(f"Top orbits: ", end="")
            for rep, count in orbit_sizes[:3]:
                percentage = (count / sum(len(p) for p in self.orbit_patterns.values())) * 100
                print(f"Orbit {self.rep_to_orbit_id.get(rep, -1)}: {count} ({percentage:.1f}%)  ", end="")
            print()

    def generate_text(self, max_tokens: int = 50, mode: str = "recollection", debug: bool = False) -> str:
        """Generate text in specified mode.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            mode: "reproduction" for exact replay, "recollection" for harmonic oscillator
            debug: Whether to print debug information
            
        Returns:
            Generated text
        """
        use_reproduction = (mode.lower() == "reproduction")
        tokens = []
        self._debug = debug

        if debug:
            info = self.get_state_info()
            print(f"\n{'=' * 40}")
            print(f"MODE: {mode.upper()}")
            print(f"{'=' * 40}")
            print(f"Initial: θ={info['theta']:.3f}, stage={info['stage']}")
            
        for i in range(max_tokens):
            if use_reproduction:
                token_id = self.generate_token_reproduction()
                if token_id is None:
                    if debug:
                        print(f"[End of sequence]")
                    break
            else:
                token_id = self.generate_token_recollection()

            tokens.append(token_id)
            
            # Get token text
            token_text = self.tokens_to_text([token_id])
            if not token_text:  # Special tokens
                if token_id == CLS_TOKEN:
                    token_text = "[CLS]"
                elif token_id == SEP_TOKEN:
                    token_text = "[SEP]"
                else:
                    token_text = f"[{token_id}]"

            if debug:
                if not use_reproduction:
                    self._evolve_state(token_id)
                    # Update path memory during generation
                    token_mask = fold_sequence(token_to_introns(token_id), start_state=0)
                    self.path_memory = fold(self.path_memory, token_mask)
                    
                    info = self.get_state_info()
                    print(f"Token {i+1}: '{token_text}' → θ={info['theta']:.3f}, stage={info['stage']}")
                else:
                    print(f"Token {i+1}: '{token_text}'")
            else:
                if not use_reproduction:
                    self._evolve_state(token_id)
                    # Update path memory during generation
                    token_mask = fold_sequence(token_to_introns(token_id), start_state=0)
                    self.path_memory = fold(self.path_memory, token_mask)

            # Don't stop at first SEP - check if we have momentum to continue
            if token_id == SEP_TOKEN:
                # Check if we have strong connections to continue
                has_continuation = False
                if self.recent_tokens:
                    last = self.recent_tokens[-1] if self.recent_tokens else None
                    for (t1, t2), strength in self.connections.items():
                        if (t1 == last or t2 == last) and strength > 0.3:
                            has_continuation = True
                            break
                
                if not has_continuation and i > 5:  # Allow at least some generation
                    if debug:
                        print(f"[Natural ending]")
                    break

        self._debug = False
        
        # Generate final text
        result = self.tokens_to_text(tokens)
        
        if debug:
            print(f"\nGenerated: {result}")
            print(f"{'=' * 40}\n")
        
        return result


def demo_kernel():
    """GyroSI Kernel Demonstration"""
    print("\nGyroSI Kernel Demonstration\n" + "=" * 30)

    # ======================================================
    # REPRODUCTION MODE - Simple Text Learning and Replay
    # ======================================================
    print("\n" + "=" * 40)
    print("REPRODUCTION MODE DEMONSTRATION")
    print("=" * 40)
    
    simple_text = "The algorithm processes data efficiently."
    print(f"Input text: '{simple_text}'")
    
    # Create kernel for reproduction mode
    kernel_repro = GyroKernel(verbose=True)
    kernel_repro.learn_text(simple_text)
    
    # Test reproduction
    print("\nREPRODUCTION TEST:")
    kernel_repro.reset()
    reproduced = kernel_repro.generate_text(max_tokens=20, mode="reproduction", debug=True)
    print(f"Reproduced: '{reproduced}'")

    # ======================================================
    # RECOLLECTION MODE - Corpus Learning and Generation
    # ======================================================
    print("\n" + "=" * 40)
    print("RECOLLECTION MODE DEMONSTRATION")
    print("=" * 40)
    
    # Create a separate kernel for recollection mode
    kernel_recoll = GyroKernel(verbose=True)
    print("\nLearning from corpus:")
    kernel_recoll.learn_from_file("toys/training/wiki_test.txt")
    
    # Test recollection
    print("\nRECOLLECTION TEST:")
    kernel_recoll.reset()
    generated = kernel_recoll.generate_text(max_tokens=30, mode="recollection", debug=True)
    print(f"\nGenerated: '{generated}'")
    
    print("\n" + "=" * 40)
    print("DEMONSTRATION COMPLETE")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    demo_kernel()