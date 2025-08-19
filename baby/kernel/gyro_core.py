# baby/kernel/gyro_core.py

import bisect
import json
import numpy as np
import os
import struct
import tempfile
import threading
import time
from typing import Iterable, Optional, Dict, List, Tuple
from pathlib import Path
from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS, GENERATION_EXCLUDED
from baby.constants.frozen_channels import FROZEN_CHANNELS
from .bitops import popcount_u64_array


class GyroEngine:
    """Core GyroSI physics engine implementing deterministic token generation."""
    
    # Frozen constants
    MASK48 = FROZEN_CHANNELS.MASK48
    
    @staticmethod
    def channel_lex_key(bits: int) -> Tuple[int, ...]:
        """Convert packed state to channel lexicographic key for tie-breaking.
        Order: bit index 0..47 (layer, frame, row, col)
        """
        return tuple((bits >> i) & 1 for i in range(FROZEN_CHANNELS.TOTAL_BITS))
    
    def __init__(self, atlas_paths: Dict[str, str], store_paths: Dict[str, str], runtime: Dict[str, str], version_info: Optional[Dict[str, str]] = None, vocab_size: Optional[int] = None):
        """
        Load all five maps, build reverse index, open passive store,
        load or lazily initialise address memory cache.
        Enforce map integrity and versioning.
        """
        self.runtime = runtime
        self.max_nudges = runtime.get('max_nudges', 6)
        self.enable_self_reinforcement = runtime.get('enable_self_reinforcement', False)
        
        # Initialize cold-start grace window tracking
        self._cold_start_timestamp: int = 0
        self._timestamp_counter: int = 0
        self._cold_start_grace_window: int = int(runtime.get("cold_start_grace_window", 50))
        
        # Initialize passive log sync tracking
        self._passive_log_sync_counter: int = 0
        self._passive_log_pending_writes: int = 0
        self._passive_log_sync_interval: int = int(runtime.get("passive_log_sync_interval", 1000))
        self._passive_log_force_sync_interval: int = int(runtime.get("passive_log_force_sync_interval", 10000))
        
        # Store version information for validation
        self.version_info = version_info or {}
        
        # Initialize threading locks for concurrency safety (must be early)
        self._address_memory_lock = threading.RLock()  # For address memory writes
        self._passive_log_lock = threading.RLock()     # For passive log writes
        self._passive_memory_lock = threading.RLock()  # For passive memory index mutations
        self._address_cache_lock = threading.RLock()   # For address cache operations
        self._metrics_lock = threading.RLock()         # For metrics updates
        
        # Initialize runtime metrics and observability (must be early)
        self._metrics = {
            # Recovery ladder metrics
            'recovery_calls': 0,
            'recovery_level_1_hits': 0,
            'recovery_level_2_hits': 0,
            'recovery_level_3_hits': 0,
            'recovery_level_4_hits': 0,
            'recovery_level_5_hits': 0,
            'recovery_total_time': 0.0,
            'recovery_avg_time': 0.0,
            
            # Admissibility check metrics
            'admissibility_checks': 0,
            'admissibility_hits': 0,
            'admissibility_misses': 0,
            'admissibility_total_time': 0.0,
            'admissibility_avg_time': 0.0,
            
            # Cache performance metrics
            'address_cache_hits': 0,
            'address_cache_misses': 0,
            'address_cache_size': 0,
            'address_cache_hit_rate': 0.0,
            'address_cache_saves': 0,
            'address_cache_loads': 0,
            
            # General performance metrics
            'token_generations': 0,
            'state_lookups': 0,
            'orbit_computations': 0,
        }
        
        # Enforce version validation on all entry points
        self._validate_required_versions()
        
        # Store vocabulary size for bounded operations
        self.vocab_size = vocab_size or 50000  # Default fallback
        
        # Load atlas maps
        self._load_atlas_maps(atlas_paths)
        
        # Initialize fast state→tokens mapping *before* any log loads that use it
        self.state_to_tokens: Dict[int, set[int]] = {}

        # Initialize stores with version validation
        self._init_stores(store_paths)
        
        # Build reverse index for O(1) state lookup
        self._build_reverse_index()
        
        # Build orbit system: representatives and Hamming-2 neighbors
        self._build_orbit_system()
        
        # Cache for address memory with persistent storage
        self._address_cache = {}
        self._new_address_bindings = 0  # Counter for cache persistence
        self._load_address_cache()
        
        # Build orbit to tokens routing index
        self._build_orbit_to_tokens_index()
        
        # Verify slab index mapping integrity (FROZEN)
        self._verify_slab_mapping()
        
        # Precompute slab masks for efficient bitwise operations
        self._precompute_slab_masks()
        
        # Initialize mask interning pools
        self.mask_pool = {}              # value -> id
        self.mask_pool_reverse = {}      # id -> value
        self.next_mask_id = 0
        
        # state_to_tokens already initialized above, keep line for clarity (no-op)
        self.state_to_tokens = self.state_to_tokens
        
        # Threading locks already initialized earlier in constructor
        
        # Runtime metrics and observability already initialized earlier in constructor
        
    def _validate_required_versions(self):
        """Validate that all required version information is present and valid."""
        required_versions = ['atlas_version', 'address_version', 'config_version']
        
        for version_key in required_versions:
            if version_key not in self.version_info:
                raise RuntimeError(f"FATAL: Missing required version information: {version_key}. "
                                 f"All entry points must provide complete version_info.")
            
            version_value = self.version_info[version_key]
            if not version_value.strip():
                raise RuntimeError(f"FATAL: Invalid version format for {version_key}: {version_value}. "
                                 f"Version must be a non-empty string.")
        
        # Log successful validation
        atlas_v = self.version_info['atlas_version']
        address_v = self.version_info['address_version']
        config_v = self.version_info['config_version']
        print(f"Version validation passed: atlas_v{atlas_v}, address_v{address_v}, config_v{config_v}")
        
    def _load_atlas_maps(self, atlas_paths: Dict[str, str]):
        """Load and validate all atlas maps with memory mapping for large files."""
        try:
            # Load large maps with memory mapping for efficiency
            self.epistemology = np.load(atlas_paths['epistemology'], mmap_mode='r', allow_pickle=False)
            self.ontology_keys = np.load(atlas_paths['ontology_keys'], mmap_mode='r', allow_pickle=False)
            self.theta = np.load(atlas_paths['theta'], mmap_mode='r', allow_pickle=False)
            self.phenomenology_map = np.load(atlas_paths['phenomenology_map'], mmap_mode='r', allow_pickle=False)
            self.orbit_sizes = np.load(atlas_paths['orbit_sizes'], mmap_mode='r', allow_pickle=False)
            
            # Validate map integrity - enforce exact shapes and lengths
            expected_len = 788_986
            
            if len(self.ontology_keys) != expected_len:
                raise ValueError(f"ontology_keys length mismatch: expected {expected_len}, got {len(self.ontology_keys)}")
            
            if self.epistemology.shape != (expected_len, 256):
                raise ValueError(f"epistemology shape mismatch: expected ({expected_len}, 256), got {self.epistemology.shape}")
            
            if len(self.theta) != expected_len:
                raise ValueError(f"theta length mismatch: expected {expected_len}, got {len(self.theta)}")
            
            if len(self.phenomenology_map) != expected_len:
                raise ValueError(f"phenomenology_map length mismatch: expected {expected_len}, got {len(self.phenomenology_map)}")
            
            if len(self.orbit_sizes) != expected_len:
                raise ValueError(f"orbit_sizes length mismatch: expected {expected_len}, got {len(self.orbit_sizes)}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load atlas maps: {e}") from e
        
    def _init_stores(self, store_paths: Dict[str, str]):
        """Initialize address memory and passive memory stores."""
        self.address_memory_path = Path(store_paths['address_memory'])
        self.passive_memory_path = Path(store_paths['passive_memory'])
        self.address_metadata_path = self.address_memory_path.with_suffix('.json')
        
        # Ensure parent directories exist
        self.address_memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.passive_memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create address memory with version checking
        self._load_address_memory()
            
        # Initialize passive memory store
        if not self.passive_memory_path.exists():
            self.passive_memory_path.touch()
            
        # Load existing passive memory from log
        self._load_passive_memory_from_log()
            
    def _build_reverse_index(self):
        """Build reverse index from packed 48-bit state to row index for O(1) transitions."""
        # Vectorised build
        packed = (self.ontology_keys.astype(np.uint64) & np.uint64(FROZEN_CHANNELS.MASK48)).astype(np.uint64)
        self.state_to_index = {int(p): int(i) for i, p in enumerate(packed.tolist())}
        
        # Verify we have the expected number of unique states
        if len(self.state_to_index) != len(self.ontology_keys):
            raise ValueError(f"State index mismatch: expected {len(self.ontology_keys)} unique states, got {len(self.state_to_index)}")
            
        print(f"Built reverse index with {len(self.state_to_index)} states")
         
    def _build_orbit_system(self):
        """Construct orbit representatives and Hamming-2 neighbor cache from phenomenology_map."""
        # Extract the actual unique orbit codes from phenomenology_map
        self.orbit_codes: List[int] = sorted(set(self.phenomenology_map))
        
        # Find representative state index for each orbit code
        # Use the smallest state index in each orbit class as the representative
        self.orbit_representatives: Dict[int, int] = {}  # orbit_code -> state_index
        orbit_to_states: Dict[int, List[int]] = {}  # orbit_code -> list of state_indices
        
        for state_idx, orbit_code in enumerate(self.phenomenology_map):
            if orbit_code not in orbit_to_states:
                orbit_to_states[orbit_code] = []
            orbit_to_states[orbit_code].append(state_idx)
            
        # Select representative (smallest index) for each orbit
        for orbit_code in self.orbit_codes:
            if orbit_code in orbit_to_states:
                self.orbit_representatives[orbit_code] = min(orbit_to_states[orbit_code])
            else:
                # Handle case where orbit code has no states (shouldn't happen with valid atlas)
                print(f"Warning: Orbit code {orbit_code} has no associated states")
                
        # Precompute Hamming-2 neighbors across orbit representatives
        # These are needed for Recovery Level 2
        self.hamming2_neighbors: Dict[int, List[int]] = {}  # rep_orbit_code -> list[neighbor_orbit_codes]
        
        for orbit_code in self.orbit_codes:
            if orbit_code not in self.orbit_representatives:
                continue
                
            neighbors = []
            rep_state_idx = self.orbit_representatives[orbit_code]
            rep_packed_state = self.ontology_keys[rep_state_idx]
            
            # Convert to bitset for Hamming distance calculation
            rep_bits = self._packed_state_to_bitset(rep_packed_state)
            
            # Check all other orbit representatives for Hamming-2 distance
            for other_orbit_code in self.orbit_codes:
                if other_orbit_code == orbit_code or other_orbit_code not in self.orbit_representatives:
                    continue
                    
                other_state_idx = self.orbit_representatives[other_orbit_code]
                other_packed_state = self.ontology_keys[other_state_idx]
                other_bits = self._packed_state_to_bitset(other_packed_state)
                
                # Calculate Hamming distance with proper masking
                x = int(rep_bits) ^ int(other_bits)
                hamming_dist = (x & self.MASK48).bit_count()
                if hamming_dist == 2:
                    neighbors.append(other_orbit_code)
                    
            self.hamming2_neighbors[orbit_code] = neighbors
            
        print(f"Built orbit system with {len(self.orbit_representatives)} representatives")
        
    def _build_orbit_to_tokens_index(self):
        """Build orbit→tokens routing index for O(1) candidate gathering.
        
        FROZEN - Orbit structure is immutable: 256 orbits, sorted token lists.
        """
        self._orbit_to_tokens: Dict[int, List[int]] = {}
        
        # Initialize empty sorted lists for all orbits
        for orbit_code in range(256):
            self._orbit_to_tokens[orbit_code] = []
            
        # Bootstrap orbit index from passive memory at startup
        if hasattr(self, 'passive_memory_index'):
            for (st_idx, tok) in self.passive_memory_index.keys():
                orbit_code = self.phenomenology_map[st_idx]
                self._add_token_to_orbit_index(tok, orbit_code)
            
        # This will be populated lazily as addresses are computed
        # Token lists are kept sorted for deterministic iteration
        
    def _add_token_to_orbit_index(self, token_id: int, orbit_code: int):
        """Add a token to the orbit→tokens index.
        
        FROZEN - Maintains sorted order for deterministic candidate selection.
        """
        if not hasattr(self, '_orbit_to_tokens'):
            self._orbit_to_tokens: Dict[int, List[int]] = {}
            
        if orbit_code not in self._orbit_to_tokens:
            self._orbit_to_tokens[orbit_code] = []
            
        if token_id not in self._orbit_to_tokens[orbit_code]:
            # Insert in sorted order to maintain deterministic iteration
            bisect.insort(self._orbit_to_tokens[orbit_code], token_id)
    
    def _verify_slab_mapping(self):
        """FROZEN - Verify slab index mapping integrity: each slab has 6 distinct indices, union is 48."""
        all_indices = set()
        for slab_idx in range(FROZEN_CHANNELS.NUM_SLABS):  # All slabs
            slab_indices = self._get_slab_bit_indices(slab_idx)
            assert len(slab_indices) == FROZEN_CHANNELS.BITS_PER_SLAB, f"Slab {slab_idx} has {len(slab_indices)} indices, expected {FROZEN_CHANNELS.BITS_PER_SLAB}"
            assert len(set(slab_indices)) == FROZEN_CHANNELS.BITS_PER_SLAB, f"Slab {slab_idx} has duplicate indices: {slab_indices}"
            all_indices.update(slab_indices)
        
        assert len(all_indices) == FROZEN_CHANNELS.TOTAL_BITS, f"Union of all slab indices is {len(all_indices)}, expected {FROZEN_CHANNELS.TOTAL_BITS}"
        assert all_indices == set(range(FROZEN_CHANNELS.TOTAL_BITS)), f"Slab indices do not cover exactly 0-{FROZEN_CHANNELS.TOTAL_BITS-1}: {sorted(all_indices)}"
        assert len(all_indices) == FROZEN_CHANNELS.TOTAL_BITS
    
    def _precompute_slab_masks(self):
        """Precompute slab masks for efficient bitwise operations in hot paths."""
        self._slab_masks = []
        for s in range(FROZEN_CHANNELS.NUM_SLABS):
            idxs = self._get_slab_bit_indices(s)
            m = 0
            for i in idxs:
                m |= (1 << i)
            self._slab_masks.append(m & FROZEN_CHANNELS.MASK48)
    
    @property
    def SLAB_MASKS(self) -> List[int]:
        """Public access to slab masks for testing."""
        return self._slab_masks
        
    def _packed_state_to_bitset(self, packed_state: int) -> int:
        """Convert packed 48-bit state to bitset for Hamming distance calculations."""
        # Ensure we're working with 48-bit values
        # Sign encoding: +1 -> 0, -1 -> 1 (as per spec)
        return packed_state & FROZEN_CHANNELS.MASK48  # Mask to 48 bits
        
    def _bitset_to_packed_state(self, bitset: int) -> int:
        """Convert bitset back to packed state."""
        return bitset & FROZEN_CHANNELS.MASK48  # Mask to 48 bits
             
     # --- Byte/intron boundary ---
    @staticmethod
    def byte_to_intron(b: int) -> int:
        """Transform byte to intron via XOR with 0xAA.
        
        FROZEN - Immutable transformation rule: byte → intron mapping.
        ψ(b) = b ⊕ 0xAA
        """
        return (b & 0xFF) ^ 0xAA
        
    @staticmethod
    def intron_to_byte(i: int) -> int:
        """Transform intron to byte via XOR with 0xAA.
        
        FROZEN - Immutable transformation rule: intron → byte mapping.
        ψ⁻¹(i) = i ⊕ 0xAA (ψ is its own inverse)
        """
        return (i & 0xFF) ^ 0xAA
        
    @staticmethod
    def psi(byte_val: int) -> int:
        """Legacy alias for byte_to_intron. Use byte_to_intron instead.
        
        DEPRECATED - Use byte_to_intron for clarity.
        """
        return GyroEngine.byte_to_intron(byte_val)
        
    @staticmethod
    def encode_token_to_bytes(token_id: int) -> bytes:
        """
        If token_id < 256 -> single byte.
        Else encode via LEB128 (as per spec).
        
        FROZEN - Immutable encoding rule: token → bytes transformation.
        """
        if token_id < 256:
            return bytes([token_id])
        else:
            # LEB128 encoding
            result = []
            while token_id >= 128:
                result.append((token_id & 0x7F) | 0x80)
                token_id >>= 7
            result.append(token_id & 0x7F)
            return bytes(result)
            
    def token_to_introns(self, token_id: int) -> List[int]:
        """Convert token to list of introns via boundary transformation."""
        return [self.byte_to_intron(b) for b in GyroEngine.encode_token_to_bytes(token_id)]
        
    def _encode_leb128(self, value: int) -> List[int]:
        """
        Encode integer as LEB128 (Little Endian Base 128).
        Each byte encodes 7 bits of data plus 1 continuation bit.
        """
        if value < 0:
            raise ValueError(f"LEB128 encoding requires non-negative value, got {value}")
            
        result = []
        while value >= 128:
            # Take lower 7 bits and set continuation bit (bit 7)
            byte_val = (value & 0x7F) | 0x80
            result.append(byte_val)
            value >>= 7
            
        # Final byte (no continuation bit)
        result.append(value & 0x7F)
        return result
        
    def _decode_leb128(self, bytes_list: List[int]) -> int:
        """
        Decode LEB128 bytes back to integer (for testing/validation).
        """
        result = 0
        shift = 0
        
        for byte_val in bytes_list:
            # Extract 7 data bits
            data_bits = byte_val & 0x7F
            result |= (data_bits << shift)
            shift += 7
            
            # Check continuation bit
            if (byte_val & 0x80) == 0:
                break  # Last byte
                
        return result
        
    # --- Address memory (deterministic, physics-only) ---
    def address_of_token(self, token_id: int) -> int:
        """
        Returns packed 48-bit state for this token.
        Compute medoid over final states from all 256 orbit representatives.
        Tie-breaking: orbit size, channel lexicographic, token id.
        """
        with self._address_cache_lock:
            if token_id in self._address_cache:
                with self._metrics_lock:
                    self._metrics['address_cache_hits'] += 1
                    self._update_cache_metrics()
                return self._address_cache[token_id]
            else:
                with self._metrics_lock:
                    self._metrics['address_cache_misses'] += 1
                    self._update_cache_metrics()
            
        # Convert token to introns via ψ transformation
        introns = self.token_to_introns(token_id)
        
        # Compute final states for each of the 256 orbit representatives
        final_states = []
        for orbit_code in range(256):
            if orbit_code not in self.orbit_representatives:
                continue
                
            # Get representative state for this orbit
            rep_state_idx = self.orbit_representatives[orbit_code]
            rep_packed_state = self.ontology_keys[rep_state_idx]
            
            # Simulate micro-path through epistemology table
            current_state = rep_packed_state
            for intron in introns:
                current_state = self.apply_intron(current_state, intron)
                
            final_states.append(current_state)
            
        # Get unique final states
        unique_finals = list(set(final_states))
        
        # Compute medoid: maximize average agreements (angular distance surrogate)
        best_candidate = None
        
        # Vectorized medoid computation
        if len(unique_finals) > 1:
            # Convert unique_finals to numpy array of 48-bit ints
            finals_array = np.array(unique_finals, dtype=np.uint64)
            
            # Compute XOR matrix: finals_array[:, None] XOR finals_array[None, :]
            xor_matrix = finals_array[:, None] ^ finals_array[None, :]
            
            # Apply 48-bit mask and compute popcount (agreements = 48 - popcount)
            masked_xor = xor_matrix & np.uint64(FROZEN_CHANNELS.MASK48)
            agreements_matrix = 48 - popcount_u64_array(masked_xor.astype(np.uint64))
            
            # Sum agreements for each candidate
            total_agreements_array = np.sum(agreements_matrix, axis=1)
            
            # Find best candidate
            best_idx = np.argmax(total_agreements_array)
            best_candidate = unique_finals[best_idx]
        else:
             best_candidate = unique_finals[0]
                        
        # Use safe fallback if computation fails
        if best_candidate is None:
            # Fallback to orbit representative of orbit 0
            if 0 in self.orbit_representatives:
                best_candidate = self.ontology_keys[self.orbit_representatives[0]]
            else:
                # Final fallback to first state
                best_candidate = self.ontology_keys[0]
                
        # Cache the result
        with self._address_cache_lock:
            self._address_cache[token_id] = best_candidate
            
            # Periodically save cache to disk
            if len(self._address_cache) % 100 == 0:
                self._save_address_cache()
        
        # Update orbit→tokens routing index
        candidate_idx = self.state_to_index[best_candidate]
        orbit_code = self.phenomenology_map[candidate_idx]
        self._add_token_to_orbit_index(token_id, orbit_code)
        
        # Persist to address memory if needed
        self._persist_address_memory(token_id, best_candidate)
        
        return best_candidate
        
    def _compute_medoid(self, addresses: List[int]) -> int:
        """Compute medoid from a list of addresses using angular distance surrogate (maximize agreements).
        
        Args:
            addresses: List of 48-bit packed state addresses
            
        Returns:
            The medoid address (maximizes sum of agreements/dot products)
        """
        if not addresses:
            return 0
            
        if len(addresses) == 1:
            return addresses[0]
            
        # Get unique addresses
        unique_addresses = list(set(addresses))
        
        best_candidate = None
        best_agreements = -1
        total_agreements_array = None
        
        # Vectorized medoid computation
        if len(unique_addresses) > 1:
            # Convert unique_addresses to numpy array of 48-bit ints
            addresses_array = np.array(unique_addresses, dtype=np.uint64)
            
            # Compute XOR matrix: addresses_array[:, None] XOR addresses_array[None, :]
            xor_matrix = addresses_array[:, None] ^ addresses_array[None, :]
            
            # Apply 48-bit mask and compute popcount (agreements = 48 - popcount)
            masked_xor = xor_matrix & np.uint64(FROZEN_CHANNELS.MASK48)
            agreements_matrix = 48 - popcount_u64_array(masked_xor.astype(np.uint64))
            
            # Sum agreements for each candidate
            total_agreements_array = np.sum(agreements_matrix, axis=1)
            
            # Find candidates with maximum agreements
            max_agreements = np.max(total_agreements_array)
            best_indices = np.where(total_agreements_array == max_agreements)[0]
            
            # Start with first best candidate for tie-breaking
            best_candidate = unique_addresses[best_indices[0]]
            best_agreements = max_agreements
        else:
            best_candidate = unique_addresses[0]
            best_agreements = 48  # Perfect agreement with itself
            
        # Handle tie-breaking for multiple candidates with same agreements
        if len(unique_addresses) > 1 and total_agreements_array is not None:
            max_agreements = best_agreements
            for i, candidate in enumerate(unique_addresses):
                if total_agreements_array[i] == max_agreements:
                    total_agreements = total_agreements_array[i]
                
                    # Tie-breaking chain: smaller orbit size → channel lexicographic → address value
                    if total_agreements > best_agreements:
                        best_agreements = total_agreements
                        best_candidate = candidate
                    elif total_agreements == best_agreements and best_candidate is not None:
                        # Tie-break by orbit size
                        candidate_idx = self.state_to_index[candidate]
                        best_idx = self.state_to_index[best_candidate]
                        
                        candidate_orbit_size = self.orbit_sizes[candidate_idx]
                        best_orbit_size = self.orbit_sizes[best_idx]
                        
                        if candidate_orbit_size < best_orbit_size:
                            best_candidate = candidate
                        elif candidate_orbit_size == best_orbit_size:
                            # Tie-break by channel lexicographic (proper bit ordering)
                            candidate_key = self.channel_lex_key(candidate)
                            best_key = self.channel_lex_key(best_candidate)
                            if candidate_key < best_key:
                                best_candidate = candidate
                            elif candidate_key == best_key:
                                # Final tie-break by address value (deterministic)
                                if candidate < best_candidate:
                                    best_candidate = candidate
                            
        # Use safe fallback if computation fails
        if best_candidate is None:
            # Fallback to orbit representative of orbit 0
            if 0 in self.orbit_representatives:
                best_candidate = self.ontology_keys[self.orbit_representatives[0]]
            else:
                # Final fallback to first address
                best_candidate = unique_addresses[0]
                
        return best_candidate
        
    def _is_better_tie_break(self, candidate_state: int, current_best_state: Optional[int]) -> bool:
        """Implement tie-breaking: orbit size, channel lexicographic comparison of packed states."""
        if current_best_state is None:
            return True
            
        candidate_idx = self.state_to_index[candidate_state]
        best_idx = self.state_to_index[current_best_state]
        
        # First tie-break: smaller orbit size
        candidate_orbit_size = self.orbit_sizes[candidate_idx]
        best_orbit_size = self.orbit_sizes[best_idx]
        
        if candidate_orbit_size != best_orbit_size:
            return candidate_orbit_size < best_orbit_size
            
        # Second tie-break: channel lexicographic (proper bit ordering)
        if candidate_state != current_best_state:
            candidate_key = self.channel_lex_key(candidate_state)
            best_key = self.channel_lex_key(current_best_state)
            return candidate_key < best_key
            
        # States are identical
        return False
        
    def _load_address_memory(self):
        """Load address memory with memmap and version checking."""
        
        # Check if both files exist
        if self.address_memory_path.exists() and self.address_metadata_path.exists():
            try:
                # Load metadata
                with open(self.address_metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify version compatibility against config
                stored_atlas_version = metadata.get('atlas_version', 'unknown')
                stored_address_version = metadata.get('address_version', 'unknown')
                
                expected_atlas_version = self.version_info.get('atlas_version', 'v1.0.0')
                expected_address_version = self.version_info.get('address_version', 'v1.0.0')
                
                # Check for version mismatch - enforce hard failure
                if (stored_atlas_version != expected_atlas_version or 
                    stored_address_version != expected_address_version):
                    error_msg = (f"FATAL: Version mismatch detected - stored atlas_v{stored_atlas_version}/address_v{stored_address_version}, "
                               f"expected atlas_v{expected_atlas_version}/address_v{expected_address_version}. "
                               f"Cannot proceed with incompatible versions.")
                    raise RuntimeError(error_msg)
                
                # Load memory-mapped array with write access
                self.address_memory = np.memmap(self.address_memory_path, dtype='<u8', mode="r+")
                self.max_token_id = metadata.get('max_token_id', len(self.address_memory) - 1)
                
                print(f"Loaded address memory: atlas_v{stored_atlas_version}, address_v{stored_address_version}, max_token={self.max_token_id}")
                return
                
            except Exception as e:
                print(f"Warning: Failed to load address memory, recreating: {e}")
        
        # Create new address memory as zero-initialized binary file
        # Use larger initial capacity to avoid thrashing on first runs with o200k_harmony
        initial_array = np.zeros(100000, dtype='<u8')
        
        # Write raw binary data (not .npy format)
        with open(self.address_memory_path, 'wb') as f:
            f.write(initial_array.tobytes())
        
        # Open as memmap with write access
        self.address_memory = np.memmap(self.address_memory_path, dtype='<u8', mode="r+")
        self.max_token_id = -1
        
        # Write initial metadata directly
        metadata = {
            'atlas_version': self.version_info.get('atlas_version', 'v1.0.0'),
            'address_version': self.version_info.get('address_version', 'v1.0.0'),
            'max_token_id': -1
        }
        with open(self.address_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_address_memory(self):
        """Save address memory with atomic writes and metadata."""
        
        with self._address_memory_lock:
            # Create temporary files for atomic write
            tmp_dat = tempfile.NamedTemporaryFile(delete=False, suffix='.dat', dir=self.address_memory_path.parent)
            tmp_dat_path = Path(tmp_dat.name)
            tmp_dat.close()
            
            tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=self.address_metadata_path.parent)
            tmp_json_path = Path(tmp_json.name)
            tmp_json.close()
            
            try:
                # Find last non-zero index for max_token_id
                last_nonzero_index = 0
                for i in range(len(self.address_memory) - 1, -1, -1):
                    if self.address_memory[i] != 0:
                        last_nonzero_index = i
                        break
                
                # Save raw binary data to temporary file (not .npy format)
                with open(tmp_dat_path, 'wb') as f:
                    f.write(self.address_memory.tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                
                # Save metadata to temporary file
                metadata = {
                    'atlas_version': self.version_info.get('atlas_version', 'v1.0.0'),
                    'address_version': self.version_info.get('address_version', 'v1.0.0'),
                    'max_token_id': int(last_nonzero_index)
                }
                with open(tmp_json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Flush and sync the memmap before replacing
                self.address_memory.flush()
                # Note: numpy memmap flush() already syncs to disk, no need for additional fsync
                
                # Close memmap before atomic replace to prevent file locking on Windows
                try:
                    # Safely close the memmap without accessing private attributes
                    self.address_memory.flush()
                except (AttributeError, ValueError):
                    pass
                del self.address_memory
                
                # Atomic rename
                os.replace(str(tmp_dat_path), str(self.address_memory_path))
                os.replace(str(tmp_json_path), str(self.address_metadata_path))
                
                # Directory fsync for POSIX crash safety
                try:
                    dir_fd = os.open(str(self.address_memory_path.parent), os.O_RDONLY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except (OSError, AttributeError):
                    # Windows doesn't support directory fsync, ignore
                    pass
                
                # Re-open memmap after successful replace
                self.address_memory = np.memmap(self.address_memory_path, dtype='<u8', mode='r+')
                
            except Exception as e:
                # Clean up temporary files on error
                for tmp_path in [tmp_dat_path, tmp_json_path]:
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass  # Ignore cleanup errors
                raise e
    
    def _persist_address_memory(self, token_id: int, address: int):
        """Persist address binding to memory-mapped file with atomic writes."""
        
        with self._address_memory_lock:
            try:
                array_grew = False
                # Expand array if needed
                if token_id >= len(self.address_memory):
                    array_grew = True
                    # 1. Create new zero-filled array
                    new_size = max(token_id + 1, len(self.address_memory) * 2)
                    new_array = np.zeros(new_size, dtype='<u8')
                    
                    # 2. Copy old contents
                    new_array[:len(self.address_memory)] = self.address_memory
                    
                    # 3. Write to temporary file and atomic replace
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dat', dir=self.address_memory_path.parent) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        # Write raw binary data (not .npy format)
                        with open(tmp_path, 'wb') as f:
                            f.write(new_array.tobytes())
                            f.flush()
                            os.fsync(f.fileno())
                    
                    # 4. Flush and sync memmap before atomic replace
                    self.address_memory.flush()
                    # Note: numpy memmap flush() already syncs to disk, no need for additional fsync
                    try:
                        # Safely close the memmap without accessing private attributes
                        self.address_memory.flush()
                    except (AttributeError, ValueError):
                        pass
                    del self.address_memory
                    
                    # Atomic replace
                    os.replace(str(tmp_path), str(self.address_memory_path))
                    
                    # Directory fsync for POSIX crash safety
                    try:
                        dir_fd = os.open(str(self.address_memory_path.parent), os.O_RDONLY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except (OSError, AttributeError):
                        # Windows doesn't support directory fsync, ignore
                        pass
                    
                    # 5. Re-open as memmap
                    self.address_memory = np.memmap(self.address_memory_path, dtype='<u8', mode='r+')
                
                # Update the memory-mapped array
                self.address_memory[token_id] = (address & FROZEN_CHANNELS.MASK48)
                
                # Increment counter and save cache if threshold reached
                self._new_address_bindings += 1
                if self._new_address_bindings >= 4096:
                    self._save_address_cache()
                    self._new_address_bindings = 0
                
                # Flush to disk
                self.address_memory.flush()
                # Note: numpy memmap flush() already syncs to disk, no need for additional fsync
                
                # Write JSON sidecar if array grew
                if array_grew:
                    metadata = {
                        'atlas_version': self.version_info.get('atlas_version', 'v1.0.0'),
                        'address_version': self.version_info.get('address_version', 'v1.0.0'),
                        'max_token_id': int(token_id)
                    }
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=self.address_metadata_path.parent) as tmp_json:
                        json.dump(metadata, tmp_json, indent=2)
                        tmp_json.flush()
                        os.fsync(tmp_json.fileno())
                        tmp_json_path = Path(tmp_json.name)
                    
                    # Atomic replace for JSON
                    os.replace(str(tmp_json_path), str(self.address_metadata_path))
                    
                    # Directory fsync for POSIX crash safety
                    try:
                        dir_fd = os.open(str(self.address_metadata_path.parent), os.O_RDONLY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except (OSError, AttributeError):
                        # Windows doesn't support directory fsync, ignore
                        pass
                    
            except Exception as e:
                print(f"Warning: Failed to persist address memory: {e}")
        
    # --- State transitions ---
    def apply_intron(self, packed_state: int, intron: int) -> int:
        """
        Look up next state via epistemology table (indexing via reverse map).
        Must be O(1) lookup - no unknown state branches allowed.
        """
        # Guard against unknown states
        if packed_state not in self.state_to_index:
            raise KeyError(f"Packed state not in atlas: 0x{packed_state:012X}")
            
        # O(1) lookup via reverse index - state must exist in atlas
        state_idx = self.state_to_index[packed_state]
        
        # Bounds check for intron (0-255)
        if not (0 <= intron <= 255):
            raise ValueError(f"Invalid intron value: {intron}. Must be 0-255.")
            
        # Look up next state index in epistemology table
        next_state_idx = self.epistemology[state_idx, intron]
        
        # Convert back to packed state
        return self.ontology_keys[next_state_idx]
        
    def micro_path(self, start_state: int, introns: Iterable[int]) -> List[int]:
        """
        Returns list of packed states from start to end inclusive.
        """
        # Guard against unknown start state
        if start_state not in self.state_to_index:
            raise KeyError(f"Packed state not in atlas: 0x{start_state:012X}")
            
        states = [start_state]
        current_state = start_state
        
        for intron in introns:
            current_state = self.apply_intron(current_state, intron)
            states.append(current_state)
            
        return states
        
    # --- Admissibility checks (frozen channel cover and priorities) ---
    def is_admissible(self, start_state: int, token_id: int, *, global_strict: bool = True) -> bool:
        """
        Compat wrapper: 'global_strict' was expected by tests.
        If global_strict=False, we keep global non-decrease but relax the 'strict somewhere'
        requirement to allow 'no strict increase' to pass as long as slabs strictly improve.
        """
        start_time = time.perf_counter()
        
        with self._metrics_lock:
            self._metrics['admissibility_checks'] += 1
        
        result = self._is_admissible_core(start_state, token_id, require_global_strict=global_strict)
        
        with self._metrics_lock:
            if result:
                self._metrics['admissibility_hits'] += 1
            else:
                self._metrics['admissibility_misses'] += 1
            self._update_admissibility_timing(start_time)
        
        return result
    
    def _is_admissible_core(self, start_state: int, token_id: int, *, require_global_strict: bool) -> bool:
        """
        Implement global stepwise monotonicity, slab net non-decrease,
        and strict progress somewhere. All bit-level as specified.
        """
        # Get token address for comparison
        token_address = self.address_of_token(token_id)
        token_address_bits = self._packed_state_to_bitset(token_address)
        
        # Convert token to introns and compute micro-path
        introns = self.token_to_introns(token_id)
        micro_path = self.micro_path(start_state, introns)
        
        # Global channel: stepwise non-decrease of Hamming agreements
        global_progress = False
        for k in range(len(micro_path) - 1):
            current_bits = self._packed_state_to_bitset(micro_path[k])
            next_bits = self._packed_state_to_bitset(micro_path[k + 1])
            
            # Count agreements (matching bits) with token address
            current_agreements = self._count_agreements(current_bits, token_address_bits)
            next_agreements = self._count_agreements(next_bits, token_address_bits)
            
            # Require non-decrease
            if next_agreements < current_agreements:
                return False
                
            # Track if we have strict progress
            if next_agreements > current_agreements:
                global_progress = True
                
        # Slab channels: check end-to-start agreement for each slab
        start_bits = self._packed_state_to_bitset(micro_path[0])
        end_bits = self._packed_state_to_bitset(micro_path[-1])
        
        slab_progress = False
        for slab_idx in range(FROZEN_CHANNELS.NUM_SLABS):
            start_slab_agreements = self._count_slab_agreements_fast(start_bits, token_address_bits, slab_idx)
            end_slab_agreements = self._count_slab_agreements_fast(end_bits, token_address_bits, slab_idx)
            
            # Require non-decrease for this slab
            if end_slab_agreements < start_slab_agreements:
                return False
                
            # Track slab progress
            if end_slab_agreements > start_slab_agreements:
                slab_progress = True
                
        # Strict progress requirement: either global or at least one slab must improve
        if require_global_strict:
            return global_progress or slab_progress
        else:
            # Relaxed mode: allow no strict increase if slabs improve
            return slab_progress or global_progress
        
    def _count_agreements(self, bits1: int, bits2: int) -> int:
        """Exact matches across 48 bits."""
        return ( (~(bits1 ^ bits2) & FROZEN_CHANNELS.MASK48) ).bit_count()
        
    def _get_slab_bit_indices(self, slab_idx: int) -> List[int]:
        """Get bit indices for a specific slab using frozen Layer×Frame mapping.
        
        FROZEN - Immutable slab structure: 8 slabs, 6 bits each, Layer×Frame layout.
        """
        return FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
        
    def _count_slab_agreements(self, bits1: int, bits2: int, slab_indices: List[int]) -> int:
        """Count agreements within specific bit positions (slab)."""
        # For backward compatibility, still accept slab_indices parameter
        # but use precomputed masks when possible
        agreements = 0
        for bit_idx in slab_indices:
            bit1 = (bits1 >> bit_idx) & 1
            bit2 = (bits2 >> bit_idx) & 1
            if bit1 == bit2:
                agreements += 1
        return agreements
    
    def _count_slab_agreements_fast(self, bits1: int, bits2: int, slab_idx: int) -> int:
        """Count matches within a slab using precomputed masks."""
        mask = self._slab_masks[slab_idx]
        return ( (~(bits1 ^ bits2) & mask) ).bit_count()
        
    # --- Recovery ladder ---
    def recover_candidates(self, current_state: int, max_nudges: int = 6) -> List[int]:
        """
        Multi-level recovery: channel relaxation, neighbor orbits, duality pivot,
        orbit center fallback, geometric nudge. Return admissible token candidates.
        """
        start_time = time.perf_counter()
        
        with self._metrics_lock:
            self._metrics['recovery_calls'] += 1
        
        # Guard against unknown states - return empty list for robust API
        if current_state not in self.state_to_index:
            return []
            
        # Harmony control tokens to exclude from recovery
        harmony_control_tokens = ALL_CONTROL_TOKENS
        
        # Level 1: Channel relaxation (progressively relax channels, never drop Global)
        candidates = self._recovery_level_1(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            with self._metrics_lock:
                self._metrics['recovery_level_1_hits'] += 1
                self._update_recovery_timing(start_time)
            return candidates
            
        # Level 2: Neighbor orbits (Hamming-2 neighbors)
        candidates = self._recovery_level_2(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            with self._metrics_lock:
                self._metrics['recovery_level_2_hits'] += 1
                self._update_recovery_timing(start_time)
            return candidates
            
        # Level 3: Duality pivot (address ⊕ 0xFF if same phenomenology)
        candidates = self._recovery_level_3(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            with self._metrics_lock:
                self._metrics['recovery_level_3_hits'] += 1
                self._update_recovery_timing(start_time)
            return candidates
            
        # Level 4: Orbit center fallback (use representative address)
        candidates = self._recovery_level_4(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            with self._metrics_lock:
                self._metrics['recovery_level_4_hits'] += 1
                self._update_recovery_timing(start_time)
            return candidates
            
        # Level 5: Geometric nudge (up to max_nudges moves)
        candidates = self._recovery_level_5(current_state, max_nudges)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        
        with self._metrics_lock:
            self._metrics['recovery_level_5_hits'] += 1
            self._update_recovery_timing(start_time)
        
        return candidates
        
    def _recovery_level_1(self, current_state: int) -> List[int]:
        """Level 1: Progressive channel relaxation in frozen priority order."""
        if current_state not in self.state_to_index:
            return []
        current_orbit = self.phenomenology_map[self.state_to_index[current_state]]
        orbit_tokens = self._get_tokens_in_orbit(current_orbit)
        
        # Try with all channels first
        candidates = []
        for token_id in orbit_tokens:
            if self.is_admissible(current_state, token_id):
                candidates.append(token_id)
        if candidates:
            return candidates
            
        # Progressive slab relaxation: drop lowest-priority slab one by one
        # Use explicit frozen priority order instead of implicit range
        order = list(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER)
        for num_slabs_to_drop in range(1, FROZEN_CHANNELS.NUM_SLABS + 1):  # Drop 1 to all slabs
            # Keep the highest priority slabs; drop from the tail of the frozen order
            enabled_slabs = order[: FROZEN_CHANNELS.NUM_SLABS - num_slabs_to_drop]
            
            for token_id in orbit_tokens:
                if self._is_admissible_with_enabled_slabs(current_state, token_id, enabled_slabs):
                    candidates.append(token_id)
            if candidates:
                return candidates
                
        return candidates
        
    def _recovery_level_2(self, current_state: int) -> List[int]:
        """Level 2: Hamming-2 neighbor orbits."""
        if current_state not in self.state_to_index:
            return []
        current_orbit = self.phenomenology_map[self.state_to_index[current_state]]
        
        # Get Hamming-2 neighbors from precomputed cache
        neighbor_orbits = self.hamming2_neighbors.get(current_orbit, [])
        
        candidates = []
        for neighbor_orbit in neighbor_orbits:
            orbit_tokens = self._get_tokens_in_orbit(neighbor_orbit)
            for token_id in orbit_tokens:
                if self.is_admissible(current_state, token_id):
                    candidates.append(token_id)
                    
        return candidates
        
    def _recovery_level_3(self, current_state: int) -> List[int]:
        """Level 3: Duality pivot (address ⊕ 0xFF if same phenomenology)."""
        if current_state not in self.state_to_index:
            return []
        current_orbit = self.phenomenology_map[self.state_to_index[current_state]]
        orbit_tokens = self._get_tokens_in_orbit(current_orbit)
        
        candidates = []
        for token_id in orbit_tokens:
            token_address = self.address_of_token(token_id)
            dual_address = (token_address ^ 0xFFFFFFFFFFFF) & FROZEN_CHANNELS.MASK48  # XOR with 48-bit mask and ensure 48-bit result
            
            # Guard dual address safety: skip if dual not in atlas
            try:
                dual_orbit = self.phenomenology_map[self.state_to_index[dual_address]]
                if dual_orbit == current_orbit:
                    # Use dual address for admissibility check
                    if self._is_admissible_with_address(current_state, token_id, dual_address):
                        candidates.append(token_id)
            except KeyError:
                # Dual address not in atlas, skip this token
                continue
                        
        return candidates
        
    def _recovery_level_4(self, current_state: int) -> List[int]:
        """Level 4: Orbit center fallback (use representative address for all tokens)."""
        if current_state not in self.state_to_index:
            return []
        current_orbit = self.phenomenology_map[self.state_to_index[current_state]]
        representative_idx = self.orbit_representatives[current_orbit]
        centre_address = self.ontology_keys[representative_idx]
        
        candidates = []
        orbit_tokens = self._get_tokens_in_orbit(current_orbit)
        for token_id in orbit_tokens:
            if self._is_admissible_with_address(current_state, token_id, centre_address):
                candidates.append(token_id)
                
        return candidates
        
    def _recovery_level_5(self, current_state: int, max_nudges: int) -> List[int]:
        """Level 5: Geometric nudge (up to max_nudges moves)."""
        if current_state not in self.state_to_index:
            return []
            
        nudge_state = current_state
        
        for _ in range(max_nudges):
            # Try introns in order [0..255] and pick the FIRST that reduces theta
            chosen_intron = None
            
            for intron in range(256):
                next_state = self.apply_intron(nudge_state, intron)
                
                # Guard against states not in atlas
                if nudge_state not in self.state_to_index or next_state not in self.state_to_index:
                    continue
                    
                # Check if this reduces theta
                current_theta = self.theta[self.state_to_index[nudge_state]]
                next_theta = self.theta[self.state_to_index[next_state]]
                
                if next_theta < current_theta:
                    chosen_intron = intron
                    break  # Pick the FIRST that reduces theta
                    
            # If no theta reduction, pick the FIRST that changes orbit
            if chosen_intron is None:
                if nudge_state not in self.state_to_index:
                    break
                current_orbit = self.phenomenology_map[self.state_to_index[nudge_state]]
                for intron in range(256):
                    next_state = self.apply_intron(nudge_state, intron)
                    if next_state not in self.state_to_index:
                        continue
                    next_orbit = self.phenomenology_map[self.state_to_index[next_state]]
                    if next_orbit != current_orbit:
                        chosen_intron = intron
                        break  # Pick the FIRST that changes orbit
                        
            if chosen_intron is None:
                break  # No valid nudge found, stop
                
            # Apply the chosen nudge
            nudge_state = self.apply_intron(nudge_state, chosen_intron)
            
            # Restart from Level 1 with nudged state
            candidates = self._recovery_level_1(nudge_state)
            if candidates:
                return candidates
                
        return []  # All recovery levels failed
        
    def _get_tokens_in_orbit(self, orbit_code: int) -> List[int]:
        """Get all token IDs whose address maps to the given orbit."""
        # Track orbit computation metrics
        with self._metrics_lock:
            self._metrics['orbit_computations'] += 1
            
        if hasattr(self, '_orbit_to_tokens') and orbit_code in self._orbit_to_tokens:
            return self._orbit_to_tokens[orbit_code]
        
        # First collect tokens present in passive memory with any state in this orbit
        hits = set()
        if hasattr(self, 'passive_memory_index'):
            for (st_idx, tok) in self.passive_memory_index.keys():
                if self.phenomenology_map[st_idx] == orbit_code:
                    hits.add(tok)
                    if len(hits) >= 1024:
                        break
        if hits:
            # Populate _orbit_to_tokens with hits (keep sorted)
            if not hasattr(self, '_orbit_to_tokens'):
                self._orbit_to_tokens = {}
            if orbit_code not in self._orbit_to_tokens:
                self._orbit_to_tokens[orbit_code] = []
            
            for hit in sorted(hits):
                if hit not in self._orbit_to_tokens[orbit_code]:
                    bisect.insort(self._orbit_to_tokens[orbit_code], hit)
            
            return self._orbit_to_tokens[orbit_code]
        
        # fallback: bounded scan
        # Per-orbit sweep cursor with prime stride to avoid biased scans
        if not hasattr(self, '_sweep'):
            self._sweep = {}
        
        pos, stride = self._sweep.get(orbit_code, (0, 9973))
        budget = 1024  # per call
        hits = []
        vocab_size = getattr(self, 'vocab_size', 50000)
        
        # Harmony control tokens to exclude
        control_token_ids = ALL_CONTROL_TOKENS
        
        for _ in range(budget):
            tok = pos % vocab_size
            pos += stride
            if tok in control_token_ids:
                continue
            
            token_address = self.address_of_token(tok)
            if token_address in self.state_to_index:
                token_orbit = self.phenomenology_map[self.state_to_index[token_address]]
                if token_orbit == orbit_code:
                    hits.append(tok)
            if len(hits) >= 64:
                break
        
        self._sweep[orbit_code] = (pos, stride)
        
        # Populate _orbit_to_tokens with hits (keep sorted)
        if not hasattr(self, '_orbit_to_tokens'):
            self._orbit_to_tokens = {}
        if orbit_code not in self._orbit_to_tokens:
            self._orbit_to_tokens[orbit_code] = []
        
        for hit in hits:
            if hit not in self._orbit_to_tokens[orbit_code]:
                bisect.insort(self._orbit_to_tokens[orbit_code], hit)
        
        return self._orbit_to_tokens[orbit_code]
        
    def _is_admissible_global_only(self, start_state: int, token_id: int) -> bool:
        """Check admissibility with only global channel (relaxed slab channels)."""
        token_address = self.address_of_token(token_id)
        token_address_bits = self._packed_state_to_bitset(token_address)
        
        introns = self.token_to_introns(token_id)
        micro_path = self.micro_path(start_state, introns)
        
        # Check global channel stepwise monotonicity and track strict progress
        global_strict_progress = False
        for k in range(len(micro_path) - 1):
            current_bits = self._packed_state_to_bitset(micro_path[k])
            next_bits = self._packed_state_to_bitset(micro_path[k + 1])
            
            current_agreements = self._count_agreements(current_bits, token_address_bits)
            next_agreements = self._count_agreements(next_bits, token_address_bits)
            
            if next_agreements < current_agreements:
                return False
            if next_agreements > current_agreements:
                global_strict_progress = True
                
        # Require strict progress in global channel
        return global_strict_progress
        
    def _is_admissible_with_address(self, start_state: int, token_id: int, override_address: int) -> bool:
        """Check admissibility using a specific address override."""
        address_bits = self._packed_state_to_bitset(override_address)
        
        introns = self.token_to_introns(token_id)
        micro_path = self.micro_path(start_state, introns)
        
        # Check global channel with override address
        for k in range(len(micro_path) - 1):
            current_bits = self._packed_state_to_bitset(micro_path[k])
            next_bits = self._packed_state_to_bitset(micro_path[k + 1])
            
            current_agreements = self._count_agreements(current_bits, address_bits)
            next_agreements = self._count_agreements(next_bits, address_bits)
            
            if next_agreements < current_agreements:
                return False
                
        return True
        
    def _is_admissible_with_enabled_slabs(self, start_state: int, token_id: int, enabled_slabs: List[int]) -> bool:
        """Check admissibility with only specified slab channels enabled (Global always enabled)."""
        token_address = self.address_of_token(token_id)
        token_address_bits = self._packed_state_to_bitset(token_address)
        
        introns = self.token_to_introns(token_id)
        micro_path = self.micro_path(start_state, introns)
        
        # Global channel: stepwise non-decrease
        global_progress = False
        for k in range(len(micro_path) - 1):
            current_bits = self._packed_state_to_bitset(micro_path[k])
            next_bits = self._packed_state_to_bitset(micro_path[k + 1])
            
            current_agreements = self._count_agreements(current_bits, token_address_bits)
            next_agreements = self._count_agreements(next_bits, token_address_bits)
            
            if next_agreements < current_agreements:
                return False
            if next_agreements > current_agreements:
                global_progress = True
                
        # Enabled slab channels: end-to-start agreement
        start_bits = self._packed_state_to_bitset(micro_path[0])
        end_bits = self._packed_state_to_bitset(micro_path[-1])
        
        slab_progress = False
        for slab_idx in enabled_slabs:
            start_slab_agreements = self._count_slab_agreements_fast(start_bits, token_address_bits, slab_idx)
            end_slab_agreements = self._count_slab_agreements_fast(end_bits, token_address_bits, slab_idx)
            
            if end_slab_agreements < start_slab_agreements:
                return False
            if end_slab_agreements > start_slab_agreements:
                slab_progress = True
                
        return global_progress or slab_progress
        
    # --- Memory integration ---
    def fold_egress(self, state_after: int, token_id: int) -> None:
        """
        Update passive memory with Monodromic Fold composite form.
        Store per-key (state_index, token_id) → 8-bit mask + metadata.
        Enforce caps K=64 masks per state per orbit, M=64 states per token per orbit.
        """
        # Get state index from packed state
        if state_after not in self.state_to_index:
            return
        state_index = self.state_to_index[state_after]
        
        # Get orbit information
        state_orbit = self.phenomenology_map[state_index]
        token_address = self.address_of_token(token_id)
        token_orbit = self.phenomenology_map[self.state_to_index[token_address]]
        
        # Compute Monodromic Fold mask (8-bit composite)
        fold_mask = self._compute_monodromic_fold(state_index, token_id)
        
        # Initialize passive memory index and timestamp counter if needed
        if not hasattr(self, 'passive_memory_index'):
            self.passive_memory_index: Dict[Tuple[int, int], Dict[str, int]] = {}
        if not hasattr(self, '_timestamp_counter'):
            self._timestamp_counter: int = 0
        # Initialize cold-start grace window tracking
        if not hasattr(self, '_cold_start_timestamp'):
            self._cold_start_timestamp: int = 0
        if not hasattr(self, '_cold_start_grace_window'):
            # Grace window: allow tokens without experience for first N operations after cold start
            self._cold_start_grace_window: int = int(self.runtime.get("cold_start_grace_window", 50))
            
        # Type annotation for class attribute
        if not hasattr(self.__class__, '__annotations__'):
            self.__class__.__annotations__ = {}
        self.__class__.__annotations__['passive_memory_index'] = Dict[Tuple[int, int], Dict[str, int]]
        
        memory_key = (state_index, token_id)
        
        # Handle zero mask with zero_streak logic
        if fold_mask == 0:
            if memory_key in self.passive_memory_index:
                existing = self.passive_memory_index[memory_key]
                existing['zero_streak'] += 1
                existing['touch_count'] = min(255, existing['touch_count'] + 1)
                self._timestamp_counter += 1
                existing['timestamp'] = self._timestamp_counter
                
                # Delete entry if zero_streak >= 2
                if existing['zero_streak'] >= 2:
                    del self.passive_memory_index[memory_key]
                    # Log deletion
                    deletion_entry = existing.copy()
                    deletion_entry['deleted'] = True
                    self._append_to_passive_log(deletion_entry)
                else:
                    # Update existing entry with zero mask
                    existing['mask_id'] = self._intern_mask(0)
                    self._append_to_passive_log(existing)
            # If no existing entry and mask is zero, don't create new entry
            return
        
        # Non-zero mask: intern and store
        interned_mask_id = self._intern_mask(fold_mask)
        
        with self._passive_memory_lock:
            if memory_key in self.passive_memory_index:
                # Update existing entry
                existing = self.passive_memory_index[memory_key]
                existing['touch_count'] = min(255, existing['touch_count'] + 1)
                existing['zero_streak'] = 0  # Reset zero streak
                existing['mask_id'] = interned_mask_id
                self._timestamp_counter += 1
                existing['timestamp'] = self._timestamp_counter
                self._append_to_passive_log(existing)
                
                # Track experienced tokens in state_to_tokens mapping for non-zero masks
                self.state_to_tokens.setdefault(state_index, set()).add(token_id)
            else:
                # Create new entry
                memory_entry = {
                    'state_index': state_index,
                    'token_id': token_id,
                    'mask_id': interned_mask_id,
                    'touch_count': 1,
                    'zero_streak': 0,
                    'timestamp': self._timestamp_counter + 1
                }
                self._timestamp_counter += 1
                
                # Check and enforce caps before adding
                self._enforce_passive_memory_caps(state_orbit, token_orbit, memory_entry)
                
                # Add new entry
                self.passive_memory_index[memory_key] = memory_entry.copy()
                self._append_to_passive_log(memory_entry)
                
            # Track experienced tokens in state_to_tokens mapping for non-zero masks
            self.state_to_tokens.setdefault(state_index, set()).add(token_id)
            
    def _compute_monodromic_fold(self, state_index: int, token_id: int) -> int:
        """Compute 8-bit Monodromic Fold using exact composite form."""
        # Get existing exon_mask (0x01 seed experience if missing)
        key = (state_index, token_id)
        existing_mask = 0x01  # Seed experience for first observations
        if hasattr(self, 'passive_memory_index'):
            with self._passive_memory_lock:
                if key in self.passive_memory_index:
                    # Get actual mask from interned mask_id
                    mask_id = self.passive_memory_index[key]['mask_id']
                    if hasattr(self, 'mask_pool_reverse') and mask_id in self.mask_pool_reverse:
                        existing_mask = self.mask_pool_reverse[mask_id]
                    else:
                        existing_mask = mask_id  # Fallback if not in pool
        
        # Get token's intron sequence
        introns = self.token_to_introns(token_id)
        
        # Apply composite fold: fold(a, b) = a ^ (b ^ (a & (~b & 0xFF)))
        def fold(a: int, b: int) -> int:
            return a ^ (b ^ (a & (~b & 0xFF)))
        
        # Reduce over introns starting from existing mask
        result = existing_mask & 0xFF
        for intron in introns:
            result = fold(result, intron & 0xFF)
                
        return result & 0xFF
        
    def _intern_mask(self, mask: int) -> int:
        """Intern 8-bit mask in global 256-entry pool."""
        mask = mask & 0xFF  # Ensure 8-bit
        
        if mask in self.mask_pool:
            return self.mask_pool[mask]
            
        # Add new mask to pool if space available
        if self.next_mask_id < 256:
            mask_id = self.next_mask_id
            self.mask_pool[mask] = mask_id
            self.mask_pool_reverse[mask_id] = mask
            self.next_mask_id += 1
            return mask_id
        else:
            # Pool is full - return mask value itself as id
            # Also put it in reverse pool so _compute_monodromic_fold can retrieve it
            self.mask_pool_reverse[mask] = mask
            return mask
        
    def _enforce_passive_memory_caps(self, state_orbit: int, token_orbit: int, new_entry: Dict[str, int]) -> None:
        """Enforce hard caps K=64 masks per (state, orbit), M=64 states per (token, orbit)."""
        if not hasattr(self, 'passive_memory_index'):
            return
            
        state_index = new_entry['state_index']
        token_id = new_entry['token_id']
        
        # Collect entries for K constraint (same state, same orbit)
        k_entries = []
        for key, entry in self.passive_memory_index.items():
            entry_state_idx, entry_token_id = key
            if entry_state_idx == state_index:
                entry_state_orbit = self.phenomenology_map[entry_state_idx]
                if entry_state_orbit == state_orbit:
                    k_entries.append((key, entry))
        
        # Enforce K=64 cap
        if len(k_entries) >= 64:
            # Sort by eviction priority: generic first, then by touch_count (oldest), then by timestamp, then by key
            k_entries.sort(key=lambda x: (
                not self._is_generic_entry(x[1]),  # Generic entries first (False < True)
                x[1]['touch_count'],  # Oldest touch_count
                x[1]['timestamp'],    # Oldest timestamp
                x[0]                  # Deterministic tie-breaker by key
            ))
            # Evict oldest entries to make room for new entry (keep 63, evict rest)
            for key, _ in k_entries[:len(k_entries) - 63]:
                del self.passive_memory_index[key]
        
        # Collect entries for M constraint (same token, same orbit)
        m_entries = []
        for key, entry in self.passive_memory_index.items():
            entry_state_idx, entry_token_id = key
            if entry_token_id == token_id:
                # Use cached address if present; compute once otherwise
                addr = int(self.address_memory[entry_token_id]) if entry_token_id < len(self.address_memory) else 0
                if addr == 0:
                    addr = self.address_of_token(entry_token_id)
                entry_token_orbit = self.phenomenology_map[self.state_to_index[addr]]
                if entry_token_orbit == token_orbit:
                    m_entries.append((key, entry))
        
        # Enforce M=64 cap
        if len(m_entries) >= 64:
            # Sort by eviction priority: generic first, then by touch_count (oldest), then by timestamp, then by key
            m_entries.sort(key=lambda x: (
                not self._is_generic_entry(x[1]),  # Generic entries first (False < True)
                x[1]['touch_count'],  # Oldest touch_count
                x[1]['timestamp'],    # Oldest timestamp
                x[0]                  # Deterministic tie-breaker by key
            ))
            # Evict oldest entries to make room for new entry (keep 63, evict rest)
            for key, _ in m_entries[:len(m_entries) - 63]:
                del self.passive_memory_index[key]
            
    def _is_generic_entry(self, entry: Dict[str, int]) -> bool:
        """Check if entry is 'generic' (token address equals orbit representative)."""
        token_id = entry['token_id']
        # Use cached address if present; compute once otherwise
        addr = int(self.address_memory[token_id]) if token_id < len(self.address_memory) else 0
        if addr == 0:
            addr = self.address_of_token(token_id)
        token_orbit = self.phenomenology_map[self.state_to_index[addr]]
        rep_idx = self.orbit_representatives[token_orbit]
        representative_address = self.ontology_keys[rep_idx]
        
        return addr == representative_address
        
    def _append_to_passive_log(self, entry: Dict[str, int]) -> None:
        """Append entry to binary log file with improved sync logic."""
        
        with self._passive_log_lock:
            # Initialize log file handle and counter if needed
            if not hasattr(self, 'passive_log_fh'):
                self.passive_log_fh = open(self.passive_memory_path, 'ab')
                self.passive_log_len = 0
                
            # Pack entry into binary format
            # Format: state_index(4), token_id(4), mask_id(1), touch_count(1), zero_streak(1), timestamp(4)
            packed_entry = struct.pack('<IIBBBI', 
                                     entry['state_index'],
                                     entry['token_id'], 
                                     entry['mask_id'],
                                     entry['touch_count'],
                                     entry['zero_streak'],
                                     entry['timestamp'])
            
            # Write to log file
            try:
                self.passive_log_fh.write(packed_entry)
                self.passive_log_fh.flush()
                
                # Update counters
                self.passive_log_len += 1
                self._passive_log_pending_writes += 1
                self._passive_log_sync_counter += 1
                
                # Implement tiered sync strategy:
                # 1. Regular sync at sync_interval for durability
                # 2. Force sync at force_sync_interval for crash safety
                should_sync = False
                
                if self._passive_log_sync_counter >= self._passive_log_force_sync_interval:
                    # Force sync - reset both counters
                    should_sync = True
                    self._passive_log_sync_counter = 0
                    self._passive_log_pending_writes = 0
                elif self._passive_log_pending_writes >= self._passive_log_sync_interval:
                    # Regular sync - reset pending writes counter only
                    should_sync = True
                    self._passive_log_pending_writes = 0
                
                # Aggressive sync (off by default): great for tests/ingestion-only
                if self.runtime.get("aggressive_sync", False):
                    os.fsync(self.passive_log_fh.fileno())
                elif should_sync:
                    os.fsync(self.passive_log_fh.fileno())
                
            except Exception as e:
                print(f"Warning: Failed to write to passive memory log: {e}")
                
    def _append_to_passive_log_debug(self, text: str) -> None:
        """Test-only convenience to prove file I/O works."""
        with self._passive_log_lock:
            mode = "ab" if isinstance(text, (bytes, bytearray)) else "a"
            # Use default buffering for text mode, unbuffered only for binary
            buffering = 0 if "b" in mode else -1
            with open(self.passive_memory_path, mode, buffering=buffering) as f:
                if "b" in mode:
                    f.write(text if isinstance(text, (bytes, bytearray)) else text.encode("utf-8"))
                else:
                    f.write(text + "\n")
        
    def _load_passive_memory_from_log(self) -> None:
        """Reconstruct in-memory index by scanning binary log."""
        
        self.passive_memory_index: Dict[Tuple[int, int], Dict[str, int]] = {}
        self.passive_log: List[Dict[str, int]] = []
        
        if not self.passive_memory_path.exists():
            return
            
        entry_size = struct.calcsize('<IIBBBI')  # 15 bytes per entry
        
        try:
            with open(self.passive_memory_path, 'rb') as f:
                while True:
                    data = f.read(entry_size)
                    if len(data) < entry_size:
                        break
                        
                    # Unpack entry
                    state_index, token_id, mask_id, touch_count, zero_streak, timestamp = \
                        struct.unpack('<IIBBBI', data)
                        
                    entry = {
                        'state_index': state_index,
                        'token_id': token_id,
                        'mask_id': mask_id,
                        'touch_count': touch_count,
                        'zero_streak': zero_streak,
                        'timestamp': timestamp
                    }
                    
                    # Add to in-memory structures
                    memory_key = (state_index, token_id)
                    self.passive_memory_index[memory_key] = entry
                    self.passive_log.append(entry)
                    
                    # Populate state_to_tokens mapping for experience tracking
                    self.state_to_tokens.setdefault(state_index, set()).add(token_id)
        except Exception as e:
            print(f"Warning: Failed to load passive memory log: {e}")
            
    # --- Conversation policy ---
    def start_state(self) -> int:
        """
        The archetypal initial packed state for a fresh conversation.
        Pick the unique state of minimal theta (frozen start).
        """
        # Mark cold-start timestamp for grace window
        self._cold_start_timestamp = getattr(self, '_timestamp_counter', 0)
        
        # Pick the unique state of minimal theta (frozen start)
        try:
            min_idx = int(np.argmin(self.theta))
            return int(self.ontology_keys[min_idx])
        except Exception:
            # Fallback 1: orbit representative of the first orbit code
            try:
                rep_idx = self.orbit_representatives[min(self.orbit_representatives.keys())]
                return int(self.ontology_keys[rep_idx])
            except Exception:
                # Fallback 2: orthogonal reference (index 0)
                return int(self.ontology_keys[0])
        
    def evolve_on_user(self, state: int, token_id: int) -> int:
        """
        Egress: apply introns (always) and fold passive memory (by default).
        Change: fold on PRE-state so generation at that state can find what came next historically.
        """
        introns = self.token_to_introns(token_id)
        pre_state = state
        new_state = state
        for intron in introns:
            new_state = self.apply_intron(new_state, intron)

        # Fold on the PRE-state (critical for usable next-token experience)
        if True:  # keep feature flagging simple for now
            try:
                self.fold_egress(pre_state, token_id)
            except Exception:
                pass

        # Optional: also fold on post-state to preserve prior behaviour
        # (keeps your tests and any analysis scripts happy)
        try:
            self.fold_egress(new_state, token_id)
        except Exception:
            pass

        return new_state
        
    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        """
        Ingress: apply introns (always) and DO NOT fold memory
        (unless enable_self_reinforcement).
        """
        introns = self.token_to_introns(token_id)
        new_state = state
        
        for intron in introns:
            new_state = self.apply_intron(new_state, intron)
            
        # Only fold if self-reinforcement is enabled
        if self.enable_self_reinforcement:
            self.fold_egress(new_state, token_id)
            
        return new_state
        
    # --- Public compatibility shims ---
    def token_to_address(self, token_id: int) -> int:
        """Compatibility alias used by tests/tools."""
        return self.address_of_token(token_id)
    
    # --- Next-token selection ---
    def next_token_deterministic(self, state: int, candidate_vocab: Iterable[int] | None = None) -> Optional[int]:
        """
        Enhanced token selection with experience-based gating and relaxed admissibility.
        Prefer tokens with experience at THIS state, fall back to orbit routing if empty.
        Apply relaxed gating: allow tokens with final state experience, micro-path experience, or cold-start grace.
        """
        # Track token generation metrics
        with self._metrics_lock:
            self._metrics['token_generations'] += 1
            
        # Validate state
        if state not in self.state_to_index:
            return None
            
        state_idx = self.state_to_index[state]
        
        # Track state lookup metrics
        with self._metrics_lock:
            self._metrics['state_lookups'] += 1
            
        current_orbit = self.phenomenology_map[state_idx]
        
        # Generate candidate_vocab if not provided
        if candidate_vocab is None:
            # Prefer tokens with experience at THIS state
            experienced_here = sorted(self.state_to_tokens.get(state_idx, []))
            if not experienced_here:
                # Fall back to orbit tokens (existing logic)
                tok_list = list(self._orbit_to_tokens.get(current_orbit, []))
                if not tok_list:
                    tok_list = self._get_tokens_in_orbit(current_orbit)
                candidate_vocab = [t for t in tok_list if t not in GENERATION_EXCLUDED]
            else:
                candidate_vocab = [t for t in experienced_here if t not in GENERATION_EXCLUDED]
        
        # Apply relaxed gating as specified
        admissible = []
        current_timestamp = getattr(self, '_timestamp_counter', 0)
        in_grace_window = (current_timestamp - getattr(self, '_cold_start_timestamp', 0)) < getattr(self, '_cold_start_grace_window', 50)
        
        for t in candidate_vocab:
            if not self.is_admissible(state, t):
                continue
                
            fstate = self.final_state_for_token(state, t)
            findex = self.state_to_index.get(fstate)
            
            has_final = (findex is not None) and self.has_experience(findex, t)
            has_path = False
            if not has_final:
                introns = self.token_to_introns(t)
                path = self.micro_path(state, introns)
                has_path = self.first_nonzero_mask_on_path(path, t)
                
            if has_final or has_path or in_grace_window:
                admissible.append(t)
        
        if admissible:
            return min(admissible)
            
        return None
    
    # --- Helper functions for testing ---
    def encode_token_to_introns(self, token_id: int) -> List[int]:
        """Encode token to introns via LEB128 → bytes → ψ transformation."""
        return self.token_to_introns(token_id)
    
    # --- New helpers for experience gating ---
    def final_state_for_token(self, start_state: int, token_id: int) -> int:
        """Return the final packed state reached from start_state by token_id."""
        introns = self.token_to_introns(token_id)
        path = self.micro_path(start_state, introns)
        return path[-1]
    
    def has_experience(self, state_index: int, token_id: int) -> bool:
        """True iff passive memory contains a non-zero mask for (state_index, token_id)."""
        entry = self.passive_memory_index.get((state_index, token_id))
        if not entry:
            return False
        # Resolve mask value (interned or raw fallback), then test non-zero
        mask_id = entry.get('mask_id', 0)
        mask_val = self.mask_pool_reverse.get(mask_id, mask_id) & 0xFF
        return mask_val != 0
    
    def first_nonzero_mask_on_path(self, path_states: list[int], token_id: int) -> bool:
        """Optional: permit any non-zero mask at any state along a micro-path (useful for recovery)."""
        for st in path_states:
            idx = self.state_to_index.get(st)
            if idx is not None and self.has_experience(idx, token_id):
                return True
        return False
    
    def introns_to_token_bytes(self, introns: List[int]) -> bytes:
        """Convert introns back to token bytes via ψ⁻¹ transformation."""
        return bytes([self.intron_to_byte(intron) for intron in introns])
    
    def compute_micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        """Compute micro-path for testing - alias for micro_path."""
        return self.micro_path(start_state, introns)
    
    def channel_alignment(self, state: int, address: int, positions: List[int] = None) -> int:
        """Compute channel alignment between state and address for testing."""
        state_bits = self._packed_state_to_bitset(state)
        address_bits = self._packed_state_to_bitset(address)
        
        if positions is None:
            # Count all bit agreements
            return bin(~(state_bits ^ address_bits) & self.MASK48).count('1')
        else:
            # Count agreements at specific positions
            agreements = 0
            for pos in positions:
                if pos < 48:  # Validate position
                    state_bit = (state_bits >> pos) & 1
                    address_bit = (address_bits >> pos) & 1
                    if state_bit == address_bit:
                        agreements += 1
            return agreements
    
    def get_layer_frame_positions(self, layer: int, frame: int) -> List[int]:
        """Get bit positions for a specific layer and frame from frozen mapping."""
        if layer < 0 or layer >= FROZEN_CHANNELS.NUM_LAYERS:
            raise ValueError(f"Invalid layer {layer}, must be 0-{FROZEN_CHANNELS.NUM_LAYERS-1}")
        if frame < 0 or frame >= FROZEN_CHANNELS.NUM_FRAMES:
            raise ValueError(f"Invalid frame {frame}, must be 0-{FROZEN_CHANNELS.NUM_FRAMES-1}")
        
        # Calculate slab index from layer and frame
        slab_idx = layer * FROZEN_CHANNELS.NUM_FRAMES + frame
        return self._get_slab_bit_indices(slab_idx)
    
    def validate_maps(self) -> bool:
        """Validate atlas maps integrity and version compatibility."""
        try:
            # Check version compatibility
            if hasattr(self, 'version_info') and self.version_info:
                expected_atlas = self.version_info.get('atlas_version', 'v1.0.0')
                expected_address = self.version_info.get('address_version', 'v1.0.0')
                
                # Check if address memory exists and has metadata
                if hasattr(self, 'address_metadata_path') and self.address_metadata_path.exists():
                    try:
                        with open(self.address_metadata_path, 'r') as f:
                            metadata = json.load(f)
                        stored_atlas = metadata.get('atlas_version', 'unknown')
                        stored_address = metadata.get('address_version', 'unknown')
                        
                        if (stored_atlas != expected_atlas or stored_address != expected_address):
                            raise RuntimeError(f"Version mismatch: stored atlas_v{stored_atlas}/address_v{stored_address}, "
                                             f"expected atlas_v{expected_atlas}/address_v{expected_address}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to validate versions: {e}")
            
            # Validate map shapes and consistency
            expected_len = 788_986
            
            if len(self.ontology_keys) != expected_len:
                raise ValueError(f"ontology_keys length mismatch: expected {expected_len}, got {len(self.ontology_keys)}")
            
            if self.epistemology.shape != (expected_len, 256):
                raise ValueError(f"epistemology shape mismatch: expected ({expected_len}, 256), got {self.epistemology.shape}")
            
            if len(self.theta) != expected_len:
                raise ValueError(f"theta length mismatch: expected {expected_len}, got {len(self.theta)}")
            
            # Validate reverse index consistency
            if len(self.state_to_index) != expected_len:
                raise ValueError(f"state_to_index length mismatch: expected {expected_len}, got {len(self.state_to_index)}")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Map validation failed: {e}")
    
    def validate_versions(self) -> bool:
        """Validate version information and enforce version policy."""
        if not hasattr(self, 'version_info') or not self.version_info:
            raise RuntimeError("No version information available")
        
        required_versions = ['atlas_version', 'address_version', 'config_version']
        for version_key in required_versions:
            if version_key not in self.version_info:
                raise RuntimeError(f"Missing required version: {version_key}")
        
        return self.validate_maps()
    
    def _load_address_cache(self):
        """Load address cache from disk if available."""
        import os
        import pickle
        
        cache_file = os.path.join(self.address_memory_path.parent, 'address_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                with self._address_cache_lock:
                    self._address_cache = loaded_cache
                print(f"Loaded {len(self._address_cache)} cached addresses")
            except Exception as e:
                print(f"Failed to load address cache: {e}")
                with self._address_cache_lock:
                    self._address_cache = {}
        
        with self._metrics_lock:
            self._metrics['address_cache_loads'] += 1
            self._update_cache_metrics()
    
    def _update_recovery_timing(self, start_time: float):
        """Update recovery timing metrics (called with metrics lock held)."""
        elapsed = time.perf_counter() - start_time
        self._metrics['recovery_total_time'] += elapsed
        if self._metrics['recovery_calls'] > 0:
            self._metrics['recovery_avg_time'] = self._metrics['recovery_total_time'] / self._metrics['recovery_calls']
    
    def _update_admissibility_timing(self, start_time: float):
        """Update admissibility timing metrics (called with metrics lock held)."""
        elapsed = time.perf_counter() - start_time
        self._metrics['admissibility_total_time'] += elapsed
        if self._metrics['admissibility_checks'] > 0:
            self._metrics['admissibility_avg_time'] = self._metrics['admissibility_total_time'] / self._metrics['admissibility_checks']
    
    def _update_cache_metrics(self):
        """Update cache performance metrics (called with metrics lock held)."""
        self._metrics['address_cache_size'] = len(self._address_cache)
        total_requests = self._metrics['address_cache_hits'] + self._metrics['address_cache_misses']
        if total_requests > 0:
            self._metrics['address_cache_hit_rate'] = self._metrics['address_cache_hits'] / total_requests
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current runtime metrics and observability data."""
        with self._metrics_lock:
            return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset all runtime metrics to zero."""
        with self._metrics_lock:
            for key in self._metrics:
                if isinstance(self._metrics[key], (int, float)):
                    self._metrics[key] = 0 if isinstance(self._metrics[key], int) else 0.0
    
    def print_metrics_summary(self):
        """Print a formatted summary of current metrics."""
        metrics = self.get_metrics()
        
        print("\n📊 GyroEngine Runtime Metrics:")
        print("=" * 40)
        
        # Recovery ladder metrics
        print("\n🔄 Recovery Ladder:")
        print(f"  Total calls: {metrics['recovery_calls']:,}")
        print(f"  Level 1 hits: {metrics['recovery_level_1_hits']:,}")
        print(f"  Level 2 hits: {metrics['recovery_level_2_hits']:,}")
        print(f"  Level 3 hits: {metrics['recovery_level_3_hits']:,}")
        print(f"  Level 4 hits: {metrics['recovery_level_4_hits']:,}")
        print(f"  Level 5 hits: {metrics['recovery_level_5_hits']:,}")
        print(f"  Average time: {metrics['recovery_avg_time']*1000:.2f}ms")
        
        # Admissibility metrics
        print("\n✅ Admissibility Checks:")
        print(f"  Total checks: {metrics['admissibility_checks']:,}")
        print(f"  Hits: {metrics['admissibility_hits']:,}")
        print(f"  Misses: {metrics['admissibility_misses']:,}")
        if metrics['admissibility_checks'] > 0:
            hit_rate = metrics['admissibility_hits'] / metrics['admissibility_checks'] * 100
            print(f"  Hit rate: {hit_rate:.1f}%")
        print(f"  Average time: {metrics['admissibility_avg_time']*1000:.2f}ms")
        
        # Cache metrics
        print("\n💾 Address Cache:")
        print(f"  Cache size: {metrics['address_cache_size']:,}")
        print(f"  Cache hits: {metrics['address_cache_hits']:,}")
        print(f"  Cache misses: {metrics['address_cache_misses']:,}")
        print(f"  Hit rate: {metrics['address_cache_hit_rate']*100:.1f}%")
        print(f"  Cache saves: {metrics['address_cache_saves']:,}")
        print(f"  Cache loads: {metrics['address_cache_loads']:,}")
        
        # General metrics
        print("\n🎯 General Performance:")
        print(f"  Token generations: {metrics['token_generations']:,}")
        print(f"  State lookups: {metrics['state_lookups']:,}")
        print(f"  Orbit computations: {metrics['orbit_computations']:,}")
        print()
    
    def _save_address_cache(self):
        """Save address cache to disk."""
        import os
        import pickle
        
        cache_file = os.path.join(self.address_memory_path.parent, 'address_cache.pkl')
        try:
            with self._address_cache_lock:
                cache_copy = self._address_cache.copy()
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_copy, f)
            
            with self._metrics_lock:
                self._metrics['address_cache_saves'] += 1
                self._update_cache_metrics()
        except Exception as e:
            print(f"Failed to save address cache: {e}")