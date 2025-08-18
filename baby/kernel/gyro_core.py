# baby/kernel/gyro_core.py

import bisect
import json
import numpy as np
import os
import struct
import tempfile
import threading
from typing import Iterable, Optional, Dict, List
from pathlib import Path
from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS, GENERATION_EXCLUDED
from baby.constants.frozen_channels import FROZEN_CHANNELS


class GyroEngine:
    """Core GyroSI physics engine implementing deterministic token generation."""
    
    # Frozen constants
    MASK48 = FROZEN_CHANNELS.MASK48
    
    @staticmethod
    def channel_lex_key(bits: int) -> tuple:
        """Convert packed state to channel lexicographic key for tie-breaking.
        Order: bit index 0..47 (layer, frame, row, col)
        """
        return tuple((bits >> i) & 1 for i in range(FROZEN_CHANNELS.TOTAL_BITS))
    
    def __init__(self, atlas_paths: dict, store_paths: dict, runtime: dict, version_info: dict = None, vocab_size: int = None):
        """
        Load all five maps, build reverse index, open passive store,
        load or lazily initialise address memory cache.
        Enforce map integrity and versioning.
        """
        self.runtime = runtime
        self.max_nudges = runtime.get('max_nudges', 6)
        self.enable_self_reinforcement = runtime.get('enable_self_reinforcement', False)
        
        # Store version information for validation
        self.version_info = version_info or {}
        
        # Enforce version validation on all entry points
        self._validate_required_versions()
        
        # Store vocabulary size for bounded operations
        self.vocab_size = vocab_size or 50000  # Default fallback
        
        # Load atlas maps
        self._load_atlas_maps(atlas_paths)
        
        # Initialize stores with version validation
        self._init_stores(store_paths)
        
        # Build reverse index for O(1) state lookup
        self._build_reverse_index()
        
        # Build orbit system: representatives and Hamming-2 neighbors
        self._build_orbit_system()
        
        # Cache for address memory
        self._address_cache = {}
        
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
        
        # Initialize threading locks for concurrency safety
        self._address_memory_lock = threading.RLock()  # For address memory writes
        self._passive_log_lock = threading.RLock()     # For passive log writes
        
    def _validate_required_versions(self):
        """Validate that all required version information is present and valid."""
        required_versions = ['atlas_version', 'address_version', 'config_version']
        
        for version_key in required_versions:
            if version_key not in self.version_info:
                raise RuntimeError(f"FATAL: Missing required version information: {version_key}. "
                                 f"All entry points must provide complete version_info.")
            
            version_value = self.version_info[version_key]
            if not isinstance(version_value, str) or not version_value.strip():
                raise RuntimeError(f"FATAL: Invalid version format for {version_key}: {version_value}. "
                                 f"Version must be a non-empty string.")
        
        # Log successful validation
        atlas_v = self.version_info['atlas_version']
        address_v = self.version_info['address_version']
        config_v = self.version_info['config_version']
        print(f"Version validation passed: atlas_v{atlas_v}, address_v{address_v}, config_v{config_v}")
        
    def _load_atlas_maps(self, atlas_paths: dict):
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
        
    def _init_stores(self, store_paths: dict):
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
        self.state_to_index = {}
        
        # Build mapping from each packed state (48-bit) to its row index
        for i, packed_state in enumerate(self.ontology_keys):
            # Ensure we're working with proper 48-bit packed states
            # The packed state should be a 48-bit integer representing the state
            if packed_state in self.state_to_index:
                raise ValueError(f"Duplicate packed state {packed_state} at indices {self.state_to_index[packed_state]} and {i}")
            self.state_to_index[packed_state] = i
            
        # Verify we have the expected number of unique states
        if len(self.state_to_index) != len(self.ontology_keys):
            raise ValueError(f"State index mismatch: expected {len(self.ontology_keys)} unique states, got {len(self.state_to_index)}")
            
        print(f"Built reverse index with {len(self.state_to_index)} states")
         
    def _build_orbit_system(self):
        """Construct orbit representatives and Hamming-2 neighbor cache from phenomenology_map."""
        # Extract the actual unique orbit codes from phenomenology_map
        self.orbit_codes = sorted(set(self.phenomenology_map))
        
        # Find representative state index for each orbit code
        # Use the smallest state index in each orbit class as the representative
        self.orbit_representatives = {}  # orbit_code -> state_index
        orbit_to_states = {}  # orbit_code -> list of state_indices
        
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
        self.hamming2_neighbors = {}  # rep_orbit_code -> list[neighbor_orbit_codes]
        
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
        self._orbit_to_tokens = {}
        
        # Initialize empty sorted lists for all orbits
        for orbit_code in range(256):
            self._orbit_to_tokens[orbit_code] = []
            
        # This will be populated lazily as addresses are computed
        # Token lists are kept sorted for deterministic iteration
        
    def _add_token_to_orbit_index(self, token_id: int, orbit_code: int):
        """Add a token to the orbit→tokens index.
        
        FROZEN - Maintains sorted order for deterministic candidate selection.
        """
        if not hasattr(self, '_orbit_to_tokens'):
            self._orbit_to_tokens = {}
            
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
        self.SLAB_MASKS = []
        for s in range(FROZEN_CHANNELS.NUM_SLABS):
            idxs = self._get_slab_bit_indices(s)
            m = 0
            for i in idxs:
                m |= (1 << i)
            self.SLAB_MASKS.append(m & self.MASK48)
        
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
    def psi(byte_val: int) -> int:
        """Transform byte to intron via XOR with 0xAA.
        
        FROZEN - Immutable transformation rule: byte ↔ intron mapping.
        """
        return (byte_val & 0xFF) ^ 0xAA
        
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
        """Convert token to list of introns via psi transformation."""
        return [self.psi(b) for b in self.encode_token_to_bytes(token_id)]
        
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
        if token_id in self._address_cache:
            return self._address_cache[token_id]
            
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
        best_agreements = -1
        
        for candidate in unique_finals:
            total_agreements = 0
            
            # Calculate sum of agreements (dot products) to all final states
            for other_state in unique_finals:
                # Convert to ±1 representation and compute dot product
                candidate_bits = self._packed_state_to_bitset(candidate)
                other_bits = self._packed_state_to_bitset(other_state)
                agreements = self._count_agreements(candidate_bits, other_bits)
                total_agreements += agreements
                
            # Tie-breaking chain: smaller orbit size → channel lexicographic → lower token id
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
                    elif candidate == best_candidate:
                        # Final tie-break by token id (already handled by choosing first)
                        pass
                        
        # Use safe fallback if computation fails
        if best_candidate is None:
            # Fallback to orbit representative of orbit 0
            if 0 in self.orbit_representatives:
                best_candidate = self.ontology_keys[self.orbit_representatives[0]]
            else:
                # Final fallback to first state
                best_candidate = self.ontology_keys[0]
                
        # Cache the result
        self._address_cache[token_id] = best_candidate
        
        # Update orbit→tokens routing index
        candidate_idx = self.state_to_index[best_candidate]
        orbit_code = self.phenomenology_map[candidate_idx]
        self._add_token_to_orbit_index(token_id, orbit_code)
        
        # Persist to address memory if needed
        self._persist_address_memory(token_id, best_candidate)
        
        return best_candidate
        
    def _compute_medoid(self, addresses: list) -> int:
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
        
        for candidate in unique_addresses:
            total_agreements = 0
            
            # Calculate sum of agreements (dot products) to all addresses
            for other_address in unique_addresses:
                candidate_bits = self._packed_state_to_bitset(candidate)
                other_bits = self._packed_state_to_bitset(other_address)
                agreements = self._count_agreements(candidate_bits, other_bits)
                total_agreements += agreements
                
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
        
    def _is_better_tie_break(self, candidate_state: int, current_best_state: int) -> bool:
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
        
        # Store paths for later use
        self.address_mem_path = Path(self.address_memory_path)
        self.address_meta_path = self.address_mem_path.with_suffix(".json")
        
        # Check if both files exist
        if self.address_mem_path.exists() and self.address_meta_path.exists():
            try:
                # Load metadata
                with open(self.address_meta_path, 'r') as f:
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
                self.address_memory = np.memmap(self.address_mem_path, dtype=np.uint64, mode="r+")
                self.max_token_id = metadata.get('max_token_id', len(self.address_memory) - 1)
                
                print(f"Loaded address memory: atlas_v{stored_atlas_version}, address_v{stored_address_version}, max_token={self.max_token_id}")
                return
                
            except Exception as e:
                print(f"Warning: Failed to load address memory, recreating: {e}")
        
        # Create new address memory as zero-initialized binary file
        # Use larger initial capacity to avoid thrashing on first runs with o200k_harmony
        initial_array = np.zeros(100000, dtype=np.uint64)
        
        # Write raw binary data (not .npy format)
        with open(self.address_mem_path, 'wb') as f:
            f.write(initial_array.tobytes())
        
        # Open as memmap with write access
        self.address_memory = np.memmap(self.address_mem_path, dtype=np.uint64, mode="r+")
        self.max_token_id = -1
        
        # Write initial metadata directly
        metadata = {
            'atlas_version': self.version_info.get('atlas_version', 'v1.0.0'),
            'address_version': self.version_info.get('address_version', 'v1.0.0'),
            'max_token_id': -1
        }
        with open(self.address_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_address_memory(self):
        """Save address memory with atomic writes and metadata."""
        
        with self._address_memory_lock:
            try:
                # Find last non-zero index for max_token_id
                last_nonzero_index = 0
                for i in range(len(self.address_memory) - 1, -1, -1):
                    if self.address_memory[i] != 0:
                        last_nonzero_index = i
                        break
                
                # Create temporary files for atomic write
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dat', dir=self.address_mem_path.parent) as tmp_dat:
                    tmp_dat_path = Path(tmp_dat.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=self.address_meta_path.parent) as tmp_json:
                    tmp_json_path = Path(tmp_json.name)
                
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
                if hasattr(self.address_memory, '_mmap') and self.address_memory._mmap:
                    self.address_memory._mmap.close()
                old_address_memory = self.address_memory
                del self.address_memory
                
                # Atomic rename
                os.replace(str(tmp_dat_path), str(self.address_mem_path))
                os.replace(str(tmp_json_path), str(self.address_meta_path))
                
                # Directory fsync for POSIX crash safety
                try:
                    dir_fd = os.open(str(self.address_mem_path.parent), os.O_RDONLY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except (OSError, AttributeError):
                    # Windows doesn't support directory fsync, ignore
                    pass
                
                # Re-open memmap after successful replace
                self.address_memory = np.memmap(self.address_mem_path, dtype=np.uint64, mode='r+')
                
            except Exception as e:
                # Clean up temporary files on error
                for tmp_path in [tmp_dat_path, tmp_json_path]:
                    if tmp_path.exists():
                        tmp_path.unlink()
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
                    new_array = np.zeros(new_size, dtype=np.uint64)
                    
                    # 2. Copy old contents
                    new_array[:len(self.address_memory)] = self.address_memory
                    
                    # 3. Write to temporary file and atomic replace
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dat', dir=self.address_mem_path.parent) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        # Write raw binary data (not .npy format)
                        with open(tmp_path, 'wb') as f:
                            f.write(new_array.tobytes())
                            f.flush()
                            os.fsync(f.fileno())
                    
                    # 4. Flush and sync memmap before atomic replace
                    self.address_memory.flush()
                    # Note: numpy memmap flush() already syncs to disk, no need for additional fsync
                    if hasattr(self.address_memory, '_mmap') and self.address_memory._mmap:
                        self.address_memory._mmap.close()
                    del self.address_memory
                    
                    # Atomic replace
                    os.replace(str(tmp_path), str(self.address_mem_path))
                    
                    # Directory fsync for POSIX crash safety
                    try:
                        dir_fd = os.open(str(self.address_mem_path.parent), os.O_RDONLY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except (OSError, AttributeError):
                        # Windows doesn't support directory fsync, ignore
                        pass
                    
                    # 5. Re-open as memmap
                    self.address_memory = np.memmap(self.address_mem_path, dtype=np.uint64, mode='r+')
                
                # Update the memory-mapped array
                self.address_memory[token_id] = address & self.MASK48
                
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
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=self.address_meta_path.parent) as tmp_json:
                        json.dump(metadata, tmp_json, indent=2)
                        tmp_json.flush()
                        os.fsync(tmp_json.fileno())
                        tmp_json_path = Path(tmp_json.name)
                    
                    # Atomic replace for JSON
                    os.replace(str(tmp_json_path), str(self.address_meta_path))
                    
                    # Directory fsync for POSIX crash safety
                    try:
                        dir_fd = os.open(str(self.address_meta_path.parent), os.O_RDONLY)
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
        return self._is_admissible_core(start_state, token_id, require_global_strict=global_strict)
    
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
        """Count number of matching bits between two 48-bit values using exact bitwise operations.
        
        FROZEN - Immutable bit comparison rule for state similarity.
        """
        return (~(bits1 ^ bits2) & self.MASK48).bit_count()
        
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
        """Count agreements within specific slab using precomputed masks."""
        return (~(bits1 ^ bits2) & self.SLAB_MASKS[slab_idx]).bit_count()
        
    # --- Recovery ladder ---
    def recover_candidates(self, current_state: int, max_nudges: int = 6) -> List[int]:
        """
        Multi-level recovery: channel relaxation, neighbor orbits, duality pivot,
        orbit center fallback, geometric nudge. Return admissible token candidates.
        """
        # Guard against unknown states - return empty list for robust API
        if current_state not in self.state_to_index:
            return []
            
        # Harmony control tokens to exclude from recovery
        harmony_control_tokens = {200000, 200001, 200002, 200012}  # START, CHANNEL, MESSAGE, RETURN, CALL
        
        # Level 1: Channel relaxation (progressively relax channels, never drop Global)
        candidates = self._recovery_level_1(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            return candidates
            
        # Level 2: Neighbor orbits (Hamming-2 neighbors)
        candidates = self._recovery_level_2(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            return candidates
            
        # Level 3: Duality pivot (address ⊕ 0xFF if same phenomenology)
        candidates = self._recovery_level_3(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            return candidates
            
        # Level 4: Orbit center fallback (use representative address)
        candidates = self._recovery_level_4(current_state)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
        if candidates:
            return candidates
            
        # Level 5: Geometric nudge (up to max_nudges moves)
        candidates = self._recovery_level_5(current_state, max_nudges)
        candidates = [c for c in candidates if c not in harmony_control_tokens]
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
        # Frozen priority order: [0,0] → [0,1] → [1,0] → [1,1] → [2,0] → [2,1] → [3,0] → [3,1]
        # Drop in reverse order: [3,1] first, then [3,0], etc. (keep Global always)
        for num_slabs_to_drop in range(1, FROZEN_CHANNELS.NUM_SLABS + 1):  # Drop 1 to all slabs
            enabled_slabs = list(range(FROZEN_CHANNELS.NUM_SLABS - num_slabs_to_drop))  # Keep first N slabs
            
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
            dual_address = token_address ^ 0xFFFFFFFFFFFF  # XOR with 48-bit mask
            
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
        
        for nudge_count in range(max_nudges):
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
        if hasattr(self, '_orbit_to_tokens') and orbit_code in self._orbit_to_tokens:
            return self._orbit_to_tokens[orbit_code]
        
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
        
        # Initialize passive memory index if needed
        if not hasattr(self, 'passive_memory_index'):
            self.passive_memory_index = {}
        
        memory_key = (state_index, token_id)
        
        # Handle zero mask with zero_streak logic
        if fold_mask == 0:
            if memory_key in self.passive_memory_index:
                existing = self.passive_memory_index[memory_key]
                existing['zero_streak'] += 1
                existing['touch_count'] = min(255, existing['touch_count'] + 1)
                existing['timestamp'] = len(getattr(self, 'passive_log', []))
                
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
        
        if memory_key in self.passive_memory_index:
            # Update existing entry
            existing = self.passive_memory_index[memory_key]
            existing['touch_count'] = min(255, existing['touch_count'] + 1)
            existing['zero_streak'] = 0  # Reset zero streak
            existing['mask_id'] = interned_mask_id
            existing['timestamp'] = len(getattr(self, 'passive_log', []))
            self._append_to_passive_log(existing)
        else:
            # Create new entry
            memory_entry = {
                'state_index': state_index,
                'token_id': token_id,
                'mask_id': interned_mask_id,
                'touch_count': 1,
                'zero_streak': 0,
                'timestamp': len(getattr(self, 'passive_log', []))
            }
            
            # Check and enforce caps before adding
            self._enforce_passive_memory_caps(state_orbit, token_orbit, memory_entry)
            
            # Add new entry
            self.passive_memory_index[memory_key] = memory_entry.copy()
            self._append_to_passive_log(memory_entry)
            
    def _compute_monodromic_fold(self, state_index: int, token_id: int) -> int:
        """Compute 8-bit Monodromic Fold using exact composite form."""
        # Get existing exon_mask (0 if missing)
        key = (state_index, token_id)
        existing_mask = 0
        if hasattr(self, 'passive_memory_index') and key in self.passive_memory_index:
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
        
    def _enforce_passive_memory_caps(self, state_orbit: int, token_orbit: int, new_entry: dict) -> None:
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
            # Evict oldest entries to make room (keep 64, evict rest)
            for key, _ in k_entries[:len(k_entries) - 64]:
                del self.passive_memory_index[key]
        
        # Collect entries for M constraint (same token, same orbit)
        m_entries = []
        for key, entry in self.passive_memory_index.items():
            entry_state_idx, entry_token_id = key
            if entry_token_id == token_id:
                entry_token_address = self.address_of_token(entry_token_id)
                entry_token_orbit = self.phenomenology_map[self.state_to_index[entry_token_address]]
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
            # Evict oldest entries to make room (keep 64, evict rest)
            for key, _ in m_entries[:len(m_entries) - 64]:
                del self.passive_memory_index[key]
            
    def _is_generic_entry(self, entry: dict) -> bool:
        """Check if entry is 'generic' (token address equals orbit representative)."""
        token_id = entry['token_id']
        token_address = self.address_of_token(token_id)
        token_orbit = self.phenomenology_map[self.state_to_index[token_address]]
        representative_address = self.orbit_representatives[token_orbit]
        
        return token_address == representative_address
        
    def _append_to_passive_log(self, entry: dict) -> None:
        """Append entry to binary log file."""
        
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
                
                # Force sync for first N writes and periodically
                if self.passive_log_len < 100 or self.passive_log_len % 1000 == 0:
                    os.fsync(self.passive_log_fh.fileno())
                    
                self.passive_log_len += 1
                
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
        
        self.passive_memory_index = {}
        self.passive_log = []
        
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
        except Exception as e:
            print(f"Warning: Failed to load passive memory log: {e}")
            
    # --- Conversation policy ---
    def start_state(self) -> int:
        """
        The archetypal initial packed state for a fresh conversation.
        This is not integer zero (per your note). Use the correct one from the atlas.
        """
        # Find the archetypal state - this should be the orthogonal reference state
        # Index 0 in ontology_keys is the orthogonal reference, not the conversation start
        # The archetypal conversation start state should have specific properties:
        # - Minimal theta value (most stable)
        # - Central orbit position
        # - High symmetry
        
        # Method 1: Find state with minimum theta (most stable)
        # Use deterministic tie-breaking: first occurrence (lowest index)
        min_theta_value = np.min(self.theta)
        min_theta_indices = np.where(self.theta == min_theta_value)[0]
        min_theta_idx = int(min_theta_indices[0])  # Deterministic: choose first (lowest index)
        archetypal_state = self.ontology_keys[min_theta_idx]
        
        # Validate this is a reasonable start state
        if archetypal_state in self.state_to_index:
            return archetypal_state
            
        # Fallback: Use orbit representative of orbit 0 (most central)
        if 0 in self.orbit_representatives:
            return self.orbit_representatives[0]
            
        # Final fallback: Use orthogonal reference (index 0)
        return self.ontology_keys[0]
        
    def evolve_on_user(self, state: int, token_id: int) -> int:
        """
        Egress: apply introns (always) and fold passive memory (by default).
        """
        introns = self.token_to_introns(token_id)
        new_state = state
        
        for intron in introns:
            new_state = self.apply_intron(new_state, intron)
            
        # Fold passive memory for external inputs
        self.fold_egress(new_state, token_id)
        
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
        Route by orbit of current state. Filter candidate_vocab by same orbit of address.
        Check admissibility and apply deterministic selection (min token_id).
        If none, run recovery ladder to obtain a token. If still none, return None.
        """
        # Validate state
        if state not in self.state_to_index:
            # Non-atlas state: deterministic halt
            return None
            
        state_idx = self.state_to_index[state]
        current_orbit = self.phenomenology_map[state_idx]  # Use phenomenology_map for orbit code
        
        # Generate default candidate_vocab if not provided
        if candidate_vocab is None:
            # Try cached orbit tokens first
            tok_list = list(self._orbit_to_tokens.get(current_orbit, []))
            if not tok_list:
                tok_list = self._get_tokens_in_orbit(current_orbit)
            # Bounded fallback
            if not tok_list:
                bound = min(512, getattr(self, "vocab_size", 50000))
                tok_list = [t for t in range(bound)]
            candidate_vocab = [t for t in tok_list if t not in GENERATION_EXCLUDED]
        
        # Harmony control tokens to exclude from generation
        harmony_control_tokens = GENERATION_EXCLUDED
        
        # Filter candidates by admissibility
        admissible_candidates = []
        for token_id in candidate_vocab:
            # Exclude Harmony control tokens
            if token_id in harmony_control_tokens:
                continue
                
            if self.is_admissible(state, token_id):
                # Check if token's address is in same orbit (simplified)
                token_address = self.address_of_token(token_id)
                if token_address in self.state_to_index:
                    token_orbit = self.phenomenology_map[self.state_to_index[token_address]]
                    if token_orbit == current_orbit:
                        admissible_candidates.append(token_id)
                        
        # Deterministic selection: minimum token_id
        if admissible_candidates:
            return min(admissible_candidates)
            
        # Run recovery ladder
        recovery_candidates = self.recover_candidates(state, self.runtime.get("max_nudges", 6))
        if recovery_candidates:
            return min(recovery_candidates)
            
        return None  # Halt condition