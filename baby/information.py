"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.
"""

import numpy as np
import json
import time
import os
import threading
import tempfile
import pickle
import gzip
from typing import Dict, Optional, Any, Tuple, Set
from collections import deque

from . import governance
from .types import PhenotypeStore, ManifoldData, ValidationReport


class InformationEngine:
    """
    S2: Measurement & Resource Coordination. 
    
    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.
    """
    
    def __init__(self, manifold_data: ManifoldData):
        """
        Initialize with pre-discovered manifold data.
        
        Args:
            manifold_data: Dictionary containing the complete physical manifold
        """
        self.genotype_map = manifold_data['genotype_map']
        
        # Ensure integer keys for performance
        if isinstance(next(iter(self.genotype_map.keys())), str):
            self.genotype_map = {int(k): v for k, v in self.genotype_map.items()}
        
        self.endogenous_modulus = manifold_data['endogenous_modulus']
        self.manifold_diameter = manifold_data['manifold_diameter']
        
        # Build inverse map for index to state lookups
        self.inverse_genotype_map = {v: k for k, v in self.genotype_map.items()}
        
        # Validate expected constants
        if self.endogenous_modulus != 788_986:
            raise ValueError(f"Expected endogenous modulus 788,986, got {self.endogenous_modulus}")
        if self.manifold_diameter != 6:
            raise ValueError(f"Expected manifold diameter 6, got {self.manifold_diameter}")

    def get_index_from_state(self, state_int: int) -> int:
        """
        Looks up the canonical index for a physical state integer.
        
        Args:
            state_int: 48-bit integer representing physical state
            
        Returns:
            Index in the discovered manifold (0 to 788,985)
            
        Raises:
            ValueError: If state not found in manifold
        """
        index = self.genotype_map.get(state_int, -1)
        if index == -1:
            raise ValueError(
                f"CRITICAL: State integer {state_int} not found in discovered manifold. "
                f"This indicates a fundamental physics violation."
            )
        return index

    def get_state_from_index(self, index: int) -> int:
        """
        Get state integer from canonical index.
        
        Args:
            index: Canonical index (0 to 788,985)
            
        Returns:
            48-bit state integer
        """
        state_int = self.inverse_genotype_map.get(index)
        if state_int is None:
            raise ValueError(f"Invalid index {index}, must be 0 to {self.endogenous_modulus - 1}")
        return state_int

    @staticmethod
    def int_to_tensor(state_int: int) -> np.ndarray:
        """
        Converts a canonical 48-bit integer state to geometric tensor.
        
        The mapping is: bit 0 (LSB) = element 47, bit 47 (MSB) = element 0
        Bit values: 0 = +1, 1 = -1
        
        Args:
            state_int: 48-bit integer state
            
        Returns:
            Tensor with shape [4, 2, 3, 2] and values ±1
        """
        # Convert to 6 bytes (48 bits), big-endian
        state_packed_bytes = state_int.to_bytes(6, 'big')
        
        # Unpack to individual bits
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8))
        
        # Convert: 0 -> +1, 1 -> -1
        tensor_flat = (1 - 2 * bits).astype(np.int8)
        
        # Reshape to proper geometry
        return tensor_flat.reshape(4, 2, 3, 2)

    @staticmethod  
    def tensor_to_int(tensor: np.ndarray) -> int:
        """
        Converts a geometric tensor to its canonical 48-bit integer state.
        
        Args:
            tensor: NumPy array with shape [4, 2, 3, 2] and values ±1
            
        Returns:
            48-bit integer representation
        """
        # Flatten in C-order and convert: +1 -> 0, -1 -> 1
        bits = (tensor.flatten(order='C') == -1).astype(np.uint8)
        
        # Pack bits into bytes
        packed = np.packbits(bits)
        
        # Convert to integer, big-endian
        return int.from_bytes(packed.tobytes(), 'big')

    def gyrodistance_angular(self, T1: np.ndarray, T2: np.ndarray) -> float:
        """
        Calculate angular divergence between tensors in radians.
        
        This measures the geometric alignment between two states in 
        48-dimensional space using cosine similarity.
        
        Args:
            T1: First tensor [4, 2, 3, 2]
            T2: Second tensor [4, 2, 3, 2]
            
        Returns:
            Angular distance in radians (0 to π)
        """
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()
        
        # Cosine similarity in 48-dimensional space
        cosine_similarity = np.dot(T1_flat, T2_flat) / 48.0
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        return np.arccos(cosine_similarity)

    def measure_state_divergence(self, state_int: int) -> float:
        """
        Measure angular divergence from the archetypal state (GENE_Mac_S).
        
        Args:
            state_int: Current physical state
            
        Returns:
            Angular divergence in radians
        """
        current_tensor = self.int_to_tensor(state_int)
        return self.gyrodistance_angular(current_tensor, governance.GENE_Mac_S)


class PickleStore:
    """
    File-based phenotype storage using compressed pickle format.
    
    Provides thread-safe, atomic updates with automatic backup on corruption.
    """
    
    def __init__(self, store_path: str):
        """
        Initialize pickle-based storage.
        
        Args:
            store_path: File path for persistent storage
        """
        self.store_path = store_path
        self.data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(store_path) or '.', exist_ok=True)
        
        self._load()

    def get(self, context_key: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Retrieve phenotype entry for context key."""
        with self.lock:
            entry = self.data.get(context_key)
            return entry.copy() if entry else None

    def put(self, context_key: Tuple[int, int], entry: Dict[str, Any]) -> None:
        """Store phenotype entry for context key."""
        with self.lock:
            self.data[context_key] = entry.copy()
            self._save()

    def close(self) -> None:
        """Clean shutdown with final save."""
        with self.lock:
            self._save()

    def _load(self) -> None:
        """Load data from disk, handling corruption gracefully."""
        if not os.path.exists(self.store_path):
            return
            
        try:
            with gzip.open(self.store_path, 'rb') as f:
                self.data = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError) as e:
            # Handle corruption by starting fresh, but preserve backup
            backup_path = f"{self.store_path}.corrupt.{int(time.time())}"
            if os.path.exists(self.store_path):
                os.rename(self.store_path, backup_path)
                print(f"Warning: Corrupted store moved to {backup_path}")
            self.data = {}

    def _save(self) -> None:
        """Atomically save data to disk."""
        if not self.data:
            return
            
        # Create temporary file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.store_path) or '.',
            prefix=".gyro_temp_"
        )
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                with gzip.open(temp_file, 'wb') as gzip_file:
                    pickle.dump(self.data, gzip_file, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            if os.name == 'nt':  # Windows
                # Windows doesn't allow atomic rename if target exists
                if os.path.exists(self.store_path):
                    os.unlink(self.store_path)
            os.rename(temp_path, self.store_path)
            
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


class MultiAgentPhenotypeStore:
    """
    Knowledge store with public base + private agent overlay.
    
    Implements read-through cache pattern: private knowledge takes precedence,
    falls back to shared public knowledge base.
    """
    
    def __init__(self, public_store_path: str, private_store_path: str):
        """
        Initialize multi-agent store with public/private separation.
        
        Args:
            public_store_path: Path to shared knowledge base (read-only)
            private_store_path: Path to agent-specific knowledge
        """
        # Read-only public knowledge base
        self.public_store = PickleStore(public_store_path)
        self.public_store._save = lambda: None  # Make read-only
        
        # Private agent deltas (in-memory for fast access)
        self.private_deltas: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.private_store_path = private_store_path
        self.lock = threading.RLock()
        
        self._load_private_deltas()

    def get(self, context_key: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Read-through cache: check private first, then public."""
        with self.lock:
            # Private knowledge takes precedence
            if context_key in self.private_deltas:
                return self.private_deltas[context_key].copy()
            
            # Fall back to public knowledge (already returns copy)
            return self.public_store.get(context_key)

    def put(self, context_key: Tuple[int, int], entry: Dict[str, Any]) -> None:
        """All writes go to private deltas only."""
        with self.lock:
            self.private_deltas[context_key] = entry.copy()
            self._save_private_deltas()

    def close(self) -> None:
        """Clean shutdown."""
        with self.lock:
            self._save_private_deltas()
        self.public_store.close()

    def reload_public_knowledge(self) -> None:
        """Refresh public knowledge base (for updates)."""
        with self.lock:
            # Fix: Acquire public store's lock to avoid race condition
            with self.public_store.lock:
                self.public_store._load()

    def _load_private_deltas(self) -> None:
        """Load agent's private knowledge."""
        if os.path.exists(self.private_store_path):
            try:
                with gzip.open(self.private_store_path, 'rb') as f:
                    self.private_deltas = pickle.load(f)
            except (OSError, pickle.PickleError):
                self.private_deltas = {}

    def _save_private_deltas(self) -> None:
        """Save agent's private knowledge atomically."""
        if not self.private_deltas:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.private_store_path), exist_ok=True)
        
        # Atomic save using same pattern as PickleStore
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.private_store_path),
            prefix=".gyro_private_temp_"
        )
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                with gzip.open(temp_file, 'wb') as gzip_file:
                    pickle.dump(self.private_deltas, gzip_file, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            if os.name == 'nt':  # Windows
                if os.path.exists(self.private_store_path):
                    os.unlink(self.private_store_path)
            os.rename(temp_path, self.private_store_path)
            
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


class CanonicalizingStore:
    """
    Decorator that canonicalizes tensor indices before storage operations.
    
    Ensures all physically equivalent states share the same storage entry
    by mapping to canonical orbit representatives.
    """
    
    def __init__(self, base_store: PhenotypeStore, canonical_map_path: str):
        """
        Initialize canonicalizing decorator.
        
        Args:
            base_store: Underlying storage implementation
            canonical_map_path: Path to canonical mapping data
        """
        self.base_store = base_store
        
        with open(canonical_map_path, 'r') as f:
            loaded = json.load(f)
            
            # Support both dict and list formats
            if isinstance(loaded, list):
                self.canonical_map = dict(enumerate(loaded))
            else:
                self.canonical_map = {int(k): v for k, v in loaded.items()}

    def _get_canonical_key(self, context_key: Tuple[int, int]) -> Tuple[int, int]:
        """Map context key to its canonical representative."""
        tensor_index, intron = context_key
        canonical_index = self.canonical_map.get(tensor_index, tensor_index)
        return (canonical_index, intron)

    def get(self, context_key: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Retrieve using canonical key."""
        canonical_key = self._get_canonical_key(context_key)
        return self.base_store.get(canonical_key)

    def put(self, context_key: Tuple[int, int], entry: Dict[str, Any]) -> None:
        """Store using canonical key."""
        canonical_key = self._get_canonical_key(context_key)
        
        # Preserve original context for traceability
        if 'context_signature' not in entry:
            entry['context_signature'] = context_key
            
        self.base_store.put(canonical_key, entry)

    def close(self) -> None:
        """Delegate close to base store."""
        self.base_store.close()


def discover_and_save_manifold(output_path: str) -> None:
    """
    S2 responsibility: Discovers the complete physical manifold.
    
    Explores the full state space starting from GENE_Mac_S and discovers
    all reachable states. Validates the expected 788,986 states at diameter 6.
    
    Args:
        output_path: Path to save manifold data
        
    Raises:
        RuntimeError: If discovered manifold doesn't match expected constants
    """
    print("Discovering physical manifold...")
    start_time = time.time()
    
    # Start from the archetypal state
    origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    discovered_states = {origin_int}
    queue = [origin_int]
    depth = 0
    
    # Breadth-first exploration
    while queue:
        next_queue = []
        
        for current_int in queue:
            # Try all possible intron transformations
            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(current_int, intron)
                
                if next_int not in discovered_states:
                    discovered_states.add(next_int)
                    next_queue.append(next_int)
        
        if not next_queue:
            break
            
        queue = next_queue
        depth += 1
        
        if depth % 1 == 0:
            print(f"Depth {depth}: {len(discovered_states):,} states discovered")
    
    # Validate against expected constants
    if len(discovered_states) != 788_986:
        raise RuntimeError(
            f"CRITICAL: Expected 788,986 states, found {len(discovered_states):,}. "
            f"This indicates a fundamental error in the physics implementation."
        )
    
    if depth != 6:
        raise RuntimeError(
            f"CRITICAL: Expected manifold diameter 6, found {depth}. "
            f"This violates the theoretical predictions."
        )
    
    # Create canonical mapping
    sorted_state_ints = sorted(discovered_states)
    genotype_map = {state_int: i for i, state_int in enumerate(sorted_state_ints)}
    
    # Package manifold data
    manifold_data: ManifoldData = {
        "schema_version": "1.0.0",
        "genotype_map": genotype_map,
        "endogenous_modulus": len(genotype_map),
        "manifold_diameter": depth,
        "total_states": len(discovered_states),
        "build_timestamp": time.time()
    }
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifold_data, f)
    
    elapsed = time.time() - start_time
    print(f"Manifold discovery complete in {elapsed:.2f}s")
    print(f"Discovered {len(discovered_states):,} states at diameter {depth}")
    print(f"Saved to: {output_path}")


def build_canonical_map(genotype_map_path: str, output_path: str) -> None:
    """
    Build canonical mapping for orbit representatives.
    
    For each state in the manifold, computes its canonical representative
    (lexicographically smallest state in its orbit). This enables grouping
    of physically equivalent states and improves cache coherency.
    
    Args:
        genotype_map_path: Path to genotype map JSON
        output_path: Path to save canonical map
    """
    print("Building canonical map...")
    start_time = time.time()
    
    # Load genotype map
    with open(genotype_map_path, 'r') as f:
        genotype_data = json.load(f)
    
    genotype_map = genotype_data['genotype_map']
    # Ensure integer keys
    genotype_map = {int(k): v for k, v in genotype_map.items()}
    
    # Build inverse map
    inverse_genotype_map = {v: k for k, v in genotype_map.items()}
    
    # Find canonical representative for each state
    canonical_index_map = {}
    processed_states: Set[int] = set()
    
    print(f"Processing {len(genotype_map)} states...")
    
    for i, state_int in enumerate(inverse_genotype_map.values()):
        if i % 10000 == 0:
            print(f"Progress: {i}/{len(genotype_map)} states processed")
        
        if state_int in processed_states:
            continue
        
        # Find all states in the orbit
        orbit = {state_int}
        queue = deque([state_int])
        canonical_int = state_int
        
        while queue:
            current_int = queue.popleft()
            
            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(current_int, intron)
                
                if next_int not in orbit:
                    orbit.add(next_int)
                    queue.append(next_int)
                    
                    # Update canonical if we found a smaller one
                    if next_int < canonical_int:
                        canonical_int = next_int
        
        # Map all states in orbit to canonical representative
        canonical_index = genotype_map[canonical_int]
        for orbit_state in orbit:
            if orbit_state in genotype_map:
                orbit_index = genotype_map[orbit_state]
                canonical_index_map[orbit_index] = canonical_index
                processed_states.add(orbit_state)
    
    # Save canonical map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(canonical_index_map, f)
    
    elapsed = time.time() - start_time
    unique_canonicals = len(set(canonical_index_map.values()))
    
    print(f"Canonical map built in {elapsed:.2f}s")
    print(f"Found {unique_canonicals:,} unique canonical representatives")
    print(f"Compression ratio: {len(genotype_map) / unique_canonicals:.2f}x")
    print(f"Saved to: {output_path}")