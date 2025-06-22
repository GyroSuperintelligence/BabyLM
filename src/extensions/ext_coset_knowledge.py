        
"""
ext_coset_knowledge.py - Semantic Compression through Coset Decomposition

This extension implements variable-footprint knowledge compression using
coset theory to identify and compress semantic equivalence classes.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict
import hashlib
import zlib

from .base import GyroExtension


class ext_CosetKnowledge(GyroExtension):
    """
    Variable footprint, compression tracking.
    FOOTPRINT: Variable (depends on patterns discovered)
    MAPPING: Compression and coset-based knowledge structuring
    """
    
    def __init__(self):
        """Initialize coset knowledge system."""
        # Coset representatives (patterns that represent equivalence classes)
        self._coset_reps = {}  # pattern_hash -> representative_pattern
        
        # Coset members (patterns in each equivalence class)
        self._coset_members = defaultdict(set)  # rep_hash -> set of member_hashes
        
        # Pattern storage (compressed)
        self._pattern_store = {}  # pattern_hash -> compressed_pattern
        
        # Compression statistics
        self._compression_stats = {
            'total_patterns': 0,
            'unique_cosets': 0,
            'compression_ratio': 1.0,
            'bytes_saved': 0
        }
        
        # Semantic similarity threshold
        self._similarity_threshold = 0.85
        
        # Pattern window for analysis
        self._pattern_window = []
        self._window_size = 100
    
    def ext_add_pattern(self, pattern: bytes) -> str:
        """
        Add a pattern and assign it to a coset.
        
        Args:
            pattern: Raw pattern bytes
            
        Returns:
            Pattern hash for reference
        """
        # Generate pattern hash
        pattern_hash = hashlib.sha256(pattern).hexdigest()[:16]
        
        # Find or create coset
        coset_rep = self._find_coset_representative(pattern)
        
        if coset_rep is None:
            # New coset - this pattern is the representative
            self._coset_reps[pattern_hash] = pattern
            self._coset_members[pattern_hash].add(pattern_hash)
            coset_rep = pattern_hash
            self._compression_stats['unique_cosets'] += 1
        else:
            # Add to existing coset
            self._coset_members[coset_rep].add(pattern_hash)
        
        # Store compressed pattern
        compressed = zlib.compress(pattern, level=9)
        self._pattern_store[pattern_hash] = compressed
        
        # Update statistics
        self._compression_stats['total_patterns'] += 1
        self._update_compression_ratio()
        
        # Track in window
        self._pattern_window.append((pattern_hash, coset_rep))
        if len(self._pattern_window) > self._window_size:
            self._pattern_window.pop(0)
        
        return pattern_hash
    
    def _find_coset_representative(self, pattern: bytes) -> Optional[str]:
        """
        Find the coset representative for a pattern.
        
        Returns:
            Hash of representative pattern, or None if new coset
        """
        # For each existing coset representative
        for rep_hash, rep_pattern in self._coset_reps.items():
            similarity = self._calculate_similarity(pattern, rep_pattern)
            if similarity >= self._similarity_threshold:
                return rep_hash
        
        return None
    
    def _calculate_similarity(self, pattern1: bytes, pattern2: bytes) -> float:
        """
        Calculate semantic similarity between patterns.
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple byte-level similarity for now
        # Could be enhanced with more sophisticated metrics
        
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0
        
        # Normalize lengths
        min_len = min(len(pattern1), len(pattern2))
        max_len = max(len(pattern1), len(pattern2))
        
        # Count matching bytes
        matches = sum(1 for i in range(min_len) if pattern1[i] == pattern2[i])
        
        # Similarity based on matches and length difference
        match_ratio = matches / min_len if min_len > 0 else 0
        length_ratio = min_len / max_len
        
        return (match_ratio + length_ratio) / 2
    
    def _update_compression_ratio(self) -> None:
        """Update compression statistics."""
        if self._compression_stats['total_patterns'] == 0:
            return
        
        # Calculate total original size
        original_size = sum(
            len(zlib.decompress(compressed))
            for compressed in self._pattern_store.values()
        )
        
        # Calculate compressed size (store only representatives)
        compressed_size = sum(
            len(compressed)
            for pattern_hash, compressed in self._pattern_store.items()
            if pattern_hash in self._coset_reps
        )
        
        # Update stats
        if original_size > 0:
            self._compression_stats['compression_ratio'] = original_size / compressed_size
            self._compression_stats['bytes_saved'] = original_size - compressed_size
    
    def ext_get_coset_info(self, pattern_hash: str) -> Dict[str, Any]:
        """
        Get information about a pattern's coset.
        
        Args:
            pattern_hash: Hash of the pattern
            
        Returns:
            Coset information
        """
        # Find which coset this pattern belongs to
        for rep_hash, members in self._coset_members.items():
            if pattern_hash in members:
                return {
                    'representative': rep_hash,
                    'member_count': len(members),
                    'members': list(members),
                    'is_representative': pattern_hash == rep_hash
                }
        
        return {'error': 'Pattern not found'}
    
    def ext_get_semantic_groups(self) -> List[Dict[str, Any]]:
        """
        Get all semantic groups (cosets) discovered.
        
        Returns:
            List of coset information
        """
        groups = []
        
        for rep_hash, members in self._coset_members.items():
            groups.append({
                'representative': rep_hash,
                'size': len(members),
                'compression_benefit': len(members) - 1  # Patterns saved
            })
        
        # Sort by size (largest groups first)
        groups.sort(key=lambda x: x['size'], reverse=True)
        
        return groups
    
    def ext_reconstruct_pattern(self, pattern_hash: str) -> Optional[bytes]:
        """
        Reconstruct a pattern from its compressed form.
        
        Args:
            pattern_hash: Hash of the pattern
            
        Returns:
            Original pattern bytes, or None if not found
        """
        if pattern_hash in self._pattern_store:
            compressed = self._pattern_store[pattern_hash]
            return zlib.decompress(compressed)
        
        return None
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_coset_knowledge"
    
    def get_extension_version(self) -> str:
        return "0.9.1"
    
    def get_footprint_bytes(self) -> int:
        # Variable footprint based on stored patterns
        footprint = 0
        
        # Coset representatives
        footprint += len(self._coset_reps) * 32  # Hash + reference
        
        # Coset members
        footprint += sum(len(members) * 16 for members in self._coset_members.values())
        
        # Compressed patterns
        footprint += sum(len(compressed) for compressed in self._pattern_store.values()) 
        return footprint
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Return coset structure and compressed patterns."""
        return {
            'coset_representatives': list(self._coset_reps.keys()),
            'coset_structure': {
                rep: list(members) 
                for rep, members in self._coset_members.items()
            },
            'compression_stats': self._compression_stats.copy(),
            'similarity_threshold': self._similarity_threshold
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """Return pattern window and recent activity."""
        return {
            'pattern_window': self._pattern_window[-20:],  # Last 20 patterns
            'window_size': self._window_size
        }
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore coset structure."""
        if 'coset_representatives' in state:
            # Would need to restore full patterns from storage
            pass
        
        if 'coset_structure' in state:
            self._coset_members.clear()
            for rep, members in state['coset_structure'].items():
                self._coset_members[rep] = set(members)
        
        if 'compression_stats' in state:
            self._compression_stats = state['compression_stats']
        
        if 'similarity_threshold' in state:
            self._similarity_threshold = state['similarity_threshold']
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore pattern window."""
        if 'pattern_window' in state:
            self._pattern_window = state['pattern_window']
        
        if 'window_size' in state:
            self._window_size = state['window_size']
    
    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """Process navigation events to build semantic patterns."""
        # Build patterns from navigation sequences
        if len(self._pattern_window) >= 8:
            # Create pattern from last 8 navigation events
            pattern_bytes = bytes([
                event[0] for event in self._pattern_window[-8:]
                if isinstance(event, tuple) and len(event) > 0
            ])
            
            if len(pattern_bytes) == 8:
                self.ext_add_pattern(pattern_bytes)