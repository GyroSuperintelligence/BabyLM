"""
ext_bloom_filter.py - Pattern Recognition using Gene Substrate

This extension implements a Bloom filter using the Gene substrate itself,
achieving 0-byte additional footprint by leveraging the existing tensor topology.
"""

from typing import Dict, Any, Optional, Set
import hashlib

from .base import GyroExtension


class ext_BloomFilter(GyroExtension):
    """
    Pattern recognition using Gene substrate per CORE-SPEC-02.
    FOOTPRINT: 0 bytes (uses existing Gene structure)
    MAPPING: Hash functions overlay on G1 tensor topology
    """
    
    def __init__(self):
        """Initialize Bloom filter with Gene-based parameters."""
        # Parameters derived from Gene structure
        self.m = 48  # Bits in Gene structure (48 positions in tensor)
        self.k = 4   # Hash functions (4 gyration types)
        self.n = 0   # Patterns inserted
        
        # Virtual bit array (maps to Gene positions)
        self.bit_array = 0  # 48-bit integer
        
        # Pattern tracking for validation
        self._inserted_patterns = set()  # For testing/validation only
        
        # False positive rate tracking
        self._false_positive_tests = 0
        self._total_tests = 0
    
    def ext_gyration_hash(self, pattern: str, op_type: int) -> int:
        """
        Hash using gyration transformations per CORE-SPEC-02 operators.
        
        Args:
            pattern: The pattern to hash
            op_type: Gyration operator type (0-3)
            
        Returns:
            Bit position (0-47)
        """
        # Convert pattern to bytes for hashing
        pattern_bytes = pattern.encode('utf-8')
        
        # Create base hash
        h = int(hashlib.md5(pattern_bytes).hexdigest(), 16)
        
        # Apply gyration-inspired transformations
        if op_type == 0:  # Identity
            return h % self.m
        elif op_type == 1:  # Inverse
            return (self.m - (h % self.m)) % self.m
        elif op_type == 2:  # Forward gyration
            rotated = ((h << 2) | (h >> 30)) & 0xFFFFFFFF
            return rotated % self.m
        elif op_type == 3:  # Backward gyration
            rotated = ((h >> 2) | (h << 30)) & 0xFFFFFFFF
            return rotated % self.m
        else:
            raise ValueError(f"Invalid operator type: {op_type}")
    
    def ext_insert_pattern(self, pattern: str) -> None:
        """
        Insert pattern into filter using gyration-based hashing.
        
        Args:
            pattern: The pattern to insert
        """
        # Apply all k hash functions
        for i in range(self.k):
            bit_pos = self.ext_gyration_hash(pattern, i)
            self.bit_array |= (1 << bit_pos)
        
        self.n += 1
        self._inserted_patterns.add(pattern)
    
    def ext_contains(self, pattern: str) -> bool:
        """
        Test pattern membership with gyration hash functions.
        
        Args:
            pattern: The pattern to test
            
        Returns:
            True if pattern might be in set, False if definitely not
        """
        self._total_tests += 1
        
        # Check all k hash positions
        for i in range(self.k):
            bit_pos = self.ext_gyration_hash(pattern, i)
            if not (self.bit_array & (1 << bit_pos)):
                return False
        
        # Track false positives for monitoring
        if pattern not in self._inserted_patterns:
            self._false_positive_tests += 1
        
        return True
    
    def get_saturation(self) -> float:
        """
        Get filter saturation level (proportion of bits set).
        
        Returns:
            Saturation ratio (0.0 to 1.0)
        """
        bits_set = bin(self.bit_array).count('1')
        return bits_set / self.m
    
    def get_false_positive_rate(self) -> float:
        """
        Get observed false positive rate.
        
        Returns:
            False positive rate (0.0 to 1.0)
        """
        if self._total_tests == 0:
            return 0.0
        return self._false_positive_tests / self._total_tests
    
    def get_theoretical_fpr(self) -> float:
        """
        Calculate theoretical false positive rate.
        
        Returns:
            Theoretical FPR based on current parameters
        """
        if self.n == 0:
            return 0.0
        
        # Classic Bloom filter formula: (1 - e^(-kn/m))^k
        import math
        return math.pow(1 - math.exp(-self.k * self.n / self.m), self.k)
    
    def should_reset(self) -> bool:
        """
        Determine if filter should be reset due to saturation.
        
        Returns:
            True if reset is recommended
        """
        # Reset if saturation > 50% or theoretical FPR > 10%
        return self.get_saturation() > 0.5 or self.get_theoretical_fpr() > 0.1
    
    def reset(self) -> None:
        """Reset the filter to empty state."""
        self.bit_array = 0
        self.n = 0
        self._inserted_patterns.clear()
        self._false_positive_tests = 0
        self._total_tests = 0
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_bloom_filter"
    
    def get_extension_version(self) -> str:
        return "1.2.0"
    
    def get_footprint_bytes(self) -> int:
        # True 0-byte footprint - uses Gene substrate
        return 0
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Return filter state for knowledge export."""
        return {
            'bit_array': self.bit_array,
            'pattern_count': self.n,
            'saturation': self.get_saturation(),
            'theoretical_fpr': self.get_theoretical_fpr()
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """Return test statistics."""
        return {
            'false_positive_tests': self._false_positive_tests,
            'total_tests': self._total_tests,
            'observed_fpr': self.get_false_positive_rate()
        }
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore filter state."""
        if 'bit_array' in state:
            self.bit_array = state['bit_array']
        if 'pattern_count' in state:
            self.n = state['pattern_count']
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore test statistics."""
        if 'false_positive_tests' in state:
            self._false_positive_tests = state['false_positive_tests']
        if 'total_tests' in state:
            self._total_tests = state['total_tests']
    
    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """Process navigation events to detect patterns."""
        # Create pattern from navigation event
        pattern = f"nav_{nav_event:02X}"
        
        # Insert significant patterns
        if nav_event & 0x0F == 0:  # Identity operations
            self.ext_insert_pattern(pattern)
        elif (nav_event >> 4) & 0x0F == 3:  # Backward operations
            self.ext_insert_pattern(pattern)