"""FROZEN channel definitions - Single source of truth for all channel mappings.

This module defines the immutable channel structure used throughout GyroSI:
- Global channel (all 48 bits)
- 8 Layer×Frame slabs (6 bits each)
- Priority ordering for recovery ladder
- Bit index mapping formulas

FROZEN - These definitions are immutable and used across the entire system.
"""

from typing import List, Tuple, Dict

# === FROZEN CONSTANTS ===

# Total number of bits in the state representation
TOTAL_BITS = 48

# Number of slabs in the Layer×Frame structure
NUM_SLABS = 8

# Bits per slab (3 rows × 2 columns)
BITS_PER_SLAB = 6

# Tensor dimensions
NUM_LAYERS = 4
NUM_FRAMES = 2
NUM_ROWS = 3
NUM_COLS = 2

# === CHANNEL DEFINITIONS ===

class FROZEN_CHANNELS:
    """Immutable channel definitions for GyroSI physics."""
    
    # Constants from module level
    TOTAL_BITS = TOTAL_BITS
    NUM_SLABS = NUM_SLABS
    BITS_PER_SLAB = BITS_PER_SLAB
    NUM_LAYERS = NUM_LAYERS
    NUM_FRAMES = NUM_FRAMES
    NUM_ROWS = NUM_ROWS
    NUM_COLS = NUM_COLS
    
    # 48-bit mask for state representation
    MASK48 = 0xFFFFFFFFFFFF
    
    # Global channel: all 48 bit positions
    GLOBAL = list(range(TOTAL_BITS))
    
    # Layer×Frame slab definitions
    # Each slab contains 6 bits: 3 rows × 2 columns
    SLABS = {
        # Slab index: (layer, frame)
        0: (0, 0),  # Layer 0, Frame 0
        1: (0, 1),  # Layer 0, Frame 1
        2: (1, 0),  # Layer 1, Frame 0
        3: (1, 1),  # Layer 1, Frame 1
        4: (2, 0),  # Layer 2, Frame 0
        5: (2, 1),  # Layer 2, Frame 1
        6: (3, 0),  # Layer 3, Frame 0
        7: (3, 1),  # Layer 3, Frame 1
    }
    
    # Priority order for recovery ladder (Global always enabled)
    # Higher index = lower priority (dropped first in recovery)
    SLAB_PRIORITY_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
    
    @staticmethod
    def get_bit_index(layer: int, frame: int, row: int, col: int) -> int:
        """Convert tensor coordinates to bit index.
        
        Formula: bit_index = (layer * 12) + (frame * 6) + (row * 2) + col
        
        Args:
            layer: Layer index (0-3)
            frame: Frame index (0-1) 
            row: Row index (0-2)
            col: Column index (0-1)
            
        Returns:
            Bit index (0-47)
        """
        if not (0 <= layer < NUM_LAYERS):
            raise ValueError(f"Layer {layer} out of range [0, {NUM_LAYERS-1}]")
        if not (0 <= frame < NUM_FRAMES):
            raise ValueError(f"Frame {frame} out of range [0, {NUM_FRAMES-1}]")
        if not (0 <= row < NUM_ROWS):
            raise ValueError(f"Row {row} out of range [0, {NUM_ROWS-1}]")
        if not (0 <= col < NUM_COLS):
            raise ValueError(f"Column {col} out of range [0, {NUM_COLS-1}]")
            
        return (layer * 12) + (frame * 6) + (row * 2) + col
    
    @staticmethod
    def get_tensor_coords(bit_index: int) -> Tuple[int, int, int, int]:
        """Convert bit index to tensor coordinates.
        
        Args:
            bit_index: Bit index (0-47)
            
        Returns:
            Tuple of (layer, frame, row, col)
        """
        if not (0 <= bit_index < TOTAL_BITS):
            raise ValueError(f"Bit index {bit_index} out of range [0, {TOTAL_BITS-1}]")
            
        layer = bit_index // 12
        remainder = bit_index % 12
        frame = remainder // 6
        remainder = remainder % 6
        row = remainder // 2
        col = remainder % 2
        
        return (layer, frame, row, col)
    
    @staticmethod
    def get_slab_bit_indices(slab_idx: int) -> List[int]:
        """Get bit indices for a specific slab.
        
        Args:
            slab_idx: Slab index (0-7)
            
        Returns:
            List of 6 bit indices for the slab
        """
        if slab_idx not in FROZEN_CHANNELS.SLABS:
            raise ValueError(f"Invalid slab index {slab_idx}")
            
        layer, frame = FROZEN_CHANNELS.SLABS[slab_idx]
        indices = []
        
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                bit_idx = FROZEN_CHANNELS.get_bit_index(layer, frame, row, col)
                indices.append(bit_idx)
                
        return indices
    
    @staticmethod
    def get_slab_mask(slab_idx: int) -> int:
        """Get bitmask for a specific slab.
        
        Args:
            slab_idx: Slab index (0-7)
            
        Returns:
            Bitmask with bits set for the slab positions
        """
        indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
        mask = 0
        for bit_idx in indices:
            mask |= (1 << bit_idx)
        return mask
    
    @staticmethod
    def get_all_slab_masks() -> List[int]:
        """Get bitmasks for all slabs.
        
        Returns:
            List of 8 bitmasks, one for each slab
        """
        return [FROZEN_CHANNELS.get_slab_mask(i) for i in range(NUM_SLABS)]
    
    @staticmethod
    def verify_channel_integrity() -> None:
        """Verify channel mapping integrity.
        
        Raises:
            AssertionError: If channel mapping is invalid
        """
        # Verify each slab has exactly 6 distinct indices
        all_indices = set()
        for slab_idx in range(NUM_SLABS):
            slab_indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
            assert len(slab_indices) == BITS_PER_SLAB, f"Slab {slab_idx} has {len(slab_indices)} indices, expected {BITS_PER_SLAB}"
            assert len(set(slab_indices)) == BITS_PER_SLAB, f"Slab {slab_idx} has duplicate indices: {slab_indices}"
            all_indices.update(slab_indices)
        
        # Verify union of all slab indices covers exactly 0-47
        assert len(all_indices) == TOTAL_BITS, f"Union of all slab indices is {len(all_indices)}, expected {TOTAL_BITS}"
        assert all_indices == set(range(TOTAL_BITS)), f"Slab indices do not cover exactly 0-{TOTAL_BITS-1}: {sorted(all_indices)}"
        
        # Verify global channel covers all bits
        assert len(FROZEN_CHANNELS.GLOBAL) == TOTAL_BITS
        assert set(FROZEN_CHANNELS.GLOBAL) == set(range(TOTAL_BITS))
        
        # Verify slab priority order
        assert len(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER) == NUM_SLABS
        assert set(FROZEN_CHANNELS.SLAB_PRIORITY_ORDER) == set(range(NUM_SLABS))


# === CONVENIENCE FUNCTIONS ===

def get_slab_name(slab_idx: int) -> str:
    """Get human-readable name for a slab.
    
    Args:
        slab_idx: Slab index (0-7)
        
    Returns:
        String like "Layer×Frame[0,0]"
    """
    if slab_idx not in FROZEN_CHANNELS.SLABS:
        raise ValueError(f"Invalid slab index {slab_idx}")
        
    layer, frame = FROZEN_CHANNELS.SLABS[slab_idx]
    return f"Layer×Frame[{layer},{frame}]"


def get_channel_summary() -> Dict[str, any]:
    """Get summary of all channel definitions.
    
    Returns:
        Dictionary with channel information
    """
    return {
        'total_bits': TOTAL_BITS,
        'num_slabs': NUM_SLABS,
        'bits_per_slab': BITS_PER_SLAB,
        'global_channel_size': len(FROZEN_CHANNELS.GLOBAL),
        'slab_names': [get_slab_name(i) for i in range(NUM_SLABS)],
        'priority_order': [get_slab_name(i) for i in FROZEN_CHANNELS.SLAB_PRIORITY_ORDER]
    }


# Verify integrity on import
FROZEN_CHANNELS.verify_channel_integrity()