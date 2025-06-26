"""
g1_memory.py - Genetic Memory (G1)

Provides access to the agent's static genetic memory (formerly gyro_harmonics.dat).
This is the DNA/blueprint of the agent, containing the foundational alignment rules.

The g1_memory.dat file is small, static, and read-only during runtime.
"""

import os
import hashlib
from typing import Optional, Dict, Any, Tuple
import numpy as np


class G1_Memory:
    """
    Genetic Memory: Static, global alignment rules (memory/g1_memory.dat).  # TODO: Reconsider/remove legacy memory path
    """
    HEADER_SIZE = 32  # SHA-256 checksum size

    def __init__(self):
        self.path = os.path.join("memory", "g1_memory.dat")
        self.checksum = None
        self.resonance_mask = None
        self.operator_vector = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> bool:
        """
        Load genetic memory from the agent's g1_memory.dat file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.path):
            return False
            
        try:
            with open(self.path, "rb") as f:
                # Read checksum
                self.checksum = f.read(self.HEADER_SIZE)
                
                # Read resonance mask (48×256 bits → 1536 bytes)
                mask_data = f.read(48 * 256 // 8)
                self.resonance_mask = np.frombuffer(mask_data, dtype=np.uint8).reshape(48, 256)
                
                # Read operator vector (48 bytes)
                op_data = f.read(48)
                self.operator_vector = np.frombuffer(op_data, dtype=np.uint8)
                
            return True
        except Exception as e:
            print(f"Error loading genetic memory: {e}")
            return False
    
    def save(self, gene: Dict[str, Any]) -> bool:
        """
        Save genetic memory to the agent's g1_memory.dat file.
        
        Args:
            gene: The gene object containing tensor data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Generate mask and operator vector from gene
            resonance_mask, operator_vector = self._build_from_gene(gene)
            
            # Generate checksum
            hasher = hashlib.sha256()
            for tensor_name in sorted(gene.keys()):
                tensor = gene[tensor_name]
                hasher.update(tensor.numpy().tobytes())
            checksum = hasher.digest()
            
            # Save to file
            with open(self.path, "wb") as f:
                f.write(checksum)
                f.write(resonance_mask.tobytes())
                f.write(operator_vector.tobytes())
            
            # Update internal state
            self.checksum = checksum
            self.resonance_mask = resonance_mask
            self.operator_vector = operator_vector
            
            return True
        except Exception as e:
            print(f"Error saving genetic memory: {e}")
            return False
    
    def _build_from_gene(self, gene: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build resonance mask and operator vector from gene.
        
        Args:
            gene: The gene object containing tensor data
            
        Returns:
            Tuple of (resonance_mask, operator_vector)
        """
        # Implementation depends on your tensor structure
        # This is a placeholder - you would implement the actual logic
        # based on your specific gene tensor format
        
        resonance_mask = np.zeros((48, 256), dtype=np.uint8)
        operator_vector = np.zeros(48, dtype=np.uint8)
        
        # Example logic (you would replace this with your actual implementation)
        for phase in range(48):
            for byte_val in range(256):
                # Logic to determine if this phase/byte resonates
                # This is just an example
                hi_nibble = (byte_val >> 4) & 0xF
                lo_nibble = byte_val & 0xF
                
                if (hi_nibble + lo_nibble) % 3 == 0:
                    resonance_mask[phase, byte_val] = 1
            
            # Logic to determine operator for this phase
            # This is just an example
            operator_vector[phase] = phase % 4
            
        return resonance_mask, operator_vector
    
    def get_operator(self, phase: int, input_byte: int) -> int:
        """
        Get the operator for the given phase and input byte.
        
        Args:
            phase: Current phase (0-47)
            input_byte: Input byte (0-255)
            
        Returns:
            Operator code (0-3)
        """
        if self.operator_vector is None:
            self.load()
            
        if self.operator_vector is None:
            return 0  # Default to identity
            
        return self.operator_vector[phase % 48]
    
    def validate_checksum(self, gene: Dict[str, Any]) -> bool:
        """
        Validate that the gene matches the stored checksum.
        
        Args:
            gene: The gene object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if self.checksum is None:
            self.load()
            
        if self.checksum is None:
            return False
            
        hasher = hashlib.sha256()
        for tensor_name in sorted(gene.keys()):
            tensor = gene[tensor_name]
            hasher.update(tensor.numpy().tobytes())
        
        return hasher.digest() == self.checksum