"""
ext_spin_piv.py - Navigation-Driven Cryptographic Evolution

This extension implements Spin-based Particle Image Velocimetry (PIV) for
cryptographic evolution, leveraging the 720° navigation cycle.
"""

from typing import Dict, Any, List, Optional
import random
import hashlib
from collections import deque

from .base import GyroExtension


class ext_SpinPIV(GyroExtension):
    """
    Navigation-driven cryptographic evolution leveraging 720° cycle.
    FOOTPRINT: 3 bytes (16-bit PIV + counter)
    MAPPING: Encrypts navigation patterns for secure transmission
    """
    
    def __init__(self, initial_piv: Optional[int] = None):
        """
        Initialize Spin PIV system.
        
        Args:
            initial_piv: Initial PIV value (0-65535), random if None
        """
        # 16-bit PIV (2 bytes)
        self.piv = initial_piv if initial_piv is not None else random.randint(0, 65535)
        
        # Evolution counter (1 byte, wraps at 256)
        self.evolution_counter = 0
        
        # Navigation history for entropy
        self._nav_history = deque(maxlen=16)  # Last 16 navigation events
        
        # PIV evolution history for analysis
        self._piv_history = deque(maxlen=100)
        self._piv_history.append(self.piv)
        
        # Encryption statistics
        self._crypto_stats = {
            'bytes_encrypted': 0,
            'evolution_count': 0,
            'entropy_quality': 1.0
        }
    
    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """
        Process navigation event for PIV evolution.
        
        Args:
            nav_event: Packed navigation byte
            input_byte: Original input byte
        """
        # Add to navigation history
        self._nav_history.append(nav_event)
        
        # Evolve PIV every 8 navigation events (48/6 = 8 per boundary)
        if len(self._nav_history) >= 8 and len(self._nav_history) % 8 == 0:
            self.ext_evolve_piv()
    
    def ext_evolve_piv(self) -> None:
        """Evolve PIV using navigation pattern as entropy source."""
        if len(self._nav_history) < 8:
            return
        
        # Combine recent navigation into entropy
        entropy = 0
        for i, nav in enumerate(list(self._nav_history)[-8:]):
            # Shift and XOR to mix bits
            entropy ^= (nav << (2 * i)) & 0xFFFF
        
        # Apply spin transformation (720° = 2 full rotations)
        # Each evolution represents 720°/256 = 2.8125° of rotation
        rotation_bits = (self.evolution_counter * 3) % 16
        
        # Transform PIV through navigation-driven rotation
        rotated = ((self.piv << rotation_bits) | (self.piv >> (16 - rotation_bits))) & 0xFFFF
        self.piv = (rotated ^ entropy) & 0xFFFF
        
        # Update counter and history
        self.evolution_counter = (self.evolution_counter + 1) & 0xFF
        self._piv_history.append(self.piv)
        self._crypto_stats['evolution_count'] += 1
        
        # Update entropy quality metric
        self._update_entropy_quality()
    
    def ext_encrypt(self, data: int) -> int:
        """
        XOR encryption with current PIV.
        
        Args:
            data: Byte to encrypt (0-255)
            
        Returns:
            Encrypted byte
        """
        if not (0 <= data <= 255):
            raise ValueError(f"Data must be 0-255, got {data}")
        
        # Use lower 8 bits of PIV for byte encryption
        encrypted = data ^ (self.piv & 0xFF)
        
        self._crypto_stats['bytes_encrypted'] += 1
        
        return encrypted
    
    def ext_decrypt(self, data: int) -> int:
        """
        XOR decryption with current PIV.
        
        Args:
            data: Byte to decrypt (0-255)
            
        Returns:
            Decrypted byte
        """
        # XOR encryption is symmetric
        return self.ext_encrypt(data)
    
    def ext_encrypt_stream(self, data: bytes) -> bytes:
        """
        Encrypt a stream of bytes, evolving PIV periodically.
        
        Args:
            data: Bytes to encrypt
            
        Returns:
            Encrypted bytes
        """
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            # Evolve PIV every 16 bytes for forward secrecy
            if i > 0 and i % 16 == 0:
                self.ext_evolve_piv()
            
            encrypted.append(self.ext_encrypt(byte))
        
        return bytes(encrypted)
    
    def _update_entropy_quality(self) -> None:
        """Update entropy quality metric based on PIV distribution."""
        if len(self._piv_history) < 10:
            return
        
        # Check bit distribution in recent PIV values
        recent_pivs = list(self._piv_history)[-10:]
        bit_counts = [0] * 16
        
        for piv in recent_pivs:
            for bit in range(16):
                if piv & (1 << bit):
                    bit_counts[bit] += 1
        
        # Good entropy has roughly equal bit distribution
        expected = len(recent_pivs) / 2
        variance = sum((count - expected) ** 2 for count in bit_counts) / 16
        
        # Normalize to 0-1 quality score (lower variance = higher quality)
        max_variance = expected ** 2
        self._crypto_stats['entropy_quality'] = 1.0 - min(variance / max_variance, 1.0)
    
    def ext_get_piv_evolution_pattern(self) -> List[int]:
        """
        Get PIV evolution pattern for analysis.
        
        Returns:
            List of recent PIV values
        """
        return list(self._piv_history)[-20:]
    
    def ext_get_crypto_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive cryptographic analysis.
        
        Returns:
            Crypto metrics and statistics
        """
        # Calculate PIV period (if cycling)
        period = self._detect_piv_period()
        
        # Calculate bit entropy
        bit_entropy = self._calculate_bit_entropy()
        
        return {
            'current_piv': self.piv,
            'evolution_counter': self.evolution_counter,
            'statistics': self._crypto_stats.copy(),
            'piv_period': period,
            'bit_entropy': bit_entropy,
            'recent_pivs': self.ext_get_piv_evolution_pattern()
        }
    
    def _detect_piv_period(self) -> Optional[int]:
        """Detect if PIV values are cycling with a period."""
        if len(self._piv_history) < 20:
            return None
        
        recent = list(self._piv_history)[-50:]
        
        # Check for periods up to 24 (half of 48-step cycle)
        for period in range(2, 25):
            if period >= len(recent):
                continue
            
            # Check if pattern repeats with this period
            is_periodic = True
            for i in range(period, len(recent)):
                if recent[i] != recent[i - period]:
                    is_periodic = False
                    break
            
            if is_periodic:
                return period
        
        return None
    
    def _calculate_bit_entropy(self) -> float:
        """Calculate Shannon entropy of PIV bits."""
        if not self._piv_history:
            return 0.0
        
        # Count bit occurrences
        bit_counts = [0] * 16
        total_samples = len(self._piv_history)
        
        for piv in self._piv_history:
            for bit in range(16):
                if piv & (1 << bit):
                    bit_counts[bit] += 1
        
        # Calculate entropy
        import math
        entropy = 0.0
        
        for count in bit_counts:
            if count > 0:
                p = count / total_samples
                entropy -= p * math.log2(p)
                if total_samples - count > 0:
                    p_inv = (total_samples - count) / total_samples
                    entropy -= p_inv * math.log2(p_inv)
        
        # Normalize to 0-1 (max entropy for 16 bits is 16)
        return entropy / 16
    
    def ext_reseed(self, seed: Optional[int] = None) -> None:
        """
        Reseed the PIV for new cryptographic sequence.
        
        Args:
            seed: New seed value, or None for random
        """
        if seed is None:
            # Use navigation history as entropy source
            if self._nav_history:
                seed = sum(self._nav_history) & 0xFFFF
            else:
                seed = random.randint(0, 65535)
        
        self.piv = seed & 0xFFFF
        self.evolution_counter = 0
        self._piv_history.clear()
        self._piv_history.append(self.piv)
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_spin_piv"
    
    def get_extension_version(self) -> str:
        return "0.8.8"
    
    def get_footprint_bytes(self) -> int:
        # 2 bytes for PIV + 1 byte for counter = 3 bytes
        return 3
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Cryptographic evolution state for knowledge export."""
        return {
            'evolution_counter': self.evolution_counter,
            'piv_evolution_pattern': self.ext_get_piv_evolution_pattern(),
            'crypto_stats': self._crypto_stats.copy()
        }
    
    def get_session_state(self) -> Dict[str, Any]:
        """Session-specific cryptographic state."""
        return {
            'current_piv': self.piv,
            'nav_history': list(self._nav_history)
        }
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore evolution patterns."""
        if 'evolution_counter' in state:
            self.evolution_counter = state['evolution_counter']
        
        if 'piv_evolution_pattern' in state:
            self._piv_history.clear()
            for piv in state['piv_evolution_pattern']:
                self._piv_history.append(piv)
        
        if 'crypto_stats' in state:
            self._crypto_stats = state['crypto_stats']
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore current PIV state."""
        if 'current_piv' in state:
            self.piv = state['current_piv']
        
        if 'nav_history' in state:
            self._nav_history.clear()
            for nav in state['nav_history']:
                self._nav_history.append(nav)