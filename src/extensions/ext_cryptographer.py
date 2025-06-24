"""
Unified, gyration-aware cryptographic extension for GyroSI (GyroCryptography).

This module provides per-user symmetric cryptography for data at rest,
with a keystream that is traceably enhanced by the navigation-driven
gyration state. This creates a unique fusion of
user-keyed security and process-driven entropy.

CRITICAL GUARANTEES:
- Gene/Genome remain plaintext in RAM
- Encryption happens ONLY at storage boundaries
- Navigation mechanics are NEVER affected
- Operator matrix remains untouched
"""
import hashlib
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from .base import GyroExtension


class ext_Cryptographer(GyroExtension):
    """
    Implements a gyration-aware stream cipher, merging user-key security
    with navigation-driven state evolution (successor to the original ext_SpinPIV approach).

    FOOTPRINT: 5 bytes (2B counter + 2B GyroCryptography + 1B evolution_counter)
    """

    def __init__(self, user_key: bytes):
        """
        Initialize with a user-specific key and internal GyroCryptography state.

        Args:
            user_key: 16-32 byte user-specific encryption key.
        """
        if len(user_key) < 16:
            raise ValueError("User key must be at least 16 bytes")

        # Core cryptographic state
        self.user_key = user_key[:32]  # Max 32 bytes
        self.counter = 0  # 16-bit stream counter

        # Traceable 16-bit GyroCryptography – first two bytes of BLAKE2s(user_key)
        self.gyro_cryptography = int.from_bytes(
            hashlib.blake2s(user_key, digest_size=2).digest(), "big"
        )
        self.evolution_counter = 0  # 8-bit

        # Navigation history for entropy
        self._nav_history = deque(maxlen=16)
        self._gyro_cryptography_history = deque(maxlen=100)
        self._gyro_cryptography_history.append(self.gyro_cryptography)

        # Diagnostic statistics
        self._crypto_stats = {
            "bytes_encrypted": 0,
            "bytes_decrypted": 0,
            "evolution_count": 0,
            "entropy_quality": 1.0
        }

    # --- Core Cryptographic Logic ---

    def _keystream_block(self, block_idx: int, length: int) -> bytes:
        """
        Generate keystream block up to 32 bytes (BLAKE2s limit).

        Args:
            block_idx: Block index for this keystream segment
            length: Number of bytes needed (max 32)

        Returns:
            Keystream bytes
        """
        if length > 32:
            raise ValueError("Block size must be <= 32 bytes")

        ctx = hashlib.blake2s(digest_size=length, key=self.user_key)
        ctx.update(block_idx.to_bytes(2, "big"))   # Stream counter
        ctx.update(self.gyro_cryptography.to_bytes(2, "big"))    # Navigation-derived salt
        return ctx.digest()

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using the GyroCryptography-aware XOR stream cipher.
        Used ONLY at storage boundaries (never on in-memory structures).

        Args:
            data: Plaintext bytes to encrypt

        Returns:
            Encrypted bytes
        """
        out = bytearray()
        # Process in 16-byte chunks to evolve GyroCryptography at correct intervals
        for block_idx, start in enumerate(range(0, len(data), 16)):
            # Evolve GyroCryptography every 16 bytes (except at start)
            if start > 0:
                self._evolve_gyro_cryptography()
            chunk = data[start:start+16]
            ks = self._keystream_block(self.counter + block_idx, len(chunk))
            out.extend(b ^ k for b, k in zip(chunk, ks))
        # Update counter based on 16-byte blocks used
        blocks_used = math.ceil(len(data) / 16)
        self.counter = (self.counter + blocks_used) & 0xFFFF
        self._crypto_stats["bytes_encrypted"] += len(data)
        return bytes(out)

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data. The stream cipher is symmetric.

        IMPORTANT: This method preserves GyroCryptography state to match encryption.
        If you need stateless decryption, create a fresh instance.
        """
        # Save initial state
        initial_gyro_cryptography = self.gyro_cryptography
        initial_evolution_counter = self.evolution_counter
        initial_counter = self.counter

        out = bytearray()
        # Start from counter 0 and process in 16-byte chunks exactly as encryption did
        current_counter = 0
        for block_idx, start in enumerate(range(0, len(data), 16)):
            # Evolve GyroCryptography every 16 bytes (except at start)
            if start > 0:
                self._evolve_gyro_cryptography()
            chunk = data[start:start+16]
            ks = self._keystream_block(current_counter + block_idx, len(chunk))
            out.extend(b ^ k for b, k in zip(chunk, ks))

        # Restore full instance state (counter, GyroCryptography, evo_counter)
        self.gyro_cryptography = initial_gyro_cryptography
        self.evolution_counter = initial_evolution_counter
        self.counter = initial_counter
        self._crypto_stats["bytes_decrypted"] += len(data)
        return bytes(out)

    # --- GyroCryptography Evolution Logic ---

    def process_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """
        Process navigation events to evolve GyroCryptography state.
        This ties cryptographic state to navigation history.

        Args:
            nav_event: Packed navigation event byte
            input_byte: Original input byte (unused)
        """
        self._nav_history.append(nav_event)

        # Evolve GyroCryptography every 8 navigation events
        if len(self._nav_history) >= 8 and len(self._nav_history) % 8 == 0:
            self._evolve_gyro_cryptography()

    def _evolve_gyro_cryptography(self) -> None:
        """
        Evolve GyroCryptography using navigation pattern as entropy source.
        This is the heart of navigation-crypto fusion.
        """
        # Always build an 8-entry list, pad with zeros on the left
        recent_nav = list(self._nav_history)[-8:]
        recent_nav = ([0] * (8 - len(recent_nav))) + recent_nav

        # Combine recent navigation into entropy
        entropy = 0
        for i, nav in enumerate(recent_nav):
            entropy ^= (nav << (2 * i)) & 0xFFFF

        # Apply spin transformation (720° = 2 full rotations)
        # Each evolution represents 720°/256 = 2.8125° of rotation
        rotation_bits = (self.evolution_counter * 3) % 16
        rotated = ((self.gyro_cryptography << rotation_bits) | (self.gyro_cryptography >> (16 - rotation_bits))) & 0xFFFF
        self.gyro_cryptography = (rotated ^ entropy) & 0xFFFF

        # Update counters and history
        self.evolution_counter = (self.evolution_counter + 1) & 0xFF
        self._gyro_cryptography_history.append(self.gyro_cryptography)
        self._crypto_stats["evolution_count"] += 1
        self._update_entropy_quality()

    # --- Diagnostic & Analysis Tools ---

    def ext_get_crypto_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive cryptographic analysis.
        Useful for monitoring system health and entropy quality.
        """
        return {
            "current_gyro_cryptography": self.gyro_cryptography,
            "evolution_counter": self.evolution_counter,
            "statistics": self._crypto_stats.copy(),
            "gyro_cryptography_period": self._detect_gyro_cryptography_period(),
            "bit_entropy": self._calculate_bit_entropy(),
            "recent_gyro_cryptographies": list(self._gyro_cryptography_history)[-20:],
            "nav_history_size": len(self._nav_history)
        }

    def _update_entropy_quality(self) -> None:
        """Update entropy quality metric based on GyroCryptography distribution."""
        if len(self._gyro_cryptography_history) < 10:
            return

        recent_gyro_cryptographies = list(self._gyro_cryptography_history)[-10:]
        bit_counts = [0] * 16

        for gyro_cryptography in recent_gyro_cryptographies:
            for bit in range(16):
                if gyro_cryptography & (1 << bit):
                    bit_counts[bit] += 1

        expected = len(recent_gyro_cryptographies) / 2
        variance = sum((count - expected) ** 2 for count in bit_counts) / 16
        max_variance = expected ** 2
        self._crypto_stats["entropy_quality"] = 1.0 - min(variance / max_variance, 1.0)

    def _detect_gyro_cryptography_period(self) -> Optional[int]:
        """Detect if GyroCryptography values are cycling with a period."""
        if len(self._gyro_cryptography_history) < 20:
            return None

        recent = list(self._gyro_cryptography_history)[-50:]

        for period in range(2, 25):
            if period >= len(recent):
                continue

            is_periodic = True
            for i in range(period, len(recent)):
                if recent[i] != recent[i - period]:
                    is_periodic = False
                    break

            if is_periodic:
                return period

        return None

    def _calculate_bit_entropy(self) -> float:
        """Calculate Shannon entropy of GyroCryptography bits."""
        if not self._gyro_cryptography_history:
            return 0.0

        bit_counts = [0] * 16
        total_samples = len(self._gyro_cryptography_history)

        for gyro_cryptography in self._gyro_cryptography_history:
            for bit in range(16):
                if gyro_cryptography & (1 << bit):
                    bit_counts[bit] += 1

        entropy = 0.0
        for count in bit_counts:
            if count > 0 and (total_samples - count) > 0:
                p1 = count / total_samples
                p0 = (total_samples - count) / total_samples
                entropy -= (p1 * math.log2(p1) + p0 * math.log2(p0))

        return entropy / 16

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_Cryptographer"

    def get_extension_version(self) -> str:
        return "2.0.0"  # Major version bump for architectural change

    def get_footprint_bytes(self) -> int:
        return 5  # 2 (counter) + 2 (gyro_cryptography) + 1 (evolution_counter)

    def get_learning_state(self) -> Dict[str, Any]:
        """
        Returns learning state for knowledge export.
        This state will be encrypted when written to disk.
        """
        return {
            "counter": self.counter,
            "key_hash": hashlib.sha256(self.user_key).hexdigest()[:16],
            "evolution_counter": self.evolution_counter,
            "final_gyro_cryptography": self.gyro_cryptography,
            "gyro_cryptography_history_snapshot": list(self._gyro_cryptography_history)[-20:],
            "crypto_stats": self._crypto_stats.copy(),
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learning state from knowledge package."""
        self.counter = state.get("counter", 0)
        self.evolution_counter = state.get("evolution_counter", 0)
        self.gyro_cryptography = state.get("final_gyro_cryptography", random.randint(0, 65535))

        self._gyro_cryptography_history.clear()
        self._gyro_cryptography_history.extend(state.get("gyro_cryptography_history_snapshot", []))
        if not self._gyro_cryptography_history:
            self._gyro_cryptography_history.append(self.gyro_cryptography)

        self._crypto_stats = state.get("crypto_stats", self._crypto_stats.copy())

    def get_session_state(self) -> Dict[str, Any]:
        """Returns session-specific state (non-exportable)."""
        return {
            "current_gyro_cryptography": self.gyro_cryptography,
            "nav_history": list(self._nav_history)
        }

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore session state."""
        self.gyro_cryptography = state.get("current_gyro_cryptography", self.gyro_cryptography)
        self._nav_history.clear()
        self._nav_history.extend(state.get("nav_history", []))

    def get_pattern_filename(self) -> str:
        return "ext_Cryptographer@2.0.0.keystate"

    # --- Utility Methods for Testing ---

    def reset_counter(self) -> None:
        """Reset stream counter. Use only for testing."""
        self.counter = 0
