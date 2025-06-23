"""
Per-user symmetric cryptography for Gene and Genome encryption.
Implements the missing encryption capability from your current system.
"""

import hashlib
from typing import Dict, Any
from .base import GyroExtension


class ext_Cryptographer(GyroExtension):
    """
    Symmetric encryption for Gene and Genome data at rest.
    Uses deterministic stream cipher based on user key.
    """

    def __init__(self, user_key: bytes):
        """
        Initialize with user-specific key.

        Args:
            user_key: 16-32 byte user-specific encryption key
        """
        if len(user_key) < 16:
            raise ValueError("User key must be at least 16 bytes")

        self.user_key = user_key[:32]  # Truncate to 32 bytes max
        self.counter = 0  # 2-byte session counter for keystream generation

    def get_extension_name(self) -> str:
        return "ext_Cryptographer"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        return 2  # Just the counter

    def _generate_keystream(self, length: int) -> bytes:
        """Generate deterministic keystream for encryption/decryption"""
        hasher = hashlib.blake2s(digest_size=length, key=self.user_key)
        hasher.update(self.counter.to_bytes(2, "big"))
        return hasher.digest()

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using XOR stream cipher.

        Args:
            data: Plaintext bytes to encrypt

        Returns:
            Encrypted bytes
        """
        keystream = self._generate_keystream(len(data))
        self.counter = (self.counter + 1) & 0xFFFF  # Wrap at 16-bit

        return bytes(b ^ k for b, k in zip(data, keystream))

    def decrypt(self, data: bytes) -> bytes:
        """
        Decrypt data (XOR stream cipher is symmetric).

        Args:
            data: Encrypted bytes to decrypt

        Returns:
            Decrypted bytes
        """
        # XOR stream cipher: decrypt = encrypt
        return self.encrypt(data)

    def get_learning_state(self) -> Dict[str, Any]:
        """Learning state for knowledge export"""
        return {
            "counter": self.counter,
            "key_hash": hashlib.sha256(self.user_key).hexdigest()[:16],  # For verification
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Session state (non-exportable)"""
        return {}

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learning state"""
        self.counter = state.get("counter", 0)

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore session state"""
        pass

    def process_navigation_event(self, event, input_byte=None):
        """No processing needed for crypto extension"""
        return None

    def get_pattern_filename(self) -> str:
        """Pattern filename for knowledge export"""
        return "ext_cryptographer@1.0.0.keystate"
