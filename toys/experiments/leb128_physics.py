"""
LEB128 ↔ GyroSI Physics Mapping

This module implements the mathematical isomorphism between LEB128 encoding
and GyroSI intron physics, enabling direct token-level state transitions
and endogenous compression.
"""

import struct
from typing import List, Tuple, Optional, Iterator
import numpy as np

# The boundary isomorphism: byte ↔ intron
def ψ(byte: int) -> int:
    """Boundary transcription: byte → intron via XOR 0xAA."""
    return byte ^ 0xAA

def ψ_inv(intron: int) -> int:
    """Inverse boundary transcription: intron → byte."""
    return intron ^ 0xAA  # ψ is its own inverse

# LEB128 encoding/decoding with GyroSI physics
def encode_token_to_leb128(token_id: int) -> List[int]:
    """Encode token_id to LEB128 bytes."""
    bytes_list = []
    value = token_id
    
    while True:
        byte = value & 0x7F
        value >>= 7
        if value == 0:
            bytes_list.append(byte)  # Final byte: bit 7 = 0
            break
        else:
            bytes_list.append(byte | 0x80)  # Continue: bit 7 = 1
    
    return bytes_list

def decode_leb128_to_token(bytes_list: List[int]) -> int:
    """Decode LEB128 bytes back to token_id."""
    token_id = 0
    shift = 0
    
    for i, byte in enumerate(bytes_list):
        payload = byte & 0x7F
        token_id |= payload << shift
        shift += 7
        
        if (byte & 0x80) == 0:  # Final byte
            break
    
    return token_id

def token_to_introns(token_id: int) -> List[int]:
    """Convert token_id directly to intron sequence using ψ."""
    leb_bytes = encode_token_to_leb128(token_id)
    return [ψ(b) for b in leb_bytes]

def introns_to_token(introns: List[int]) -> int:
    """Convert intron sequence back to token_id using ψ⁻¹."""
    leb_bytes = [ψ_inv(i) for i in introns]
    return decode_leb128_to_token(leb_bytes)

# Token-level state transitions
class TokenSTT:
    """Pre-computed token-level state transition table."""
    
    def __init__(self, epistemology: np.ndarray, vocab_size: int):
        self.epistemology = epistemology
        self.vocab_size = vocab_size
        self.cache: dict = {}  # Lazy loading of token transitions
        
    def get_token_transition(self, state_index: int, token_id: int) -> int:
        """Get the final state after applying a token's intron sequence."""
        key = (state_index, token_id)
        
        if key not in self.cache:
            # Compute the full token walk
            introns = token_to_introns(token_id)
            final_state = state_index
            
            for intron in introns:
                final_state = self.epistemology[final_state, intron]
            
            self.cache[key] = final_state
        
        return self.cache[key]
    
    def precompute_common_tokens(self, token_frequencies: dict, threshold: float = 0.01):
        """Pre-compute transitions for frequently used tokens."""
        total_freq = sum(token_frequencies.values())
        
        for token_id, freq in token_frequencies.items():
            if freq / total_freq > threshold:
                # Pre-compute for all states
                for state in range(self.epistemology.shape[0]):
                    self.get_token_transition(state, token_id)

# Minimal phenotype record structure
class MinimalPhenotype:
    """Minimal phenotype record with only essential fields."""
    
    def __init__(self, state_index: int, token_id: int, exon_mask: int = 0, confidence: float = 0.1):
        self.state_index = state_index
        self.token_id = token_id
        self.exon_mask = exon_mask
        self.confidence = confidence
    
    def to_bytes(self) -> bytes:
        """Serialize to minimal byte representation."""
        # Pack as: state_index (24 bits) + token_id (18 bits) + exon_mask (8 bits) + confidence (32 bits)
        # This gives us 82 bits = ~10.25 bytes, rounded to 12 bytes for alignment
        return struct.pack("<IIf", self.state_index, self.token_id, self.confidence)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MinimalPhenotype':
        """Deserialize from byte representation."""
        state_index, token_id, confidence = struct.unpack("<IIf", data)
        return cls(state_index, token_id, 0, confidence)  # exon_mask computed on demand

# Endogenous compression utilities
def compress_intron_stream(introns: List[int], output_file: str):
    """Compress intron stream using Zstandard."""
    try:
        import zstandard as zstd  # type: ignore
        compressor = zstd.ZstdCompressor(level=5)
        
        with open(output_file, 'wb') as f:
            compressed = compressor.compress(bytes(introns))
            f.write(compressed)
            
        return len(compressed)
    except ImportError:
        # Fallback to simple storage
        with open(output_file, 'wb') as f:
            f.write(bytes(introns))
        return len(introns)

def decompress_intron_stream(input_file: str) -> List[int]:
    """Decompress intron stream."""
    try:
        import zstandard as zstd  # type: ignore
        decompressor = zstd.ZstdDecompressor()
        
        with open(input_file, 'rb') as f:
            compressed = f.read()
            introns = list(decompressor.decompress(compressed))
            
        return introns
    except ImportError:
        # Fallback to simple reading
        with open(input_file, 'rb') as f:
            return list(f.read())

# Stream processing utilities
def text_to_intron_stream(text: str, tokenizer) -> Iterator[int]:
    """Convert text to intron stream using tokenizer + LEB128 + ψ."""
    for token_id in tokenizer.encode(text):
        for intron in token_to_introns(token_id):
            yield intron

def intron_stream_to_text(intron_stream: Iterator[int], tokenizer) -> str:
    """Convert intron stream back to text."""
    current_token_bytes = []
    tokens = []
    
    for intron in intron_stream:
        byte = ψ_inv(intron)
        current_token_bytes.append(byte)
        
        if (byte & 0x80) == 0:  # Token complete
            token_id = tokenizer.bytes_to_id(bytes(current_token_bytes))
            tokens.append(token_id)
            current_token_bytes.clear()
    
    return tokenizer.decode(tokens)

# Physics integration
def apply_token_physics(state: int, token_id: int, epistemology: np.ndarray) -> int:
    """Apply a token's physics directly using intron sequence."""
    introns = token_to_introns(token_id)
    
    for intron in introns:
        state = epistemology[state, intron]
    
    return state

def compute_token_divergence(token_id: int, theta_map: np.ndarray, epistemology: np.ndarray) -> float:
    """Compute the angular divergence introduced by a token."""
    # Start from archetypal state
    archetypal_state = 0  # GENE_Mac_S equivalent
    final_state = apply_token_physics(archetypal_state, token_id, epistemology)
    
    return theta_map[final_state] 