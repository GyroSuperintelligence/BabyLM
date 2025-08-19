# baby/kernel/bitops.py
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

# 0..255 popcount table
_POP_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def popcount_u64_array(arr: "NDArray[np.uint64]") -> "NDArray[np.int32]":
    """
    Vectorized popcount for uint64 ndarray.
    Returns an int32 ndarray of same shape.
    """
    a = arr.view(np.uint8).reshape(arr.size, 8)
    return _POP_LUT[a].sum(axis=1).astype(np.int32).reshape(arr.shape)

def agreements_u64_matrix(a: "NDArray[np.uint64]") -> "NDArray[np.int32]":
    """
    Given uint64 array 'a' of length N, compute N×N matrix of (48 - Hamming distance).
    """
    xor = (a[:, None] ^ a[None, :]).astype(np.uint64)
    hd = popcount_u64_array(xor)
    return 48 - hd

def bit_agreements(x: int, y: int) -> int:
    """48 - Hamming distance for two python ints masked to 48 bits."""
    return 48 - int((x ^ y) & 0xFFFFFFFFFFFF).bit_count()

def popcount_u64(x: int) -> int:
    """Popcount for a single 64-bit integer."""
    return int(x & 0xFFFFFFFFFFFFFFFF).bit_count()

def hamming_distance(x: int, y: int) -> int:
    """Hamming distance between two 48-bit values."""
    return int((x ^ y) & 0xFFFFFFFFFFFF).bit_count()