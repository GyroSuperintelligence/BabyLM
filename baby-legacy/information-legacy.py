import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from functools import lru_cache

from baby import governance
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
#  Common Source integer exported for tooling
# ------------------------------------------------------------------
#
# The Common Source (CS) is not a point within the topology, but its
# emergent origin—it generates structure without being structured.
# It is unobservable yet generative: not inert, not void, but
# reflectively productive.
#
CS_INT = governance.CS_INT  # re-use constant from physics

warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.

Build steps:
    python -m baby.information ontology     --output memories/public/meta/ontology_keys.npy
    python -m baby.information epistemology --keys memories/public/meta/ontology_keys.npy \
           --output  memories/public/meta/epistemology.npy
    python -m baby.information phenomenology --ep memories/public/meta/epistemology.npy \
           --keys memories/public/meta/ontology_keys.npy \
           --output memories/public/meta/phenomenology_map.npy
"""


# ---------- Tokenization & LEB128 Functions ----------


@lru_cache(maxsize=4)
def _cached_tokenizer(name: str, base_path_str: str) -> Any:
    from tokenizers import Tokenizer

    root = Path(base_path_str)
    path = root / "public" / "tokenizers" / name / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


def _load_tokenizer(
    name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> Any:
    """Load tokenizer from HuggingFace with caching."""
    return _cached_tokenizer(name, str(base_path))


def _id_to_bytes(idx: int) -> List[int]:
    """Convert token ID to LEB128 bytes."""
    if idx < 0:
        raise ValueError("Token ID must be non-negative")

    bytes_list = []
    while True:
        byte = idx & 0x7F
        idx >>= 7
        if idx == 0:
            bytes_list.append(byte)
            break
        else:
            bytes_list.append(byte | 0x80)
    return bytes_list


def _bytes_to_ids(blob: bytes) -> List[int]:
    """Convert LEB128 bytes to token IDs."""
    ids, cur, shift = [], 0, 0
    for i, b in enumerate(blob):
        if shift > 28:  # Prevent overflow (32-bit token ID assumption)
            raise ValueError(f"Token ID too large at byte {i}")
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            ids.append(cur)
            cur, shift = 0, 0
    if shift:
        raise ValueError("Incomplete token ID sequence")
    return ids


def _apply_mask(buf: bytes) -> bytes:
    """XOR every byte with 0xAA – vectorised & memory‑efficient."""
    # bytes ↔ introns is a pure involution: f(f(x)) == x
    return bytes(b ^ 0xAA for b in buf)


def encode_text(
    text: str, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> bytes:
    """Encode text to bytes via tokenizer + LEB128 (vectorized). Uses base_path for root."""
    # 1. text → token IDs ----------------------------------------------------
    tokenizer = _load_tokenizer(name, base_path)
    ids = tokenizer.encode(text).ids

    # 2. IDs → pure‑LEB128 intron stream ------------------------------------
    introns = bytearray(len(ids) * 5)  # worst‑case pre‑alloc
    pos = 0
    for tid in ids:
        val = tid
        while True:
            byte = val & 0x7F
            val >>= 7
            introns[pos] = byte | (0x80 if val else 0x00)
            pos += 1
            if not val:
                break

    # 3. intron stream → external masked bytes ------------------------------
    return _apply_mask(bytes(introns[:pos]))


def decode_text(
    blob: bytes, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> str:
    """Decode LEB128 bytes back to text via tokenizer. Uses base_path for root."""
    # 1. external bytes → intron stream -------------------------------------
    introns = _apply_mask(blob)

    # 2. intron stream → token IDs ------------------------------------------
    try:
        ids = _bytes_to_ids(introns)
        # Trim at [SEP] if present
        if SEP_ID in ids:
            ids = ids[: ids.index(SEP_ID)]
        # 3. IDs → text ------------------------------------------------------
        tokenizer = _load_tokenizer(name, base_path)
        return str(tokenizer.decode(ids))
    except Exception:
        # malformed stream fallback
        return blob.decode("utf-8", errors="replace")


def get_vocab_size(
    name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> int:
    """Get vocabulary size of a tokenizer. Uses base_path for root."""
    tokenizer = _load_tokenizer(name, base_path)
    return int(tokenizer.get_vocab_size())


def token_id_to_bytes(tok_id: int) -> bytes:
    """Convert a single token ID to bytes via LEB128 and apply the 0xAA mask."""
    introns = bytearray(5)  # worst-case pre-alloc
    pos = 0
    val = tok_id
    while True:
        byte = val & 0x7F
        val >>= 7
        introns[pos] = byte | (0x80 if val else 0x00)
        pos += 1
        if not val:
            break
    return _apply_mask(bytes(introns[:pos]))


def bytes_to_token_id(bs: bytes) -> int:
    """Convert bytes back to a single token ID. Assumes complete token."""
    # First unmask the bytes, then decode
    unmasked = _apply_mask(bs)
    ids = _bytes_to_ids(unmasked)
    if len(ids) != 1:
        raise ValueError(f"Expected single token ID, got {len(ids)}")
    return ids[0]


def bytes_to_token_ids(bs: bytes) -> List[int]:
    """Convert bytes back to a list of token IDs."""
    unmasked = _apply_mask(bs)
    return _bytes_to_ids(unmasked)


# ---------- ψ Isomorphism Functions ----------


def ψ(byte: int) -> int:
    """ψ isomorphism: byte → intron via XOR 0xAA."""
    return byte ^ 0xAA


def ψ_inv(intron: int) -> int:
    """ψ⁻¹ isomorphism: intron → byte via XOR 0xAA."""
    return intron ^ 0xAA


# ---------- Token ↔ Intron Conversion ----------


def token_to_introns(token_id: int) -> List[int]:
    """Convert a token ID to its LEB128 intron sequence.

    This is the ψ isomorphism: token_id → LEB128 bytes → introns via XOR 0xAA.
    Each intron is a single byte that can be fed directly to GyroSI physics.

    Args:
        token_id: The token ID to convert

    Returns:
        List of intron bytes (0-255) that represent the token
    """
    # Convert token ID to LEB128 bytes
    leb_bytes = _id_to_bytes(token_id)
    # Apply ψ isomorphism (XOR with 0xAA) to get introns
    introns = [b ^ 0xAA for b in leb_bytes]
    return introns


def introns_to_token(introns: List[int]) -> int:
    """Convert an intron sequence back to a token ID.

    This is the ψ⁻¹ isomorphism: introns → LEB128 bytes → token_id.

    Args:
        introns: List of intron bytes (0-255)

    Returns:
        The token ID that produced these introns
    """
    # Apply ψ⁻¹ isomorphism (XOR with 0xAA) to get LEB128 bytes
    leb_bytes = [i ^ 0xAA for i in introns]
    # Convert LEB128 bytes to token ID
    token_ids = _bytes_to_ids(bytes(leb_bytes))
    if len(token_ids) != 1:
        raise ValueError(f"Expected single token ID, got {len(token_ids)}")
    return token_ids[0]


@lru_cache(maxsize=1)
def _get_intron_trie(
    tokenizer_name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> Dict[Any, Any]:
    """Build and cache a trie mapping intron sequences to token IDs."""
    tokenizer = _load_tokenizer(tokenizer_name, base_path)
    vocab_size = tokenizer.get_vocab_size()

    trie: Dict[Any, Any] = {}
    for token_id in range(vocab_size):
        try:
            introns = token_to_introns(token_id)
            node = trie
            for intron in introns:
                if intron not in node:
                    node[intron] = {}
                node = node[intron]
            if "tokens" not in node:
                node["tokens"] = []
            node["tokens"].append(token_id)
        except (ValueError, IndexError):
            continue  # Skip invalid tokens

    return trie


def find_tokens_by_intron_prefix(
    intron_prefix: List[int],
    tokenizer_name: str = "bert-base-uncased",
    base_path: Path = Path(__file__).resolve().parents[1] / "memories",
) -> List[int]:
    """Find all tokens whose intron sequence starts with the given prefix.

    This implements efficient tokenizer trie lookup for the exon product sieve.
    Converts intron prefix to LEB128 bytes, then finds all tokens matching that prefix.

    Args:
        intron_prefix: List of intron bytes (0-255) representing the prefix
        tokenizer_name: Name of the tokenizer to use
        base_path: Base path for tokenizer files

    Returns:
        List of token IDs whose intron sequence starts with the given prefix
    """
    trie = _get_intron_trie(tokenizer_name, base_path)

    node = trie
    for intron in intron_prefix:
        if intron in node:
            node = node[intron]
        else:
            return []  # No tokens with this prefix

    # Collect all tokens from this node and its children
    matching_tokens = []

    def _collect_tokens(n: Dict[Any, Any]) -> None:
        if "tokens" in n:
            matching_tokens.extend(n["tokens"])
        for k, v in n.items():
            if k != "tokens":
                _collect_tokens(v)

    _collect_tokens(node)

    return matching_tokens


@lru_cache(maxsize=1)
def _get_reverse_intron_trie(
    tokenizer_name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> Dict[Any, Any]:
    """Build and cache a reverse trie mapping last introns to token IDs."""
    tokenizer = _load_tokenizer(tokenizer_name, base_path)
    vocab_size = tokenizer.get_vocab_size()

    trie: Dict[Any, Any] = {}
    for token_id in range(vocab_size):
        try:
            introns = token_to_introns(token_id)
            if introns:
                last_intron = introns[-1]
                if last_intron not in trie:
                    trie[last_intron] = []
                trie[last_intron].append(token_id)
        except (ValueError, IndexError):
            continue

    return trie


def find_tokens_by_last_intron(
    last_intron: int,
    tokenizer_name: str = "bert-base-uncased",
    base_path: Path = Path(__file__).resolve().parents[1] / "memories",
) -> List[int]:
    """Find tokens whose last intron matches the given byte.

    This is the core function for the exon product sieve generation.
    Efficiently finds tokens that could be generated from a specific exon product.

    Args:
        last_intron: Single intron byte (0-255) to match
        tokenizer_name: Name of the tokenizer to use
        base_path: Base path for tokenizer files

    Returns:
        List of token IDs whose last intron matches the given byte
    """
    trie = _get_reverse_intron_trie(tokenizer_name, base_path)
    return trie.get(last_intron, [])  # type: ignore[no-any-return]


# ---------- SEP Token Utilities ----------

SEP_ID = 102


def sep_bytes(count: int = 1) -> bytes:
    """Generate SEP token bytes for sentence/article boundaries."""
    introns = bytearray(count * 5)  # worst-case pre-alloc
    pos = 0
    for _ in range(count):
        val = SEP_ID
        while True:
            byte = val & 0x7F
            val >>= 7
            introns[pos] = byte | (0x80 if val else 0x00)
            pos += 1
            if not val:
                break
    return _apply_mask(bytes(introns[:pos]))


def encode_text_with_sep(
    text: str, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[1] / "memories"
) -> bytes:
    """Encode text and append a single SEP token."""
    return encode_text(text, name, base_path) + sep_bytes()


# ---------- Information Engine ----------


class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_array_indexing is True, ontology_map and inverse_ontology_map are stored
    as numpy arrays for better memory/cache performance.
    """

    _keys: NDArray[np.uint64] | None
    _inverse: NDArray[np.uint64] | None
    ontology_map: None
    inverse_ontology_map: None
    use_array_indexing: bool
    orbit_cardinality: NDArray[np.uint32]
    _theta_table: NDArray[np.float32] | None
    _v_max: int

    def __init__(self, keys_path: str, ep_path: str, phenomap_path: str, theta_path: str):
        import numpy as np
        from pathlib import Path

        # Load ontology keys
        self._keys = np.load(keys_path, mmap_mode="r")
        self._inverse = self._keys
        self.ontology_map = None
        self.inverse_ontology_map = None
        self.use_array_indexing = True

        # Load epistemology
        self.ep = np.load(ep_path, mmap_mode="r")

        # Load phenomenology and orbit cardinality
        self.orbit_cardinality = np.ones(len(self._keys) if self._keys is not None else 0, dtype=np.uint32)
        if phenomap_path:
            try:
                self.orbit_map = np.load(phenomap_path, mmap_mode="r")
                sizes_path = Path(phenomap_path).with_name("orbit_sizes.npy")
                if sizes_path.exists():
                    self.orbit_cardinality = np.load(sizes_path, mmap_mode="r")
                else:
                    self.orbit_cardinality = np.ones(len(self._keys) if self._keys is not None else 0, dtype=np.uint32)
            except Exception as e:
                logger.warning("Could not load phenomenology map from %s: %s", phenomap_path, e)
                logger.info("Continuing without phenomenology mapping...")
                self.orbit_map = None
                self.orbit_cardinality = np.ones(len(self._keys) if self._keys is not None else 0, dtype=np.uint32)

        # Load theta table
        if theta_path:
            try:
                self._theta_table = np.load(theta_path, mmap_mode="r")
            except Exception:
                self._theta_table = None
        else:
            self._theta_table = None

        self._v_max = 1 if self.orbit_cardinality is None else int(np.max(self.orbit_cardinality))

        # Early fail if theta.npy is missing or corrupt
        if self._theta_table is None:
            raise RuntimeError(
                f"theta.npy is missing or corrupt at {theta_path}; required for divergence calculations. "
                f"Please ensure theta.npy exists alongside epistemology.npy. "
                f"Run the ontology builder to generate missing assets."
            )

    def get_index_from_state(self, state_int: int) -> int:
        """
        Return ontology (state) index (0..N-1) for a 48-bit state integer.
        Uses fast array indexing if use_array_indexing is True, otherwise dict lookup.

        Args:
            state_int: 48-bit integer representing physical state

        Returns:
            Ontology index (0 to N-1)

        Raises:
            ValueError: If state not found in ontology
        """
        if self.use_array_indexing:
            if self._keys is None:
                raise RuntimeError("Array indexing arrays not initialized.")
            idx = np.searchsorted(self._keys, state_int)
            if idx == len(self._keys) or self._keys[idx] != state_int:
                raise ValueError(
                    f"State integer {state_int} not found in discovered ontology. "
                    f"This indicates a fundamental physics violation."
                )
            return int(idx)
        else:
            if self.ontology_map is None:
                raise RuntimeError("ontology_map is not available in array indexing mode.")
            index = self.ontology_map.get(state_int, -1)
            if index == -1:
                raise ValueError(
                    f"CRITICAL: State integer {state_int} not found in discovered ontology. "
                    f"This indicates a fundamental physics violation."
                )
            return index

    def get_state_from_index(self, index: int) -> int:
        """
        Get state integer from canonical index.

        Args:
            index: Canonical index (0 to N-1)

        Returns:
            48-bit state integer
        """
        if self.use_array_indexing:
            if self._inverse is None:
                raise RuntimeError("Array indexing arrays not initialized.")
            assert self._inverse is not None
            if index < 0 or index >= len(self._inverse):
                raise ValueError(f"Index {index} out of bounds for array indexing.")
            return int(self._inverse[index])
        else:
            if self.inverse_ontology_map is None:
                raise RuntimeError("inverse_ontology_map not initialized.")
            state_int = self.inverse_ontology_map.get(index)
            if state_int is None:
                assert self._keys is not None
                raise ValueError(f"Invalid index {index}, must be 0 to {len(self._keys) - 1}")
            return state_int

    @staticmethod
    def int_to_tensor(state_int: int) -> NDArray[np.int8]:
        """
        Converts a canonical 48-bit integer state to geometric tensor.

        Encoding: bit 0 (LSB) maps to element 47, bit 47 (MSB) maps to element 0.
        Bit values: 0 = +1, 1 = -1

        Args:
            state_int: 48-bit integer state

        Returns:
            Tensor with shape [4, 2, 3, 2] and values ±1
        """
        if state_int >= (1 << 48) or state_int < 0:
            raise ValueError(f"state_int {state_int} out of bounds for 48-bit representation")

        # Convert to 6 bytes (48 bits), big-endian
        state_packed_bytes = state_int.to_bytes(6, "big")

        # Unpack to individual bits
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8), bitorder="big")

        if bits.size != 48:
            raise ValueError(f"Expected 48 bits, got {bits.size}")

        # Convert: 0 -> +1, 1 -> -1 (encoding: bit = (value == -1))
        tensor_flat = (1 - 2 * bits).astype(np.int8)

        # Reshape to proper geometry with lexicographic ordering (C order)
        return tensor_flat.reshape(4, 2, 3, 2)

    @staticmethod
    def tensor_to_int(tensor: NDArray[np.int8]) -> int:
        """
        Converts a geometric tensor to its canonical 48-bit integer state.

        Encoding: bit = (value == -1), element 0 -> bit 47 (MSB), element 47 -> bit 0 (LSB)

        Args:
            tensor: NumPy array with shape [4, 2, 3, 2] and values ±1

        Returns:
            48-bit integer representation
        """
        if tensor.shape != (4, 2, 3, 2):
            raise ValueError(f"Expected tensor shape (4, 2, 3, 2), got {tensor.shape}")

        # Flatten in C-order and convert: +1 -> 0, -1 -> 1
        bits = (tensor.flatten(order="C") == -1).astype(np.uint8)

        # Pack bits into bytes
        packed = np.packbits(bits, bitorder="big")

        # Convert to integer, big-endian
        result = int.from_bytes(packed.tobytes(), "big")

        return result

    def gyrodistance_angular(self, T1: NDArray[np.int8], T2: NDArray[np.int8]) -> float:
        """
        Calculate angular divergence between tensors in radians.

        This measures the geometric alignment between two states in
        N-dimensional space using cosine similarity.

        Args:
            T1: First tensor [4, 2, 3, 2]
            T2: Second tensor [4, 2, 3, 2]

        Returns:
            Angular distance in radians (0 to π)
        """
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()

        # Cosine similarity in N-dimensional space
        cosine_similarity = np.dot(T1_flat, T2_flat) / T1_flat.size
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return float(np.arccos(cosine_similarity))

    def measure_state_divergence(self, state_int: int) -> float:
        if self._theta_table is None:
            raise RuntimeError("Theta table is not loaded. Cannot compute state divergence.")
        idx = self.get_index_from_state(state_int)
        return float(self._theta_table[idx])

    def measure_state_divergence_index(self, index: int) -> float:
        """Direct index-based θ calculation without binary search."""
        if self._theta_table is None:
            raise RuntimeError("Theta table missing")
        if index < 0 or index >= len(self._theta_table):
            raise IndexError("Index out of bounds")
        return float(self._theta_table[index])

    def get_orbit_cardinality(self, state_index: int) -> int:
        return int(self.orbit_cardinality[state_index])


# ==============================================================================
# Utility: Clean single-line progress reporter
# ==============================================================================
class ProgressReporter:
    def __init__(self, desc: str):
        self.desc = desc
        self.start_time = time.time()
        self.last_update = 0.0
        self.first_update = True

    def update(self, current: int, total: Optional[int] = None, extra: str = "") -> None:
        now = time.time()
        # Always show first update immediately
        if not self.first_update and now - self.last_update < 0.1 and (total is None or current != total):
            return

        self.first_update = False
        elapsed = now - self.start_time
        rate = current / elapsed if elapsed > 0 else 0

        msg = f"\r{self.desc}: {current:,}"
        if total is not None:
            pct = 100.0 * current / total
            msg += f"/{total:,} ({pct:.1f}%)"
        msg += f" | {rate:.0f}/s | {elapsed:.1f}s"
        if extra:
            msg += f" | {extra}"

        print(msg + " " * 20, end="", flush=True)
        self.last_update = now

    def done(self) -> None:
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: Done in {elapsed:.1f}s" + " " * 50)


def open_memmap_int32(
    filename: str,
    mode: str,
    shape: tuple[int, ...],
) -> NDArray[np.int32]:
    from numpy.lib.format import open_memmap as _open_memmap

    arr = _open_memmap(filename, dtype=np.int32, mode=mode, shape=shape)  # type: ignore[no-untyped-call]
    from typing import cast

    return cast(NDArray[np.int32], arr)


# ==============================================================================
# STEP 1: Ontology Discovery
# ==============================================================================
def discover_and_save_ontology(output_path: str) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """
    Discovers the complete 788,986 state manifold via BFS.

    The epistemology map E : ℳ × ℐ → ℳ embeds the formal grammar where:
    - ℳ is the ontology manifold (the set of state indices)
    - ℐ is the intron space (256 discrete elements)
    - CS is treated as extra-phenomenal, handled at the boundary layer

    The Common Source (CS) is a pre-observable axiom, not an operational node
    within the manifold. All transitions use generic physics.
    """
    EXPECTED_SIZE = 788_986  # original manifold size
    progress = ProgressReporter("Discovering ontology")

    origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    discovered = {origin_int}
    current_level = [origin_int]
    depth = 0
    layer_sizes: List[int] = []  # Track number of new states at each BFS depth

    while current_level:
        next_level_set = set()
        for state in current_level:
            for intron in range(256):
                next_state = governance.apply_gyration_and_transform(state, intron)
                if next_state not in discovered:
                    discovered.add(next_state)
                    next_level_set.add(next_state)

        current_level = list(next_level_set)
        if not current_level:
            break
        depth += 1
        layer_sizes.append(len(current_level))
        progress.update(len(discovered), total=EXPECTED_SIZE, extra=f"depth={depth}")

    progress.done()

    # Print expansion pattern for verification
    logger.info("Layer sizes (expansion pattern): %s", layer_sizes)
    assert sum(layer_sizes) + 1 == len(discovered), "Layer sizes do not sum to total states (including origin)"

    # Validate
    if len(discovered) != EXPECTED_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_SIZE:,} states, found {len(discovered):,}")

    if depth != 6:
        raise RuntimeError(f"Expected diameter 6, found {depth}")

    # Save sorted keys as npy
    keys = np.array(sorted(discovered), dtype=np.uint64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, keys)
    # --- DOCUMENTATION: Archetypal state indices ---
    # The archetypal state (GENE_Mac_S) is included in the ontology, but its index is determined by its
    # integer value after sorting all discovered states. It is NOT guaranteed to be at index 0.
    # To find its index:
    #   archetypal_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    #   archetypal_index = np.where(keys == archetypal_int)[0][0]
    # Current ontology (788,986 states after CS refactoring):
    #   GENE_Mac_S (CS state, θ=0): index 549871 (0xA9556AA9556A)
    #   UNA archetype (θ≈π/4): index 35495 (0x09116A09116A)
    # Note: GENE_Mac_S is the CS (Common Source) state with θ=0, serving as the extra-phenomenal
    # reference point. The UNA state at θ≈π/4 serves as the primary phenomenal archetype.
    logger.info("Saved ontology keys to: %s", output_path)

    return keys


# ==============================================================================
# STEP 2: Epistemology Table
# ==============================================================================
def build_state_transition_table(keys_path: str, output_path: str) -> None:
    """Builds the N×256 state transition table with validation."""
    progress = ProgressReporter("Building epistemology")

    states = np.load(keys_path, mmap_mode="r")
    N = len(states)

    # ----- θ table (angular divergence from origin) -----
    theta_path = output_path.replace("epistemology.npy", "theta.npy")
    origin = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    acos_lut = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
    theta = np.empty(N, dtype=np.float32)
    for i, s in enumerate(states):
        h = int(s ^ origin).bit_count()
        theta[i] = acos_lut[h]

    # Memory-mapped output
    ep = open_memmap_int32(output_path, "w+", (N, 256))

    # Process in chunks for memory efficiency
    CHUNK_SIZE = 10_000
    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        chunk_states = states[chunk_start:chunk_end]
        # Vectorized: apply all introns at once
        next_states_all = governance.apply_gyration_and_transform_all_introns(chunk_states)
        # next_states_all shape: (chunk_len, 256)
        idxs = np.searchsorted(states, next_states_all, side="left")
        # Debug check: ensure all next_states are in the ontology
        if __debug__:
            if idxs.max() >= states.size or not np.all(states[idxs] == next_states_all):
                raise RuntimeError("Transition produced unknown state.")
        ep[chunk_start:chunk_end, :] = idxs
        progress.update(chunk_end, N)

    # Save theta table
    np.save(theta_path, theta)
    ep.flush()  # type: ignore[attr-defined]
    progress.done()


# ==============================================================================
# STEP 3: Phenomenology Map (Core + Optional Diagnostics)
# ==============================================================================


def _compute_sccs(
    ep: NDArray[Any], idx_to_state: NDArray[Any], introns_to_use: List[int]
) -> Tuple[NDArray[Any], Dict[int, int], List[int]]:
    """
    Core Tarjan's SCC algorithm restricted to a subset of introns.
    Optimized: neighbors() does not np.unique, just returns ep[v, introns_arr].
    """
    N = ep.shape[0]
    indices = np.full(N, -1, dtype=np.int32)
    lowlink = np.zeros(N, dtype=np.int32)
    on_stack = np.zeros(N, dtype=bool)
    stack: List[int] = []
    canonical = np.full(N, -1, dtype=np.int32)
    orbit_sizes: Dict[int, int] = {}
    reps: List[int] = []
    counter = 0
    introns_arr = np.array(introns_to_use, dtype=np.int32)

    def neighbors(v: int) -> NDArray[np.int32]:
        # Return all neighbors; duplicates are fine
        return np.asarray(ep[v, introns_arr], dtype=np.int32)

    for root in range(N):
        if indices[root] != -1:
            continue
        dfs_stack = [(root, iter(neighbors(root)))]
        indices[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while dfs_stack:
            v, child_iter = dfs_stack[-1]
            try:
                while True:
                    w = int(next(child_iter))
                    if indices[w] == -1:
                        # Tree edge: recurse
                        indices[w] = lowlink[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack[w] = True
                        dfs_stack.append((w, iter(neighbors(w))))
                        break
                    elif on_stack[w]:
                        # Back edge to a node in current SCC: update lowlink
                        if indices[w] < lowlink[v]:
                            lowlink[v] = indices[w]
                        continue  # keep looping children
                    else:
                        # Edge to an already closed SCC – ignore
                        continue
            except StopIteration:
                dfs_stack.pop()
                if dfs_stack:
                    parent_v, _ = dfs_stack[-1]
                    if lowlink[v] < lowlink[parent_v]:
                        lowlink[parent_v] = lowlink[v]
                if lowlink[v] == indices[v]:
                    comp = []
                    while True:
                        node = stack.pop()
                        on_stack[node] = False
                        comp.append(node)
                        if node == v:
                            break
                    comp_arr = np.array(comp, dtype=np.int32)
                    comp_states = idx_to_state[comp_arr]
                    rep = int(comp_arr[np.argmin(comp_states)])
                    canonical[comp_arr] = rep
                    orbit_sizes[rep] = comp_arr.size
                    reps.append(rep)

    assert np.all(canonical >= 0), "Unassigned nodes after SCC computation"
    return canonical, orbit_sizes, reps


def build_phenomenology_map(ep_path: str, keys_path: str, output_path: str) -> None:
    """
    Builds the canonical phenomenology map for GyroSI runtime operations.
    Args:
        ep_path: Path to epistemology.npy
        keys_path: Path to ontology_keys.npy
        output_path: Path to save phenomenology_map.npy
    """
    logger.info("[Phenomenology Core Builder]")

    # Load data
    ep = np.load(ep_path, mmap_mode="r")
    keys = np.load(keys_path, mmap_mode="r")
    N = ep.shape[0]

    # Build index→state lookup array
    idx_to_state = keys

    # Core: Compute canonical phenomenology (all 256 introns)
    logger.info("Computing canonical phenomenology (all 256 introns)...")
    all_introns = list(range(256))
    canonical, orbit_sizes, _ = _compute_sccs(ep, idx_to_state, all_introns)

    # Detailed statistics
    unique_reps = np.unique(canonical)
    num_orbits = len(unique_reps)
    logger.info("Found %d canonical orbits (expected 256)", num_orbits)

    # Orbit size distribution analysis
    sizes = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        rep = canonical[i]
        sizes[i] = orbit_sizes[rep]

    orbit_size_distribution = {}
    for size in np.unique(sizes):
        count = np.sum(sizes == size)
        orbit_size_distribution[int(size)] = count

    logger.info("Orbit size distribution:")
    total_states_check = 0
    for size in sorted(orbit_size_distribution.keys()):
        count = orbit_size_distribution[size]
        num_orbits_of_size = count // size
        total_states_check += count
        logger.info("  Size %4d: %6s states (%4d orbits)", size, f"{count:,}", num_orbits_of_size)

    logger.info("Total states verified: %s (expected %s)", f"{total_states_check:,}", f"{N:,}")

    # Self-consistency check
    self_consistent = sum(1 for rep in unique_reps if canonical[rep] == rep)
    logger.info("Self-consistent representatives: %d/%d", self_consistent, num_orbits)

    # Save canonical as .npy
    np.save(output_path, canonical.astype(np.int32))
    # Save orbit_sizes as orbit_sizes.npy
    # Each state should have the cardinality of its orbit
    sizes = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        rep = canonical[i]  # Get the representative for this state
        sizes[i] = orbit_sizes[rep]  # Set the cardinality of this state's orbit
    np.save(str(Path(output_path).with_name("orbit_sizes.npy")), sizes)
    logger.info("Saved canonical phenomenology to: %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GyroSI asset builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ontology
    p_ont = subparsers.add_parser("ontology", help="Step 1: Discover the full state manifold")
    p_ont.add_argument(
        "--output",
        required=True,
        help="Path to save ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)",
    )

    # Epistemology
    p_epi = subparsers.add_parser("epistemology", help="Step 2: Build state transition table")
    p_epi.add_argument(
        "--keys", required=True, help="Path to ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)"
    )
    p_epi.add_argument(
        "--output",
        required=True,
        help="Path to save epistemology.npy (recommended: memories/public/meta/epistemology.npy)",
    )

    # Phenomenology
    p_pheno = subparsers.add_parser("phenomenology", help="Step 3: Build canonical orbit map")
    p_pheno.add_argument(
        "--ep", required=True, help="Path to epistemology.npy (recommended: memories/public/meta/epistemology.npy)"
    )
    p_pheno.add_argument(
        "--keys", required=True, help="Path to ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)"
    )
    p_pheno.add_argument(
        "--output",
        required=True,
        help="Path to save phenomenology_map.npy (recommended: memories/public/meta/phenomenology_map.npy)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.command == "ontology":
            logger.info("[Step 1] Ontology Generation")
            ontology_map = discover_and_save_ontology(args.output)
            logger.info("Saved: %s", args.output)
            logger.info("Total states discovered: %s", f"{len(ontology_map):,}")

        elif args.command == "epistemology":
            logger.info("[Step 2] Epistemology Table")
            build_state_transition_table(args.keys, args.output)
            file_size = os.path.getsize(args.output) / 1024**2
            logger.info("Saved: %s", args.output)
            logger.info("File size: %.1f MB", file_size)

        elif args.command == "phenomenology":
            logger.info("[Step 3] Phenomenology Mapping")
            build_phenomenology_map(args.ep, args.keys, args.output)
            # No final summary, as output is now a .npy file

        else:
            logger.error("Unknown command")
            sys.exit(1)

    except Exception as e:
        logger.exception("Error: %s", e)
        # Re-raise to preserve stack trace for debugging
        raise
