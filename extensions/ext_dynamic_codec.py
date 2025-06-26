"""
ext_dynamic_codec.py  –  Dynamic Encoder / Decoder for GyroSI
-------------------------------------------------------------

• Learns a reversible token dictionary on-the-fly (LZ78-like).
• Operates on 48-navigation-event cycles (one CGM "symbol").
• Emits a varint-encoded token stream; flush is size/time based.
• Stateless on disk – replaying the token stream rebuilds
  the identical dictionary, so the Genome log is sufficient.

Author: 2024-06 GyroSI project
"""

from __future__ import annotations

import threading
import time
from typing import Protocol
from typing_extensions import override  # For Python < 3.12


# These imports don't have type stubs, but we can't fix that here
# We'll work around it with protocols
from extensions.base import GyroExtension  # type: ignore
from extensions.ext_cryptographer import ext_Cryptographer  # type: ignore

##############################################################################
# Helper functions – var-int coding (7-bit continuation)                     #
##############################################################################


class ManagerProtocol(Protocol):
    """Protocol for manager interface."""

    def gyro_structural_memory(self, tag: str, data: object | None = None) -> object:
        """Access structural memory."""
        ...


def _varint_encode(number: int) -> bytes:
    """Encode a non-negative integer as a variable-length byte sequence."""
    if number < 0:
        raise ValueError("Cannot encode negative numbers")
    if number == 0:
        return b"\x00"
    
    result = bytearray()
    while number > 0:
        result.append(number & 0x7F)
        number >>= 7
        if number > 0:
            result[-1] |= 0x80
    return bytes(result)


def _varint_iter(stream: bytes) -> list[int]:
    """Decode a stream of varint-encoded numbers."""
    result = []
    i = 0
    while i < len(stream):
        number = 0
        shift = 0
        while i < len(stream):
            byte = stream[i]
            i += 1
            number |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        result.append(number)
    return result


##############################################################################
# Dynamic codec extension                                                    #
##############################################################################


class ext_DynamicCodec(GyroExtension):
    """
    Universal, format-agnostic codec that learns a dictionary of
    navigation-cycle patterns on the first pass and reproduces them on replay.

    Encoding pipeline:  nav_event  -> 48-cycle ring  -> dictionary lookup
                        -> varint token(s)           -> cryptographer
                        -> manager output
    """

    # ---- CONFIGURABLE -----------------------------------------------------
    cycle_len: int = 48  # CGM full cycle
    flush_bytes: int = 256  # emit when ≥ this many payload bytes
    flush_ms: int = 50  # or this many ms since last flush
    max_dict: int = 4096  # hard cap on dictionary size
    prune_keep_ratio: float = 0.5  # retain this fraction when pruning
    # ----------------------------------------------------------------------

    def __init__(
        self,
        crypto: ext_Cryptographer | None = None,
    ) -> None:
        self.lock: threading.RLock = threading.RLock()

        # external collaborators
        self.crypto: ext_Cryptographer | None = crypto
        self.manager: ManagerProtocol | None = None  # set later via set_manager()

        # encoder state
        self._cycle_buf: list[int] = []
        self._dict: dict[bytes, int] = {}  # phrase → token
        self._dict_rev: dict[int, bytes] = {}  # token → phrase (for decode)
        self._next_code: int = 1  # 0 is reserved for "null"
        self._prev_phrase: bytes = b""  # last accepted phrase

        # output buffer
        self._out: bytearray = bytearray()
        self._last_flush: float = time.time()

        # statistics
        self._stats: dict[str, int] = {
            "encoded_tokens": 0,
            "decoded_tokens": 0,
            "dict_size": 0,
            "flush_count": 0,
        }

    # ------------------------------------------------------------------ API

    def process_navigation_event(self, nav_event: int, *_: object) -> None:
        """Public hook – push a single nav-event (0-255)."""
        self.encode_nav(nav_event)

    def process_nav(self, nav_event: int, *_: object) -> None:
        """Legacy alias for process_navigation_event."""
        self.process_navigation_event(nav_event, *_)

    # -----------------  ENCODER  ------------------------------------------

    def encode_nav(self, nav_event: int) -> None:
        """Accumulate nav events, emit token stream as needed."""
        with self.lock:
            self._cycle_buf.append(nav_event & 0xFF)

            if len(self._cycle_buf) < self.cycle_len:
                return  # not full yet

            cycle_bytes = bytes(self._cycle_buf[: self.cycle_len])
            del self._cycle_buf[: self.cycle_len]

            # === LZ78-style phrase lookup / insert =======================
            phrase = self._prev_phrase + cycle_bytes
            if phrase in self._dict:
                self._prev_phrase = phrase
                return  # keep extending
            else:
                # output code for prev_phrase
                token = self._dict.get(self._prev_phrase, 0)
                self._out.extend(_varint_encode(token))
                self._stats["encoded_tokens"] += 1

                # add new phrase to dict
                if self._next_code < self.max_dict:
                    self._dict[phrase] = self._next_code
                    self._dict_rev[self._next_code] = phrase
                    self._next_code += 1
                else:
                    self._prune_dictionary()

                # reset phrase to current cycle
                self._prev_phrase = cycle_bytes

            self._maybe_flush()

    # ----------------  DECODER  ------------------------------------------

    def decode_tokens(self, token_stream: bytes) -> list[bytes]:
        """
        Replay path: given varint-encoded token stream,
        return list of navigation-cycle bytes (48 each).
        """
        cycles: list[bytes] = []
        with self.lock:
            for token in _varint_iter(token_stream):
                phrase = self._dict_rev.get(token, b"")
                if token and not phrase:  # corrupt log
                    raise ValueError(f"Unknown token {token}")

                cycles.append(phrase[-self.cycle_len :] if phrase else b"")
                if self._prev_phrase:  # build new dictionary phrase
                    new_phrase = self._prev_phrase + (phrase or cycles[-1])
                else:
                    new_phrase = phrase or cycles[-1]

                if self._next_code < self.max_dict:
                    self._dict[new_phrase] = self._next_code
                    self._dict_rev[self._next_code] = new_phrase
                    self._next_code += 1
                else:
                    self._prune_dictionary()

                self._prev_phrase = phrase or cycles[-1]
                self._stats["decoded_tokens"] += 1

        return cycles

    # ----------------  INTERNALS  ----------------------------------------

    def _maybe_flush(self) -> None:
        """Flush buffer by size or time; hand off to crypto."""
        now = time.time()
        if (len(self._out) >= self.flush_bytes) or (
            (now - self._last_flush) * 1000 >= self.flush_ms
        ):
            payload: bytes = bytes(self._out)
            self._out.clear()
            self._last_flush = now
            self._stats["flush_count"] += 1

            # cryptographer stage
            if self.crypto:
                payload = self.crypto.encrypt(payload)

            # emit to manager or stdout
            self._emit(payload)

            # housekeeping: reset phrase boundary after flush
            self._prev_phrase = b""

    def _emit(self, data: bytes) -> None:
        """Send *data* to structural memory or fallback to stdout."""
        if self.manager is not None:
            try:
                # Using Protocol type, no need for hasattr/getattr
                self.manager.gyro_structural_memory(
                    "current.gyrotensor_nest.output.compressed", data=data
                )
            except Exception as e:
                print(f"[GyroSI][WARN] manager error: {e!r}")
                print(data)
        else:
            print(f"[GyroSI][TOKENS] {data.hex()}")

    # ------------------  DICTIONARY PRUNING  -----------------------------

    def _prune_dictionary(self) -> None:
        """Deterministically drop oldest entries to keep footprint bound."""
        keep = int(self.max_dict * self.prune_keep_ratio)
        # note: code points < keep retained, rest re-numbered consecutively
        surviving_codes = range(1, keep)
        new_dict: dict[bytes, int] = {}
        new_rev: dict[int, bytes] = {}
        for new_code, old_code in enumerate(surviving_codes, start=1):
            phrase = self._dict_rev.get(old_code)
            if phrase:
                new_dict[phrase] = new_code
                new_rev[new_code] = phrase
        self._dict, self._dict_rev = new_dict, new_rev
        self._next_code = keep
        self._stats["dict_size"] = len(self._dict)

    # -------------  GYRO-EXTENSION INTERFACE  ----------------------------

    @override
    def get_extension_name(self) -> str:
        return "ext_dynamic_codec"

    @override
    def get_extension_version(self) -> str:
        return "0.1.0"

    @override
    def get_footprint_bytes(self) -> int:
        return (
            len(self._dict) * 64 + len(self._out) + len(self._cycle_buf) + 64  # rough phrase memory
        )  # fixed overhead

    # learning / session state ===================================================================

    @override
    def get_learning_state(self) -> dict[str, object]:
        # dictionary is regenerated during replay — nothing to export
        return {}

    @override
    def set_learning_state(self, state: dict[str, object]) -> None:
        pass

    @override
    def get_session_state(self) -> dict[str, object]:
        return {
            "buffer": bytes(self._out),
            "cycle_buf": self._cycle_buf.copy(),
            "stats": self._stats.copy(),
        }

    @override
    def set_session_state(self, state: dict[str, object]) -> None:
        buffer = state.get("buffer", b"")
        if isinstance(buffer, bytes):
            self._out = bytearray(buffer)

        cycle_buf = state.get("cycle_buf", [])
        if isinstance(cycle_buf, list):
            self._cycle_buf = []
            for item in cycle_buf:
                if isinstance(item, int):
                    self._cycle_buf.append(item)

        stats = state.get("stats", {})
        if isinstance(stats, dict):
            for key_str, value in stats.items():
                if isinstance(key_str, str) and isinstance(value, int):
                    self._stats[key_str] = value

    # ----------------------  UTILITIES  -----------------------------------

    def analyze_codec(self) -> dict[str, int]:
        """Return live statistics for dashboards or tests."""
        self._stats["dict_size"] = len(self._dict)
        return self._stats.copy()

    def set_manager(self, manager: ManagerProtocol) -> None:
        """Set manager reference."""
        self.manager = manager
