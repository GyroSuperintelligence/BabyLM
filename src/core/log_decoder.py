"""
RLE decoder for navigation logs with identity compression.
Handles the expansion of compressed identity runs.
"""

_RLE_FLAG_NIBBLE = 0xE  # High nibble 1110 marks RLE run
_IDENTITY_EVENT = 0x00  # Double-identity event (compressible)


def decode_log_stream(raw_byte_iterator):
    """
    Decode RLE-compressed navigation log back to individual events.

    Args:
        raw_byte_iterator: Iterator over raw log bytes

    Yields:
        Individual navigation event bytes (decompressed)
    """
    for byte in raw_byte_iterator:
        high_nibble = (byte >> 4) & 0x0F
        low_nibble = byte & 0x0F

        if high_nibble == _RLE_FLAG_NIBBLE:
            # RLE run: low nibble encodes (run_length - 2)
            run_length = low_nibble + 2
            for _ in range(run_length):
                yield _IDENTITY_EVENT
        else:
            # Regular navigation event
            yield byte


def encode_identity_run(run_length: int) -> bytes:
    """
    Encode a run of identity events using RLE.

    Args:
        run_length: Number of consecutive identity events (2-17)

    Returns:
        RLE-encoded bytes
    """
    if run_length < 2:
        # Not worth compressing
        return bytes([_IDENTITY_EVENT] * run_length)
    elif run_length <= 17:
        # Encode as single RLE byte
        count_nibble = (run_length - 2) & 0x0F
        rle_byte = (_RLE_FLAG_NIBBLE << 4) | count_nibble
        return bytes([rle_byte])
    else:
        # Split into multiple RLE runs
        result = b""
        remaining = run_length
        while remaining > 0:
            chunk = min(17, remaining)
            result += encode_identity_run(chunk)
            remaining -= chunk
        return result
