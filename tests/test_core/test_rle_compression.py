from src.core.alignment_nav import NavigationLog
from src.core.log_decoder import decode_log_stream


class MockStorage:
    def save_raw_navigation_log(self, *args):
        pass

    def load_raw_navigation_log(self, *args):
        return b""


def test_rle_identity_compression():
    """Test RLE compression of identity runs"""
    log = NavigationLog("test", MockStorage(), max_size=1024)

    # Add 34 identity operations (should compress to 2 RLE bytes)
    for _ in range(34):
        log.append(0, 0)  # Identity on both tensors

    # Add one different operation
    log.append(2, 3)  # Forward, Backward

    # Force flush
    log.shutdown()

    # Check compressed size
    raw = bytes(log._log)
    assert len(raw) == 3  # Two RLE bytes + one regular

    # Verify decompression
    decoded = list(decode_log_stream(iter(raw)))
    assert len(decoded) == 35
    assert decoded[:34] == [0x00] * 34
    assert decoded[34] == 0x32  # (3 << 4) | 2


def test_mixed_navigation_compression():
    """Test compression with mixed navigation events"""
    log = NavigationLog("test", MockStorage())

    # Pattern: 5 identity, 3 different, 10 identity
    for _ in range(5):
        log.append(0, 0)

    log.append(1, 2)  # Inverse, Forward
    log.append(2, 3)  # Forward, Backward
    log.append(3, 1)  # Backward, Inverse

    for _ in range(10):
        log.append(0, 0)

    log.shutdown()

    # Verify correct compression
    raw = bytes(log._log)
    decoded = list(decode_log_stream(iter(raw)))
    assert len(decoded) == 18  # 5 + 3 + 10

    # Check specific values
    assert decoded[5] == 0x21  # (2 << 4) | 1
    assert decoded[6] == 0x32  # (3 << 4) | 2
    assert decoded[7] == 0x13  # (1 << 4) | 3
