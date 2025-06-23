from src.core.gyro_core import GyroEngine


def test_engine_always_returns_operators():
    """Verify engine never returns None"""
    engine = GyroEngine()

    # Test all possible inputs
    for byte_val in range(256):
        ops = engine.execute_cycle(byte_val)
        assert ops is not None
        assert len(ops) == 2
        assert isinstance(ops[0], int)
        assert isinstance(ops[1], int)


def test_engine_deterministic():
    """Verify deterministic operation"""
    engine1 = GyroEngine()
    engine2 = GyroEngine()

    # Both engines should produce identical results
    for byte_val in [0, 127, 255]:
        ops1 = engine1.execute_cycle(byte_val)
        ops2 = engine2.execute_cycle(byte_val)
        assert ops1 == ops2


def test_matrix_integrity():
    """Verify operator matrix loads correctly"""
    engine = GyroEngine()
    assert hasattr(engine, "_operator_matrix")
    assert engine._operator_matrix.shape == (48, 256)
