"""
test_integration_complete.py - Complete System Integration Test

Tests the entire GyroSI stack working together:
- GyroEngine (pure navigation)
- ExtensionManager (orchestration)
- GyroAPI (public interface)
- All critical extensions
- Knowledge/session separation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.gyro_api import (
    initialize_session,
    process_text,
    process_byte_stream,
    export_knowledge,
    shutdown_session,
    validate_system_integrity,
)


class TestCompleteSystemIntegration:
    """Test the complete GyroSI system end-to-end."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_cwd = Path.cwd()

        # Create data structure
        data_dir = Path(temp_dir) / "data"
        (data_dir / "sessions").mkdir(parents=True)
        (data_dir / "knowledge").mkdir(parents=True)

        # Change to temp directory so system finds our data folder
        import os

        os.chdir(temp_dir)

        yield temp_dir

        # Cleanup
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    def test_complete_system_workflow(self, temp_data_dir):
        """Test complete workflow: initialize -> process -> export -> shutdown."""

        # 1. Initialize session
        session_id = initialize_session()
        assert session_id is not None
        assert len(session_id) > 0

        try:
            # 2. Process simple text
            resonance_count = process_text(session_id, "Hello World")
            print(f"Resonances from 'Hello World': {resonance_count}")

            # Should have some navigation events
            assert resonance_count >= 0  # Could be 0 if no structural resonance

            # 3. Process byte stream
            test_bytes = [72, 101, 108, 108, 111]  # "Hello"
            byte_resonances = process_byte_stream(session_id, test_bytes)
            print(f"Resonances from byte stream: {byte_resonances}")

            # 4. Validate system integrity
            is_valid, details = validate_system_integrity()
            print(f"System integrity: {is_valid}")
            print(f"Validation details: {details}")

            # System should be valid
            assert is_valid

            # 5. Export knowledge
            export_path = Path(temp_data_dir) / "test_export.gyro"
            export_knowledge(session_id, str(export_path))

            # Export file should exist
            assert export_path.exists()
            assert export_path.stat().st_size > 0

        finally:
            # 6. Clean shutdown
            shutdown_session(session_id)

    def test_engine_resonance_patterns(self, temp_data_dir):
        """Test that the engine produces consistent resonance patterns."""

        session_id = initialize_session()

        try:
            # Test same input produces same results
            text = "Test Pattern"

            result1 = process_text(session_id, text)

            # Reset to same state and test again
            shutdown_session(session_id)
            session_id = initialize_session()

            result2 = process_text(session_id, text)

            # Results should be deterministic
            assert result1 == result2

        finally:
            shutdown_session(session_id)

    def test_multiple_sessions(self, temp_data_dir):
        """Test multiple independent sessions."""

        session1 = initialize_session()
        session2 = initialize_session()

        try:
            # Sessions should be different
            assert session1 != session2

            # Process different text in each
            count1 = process_text(session1, "Session One")
            count2 = process_text(session2, "Session Two")

            # Both should work independently
            assert count1 >= 0
            assert count2 >= 0

        finally:
            shutdown_session(session1)
            shutdown_session(session2)

    def test_knowledge_session_separation(self, temp_data_dir):
        """Test knowledge/session separation works correctly."""

        # Create session with some learning
        session_id = initialize_session()

        try:
            # Process some text to create learning
            process_text(session_id, "Learning content for knowledge package")

            # Export the knowledge
            export_path = Path(temp_data_dir) / "knowledge_test.gyro"
            export_knowledge(session_id, str(export_path))

            # Shutdown session
            shutdown_session(session_id)

            # Knowledge file should persist
            assert export_path.exists()

            # TODO: Test import when implemented
            # new_session = import_knowledge(str(export_path))
            # assert new_session != session_id

        finally:
            try:
                shutdown_session(session_id)
            except:
                pass  # May already be shut down

    def test_error_handling(self, temp_data_dir):
        """Test system handles errors gracefully."""

        # Test invalid session operations
        with pytest.raises(Exception):  # Should be GyroSessionError
            process_text("invalid-session-id", "test")

        # Test invalid inputs
        session_id = initialize_session()

        try:
            # These should not crash the system
            with pytest.raises(Exception):
                process_byte_stream(session_id, [256])  # Invalid byte

            with pytest.raises(Exception):
                process_byte_stream(session_id, [-1])  # Invalid byte

            # System should still be functional
            result = process_text(session_id, "Recovery test")
            assert result >= 0

        finally:
            shutdown_session(session_id)

    def test_performance_characteristics(self, temp_data_dir):
        """Test system meets basic performance requirements."""
        import time

        session_id = initialize_session()

        try:
            # Test processing speed
            start_time = time.time()

            # Process 1000 bytes
            test_data = list(range(256)) * 4  # 1024 bytes
            resonances = process_byte_stream(session_id, test_data)

            end_time = time.time()
            duration = end_time - start_time

            print(f"Processed 1024 bytes in {duration:.4f}s")
            print(f"Throughput: {len(test_data)/duration:.1f} bytes/sec")
            print(f"Resonances: {resonances}")

            # Should process at reasonable speed (>100 bytes/sec)
            assert len(test_data) / duration > 100

        finally:
            shutdown_session(session_id)


def test_system_components_exist():
    """Test that all required system components can be imported."""

    # Test core components
    from core.gyro_core import GyroEngine, gyration_op
    from core.gyro_tag_parser import validate_tag, parse_tag
    from core.alignment_nav import NavigationLog
    from core.gyro_errors import GyroError, GyroTagError

    # Test extension manager
    from core.extension_manager import ExtensionManager

    # Test API
    import core.gyro_api as api

    # Test extensions exist
    from extensions.base import GyroExtension
    from extensions.ext_storage_manager import ext_StorageManager
    from extensions.ext_fork_manager import ext_ForkManager

    # All imports successful
    assert True


def test_tag_parser_functionality():
    """Test TAG parser works correctly."""
    from core.gyro_tag_parser import validate_tag, parse_tag

    # Valid TAGs
    assert validate_tag("current.gyrotensor_add")
    assert validate_tag("previous.gyrotensor_quant")
    assert validate_tag("next.gyrotensor_id.context")

    # Invalid TAGs
    assert not validate_tag("invalid.tag")
    assert not validate_tag("current.invalid_invariant")
    assert not validate_tag("invalid_temporal.gyrotensor_add")

    # Parse valid TAG
    parsed = parse_tag("current.gyrotensor_add")
    assert parsed["temporal"] == "current"
    assert parsed["invariant"] == "gyrotensor_add"

    # Parse with context
    parsed = parse_tag("previous.gyrotensor_com.context")
    assert parsed["temporal"] == "previous"
    assert parsed["invariant"] == "gyrotensor_com"
    assert parsed["context"] == "context"


def test_gyro_engine_standalone():
    """Test GyroEngine works in isolation."""
    from core.gyro_core import GyroEngine

    engine = GyroEngine()

    # Test initialization
    assert engine.phase == 0
    assert "id_0" in engine.gene
    assert "id_1" in engine.gene

    # Test cycle execution
    result = engine.execute_cycle(0x88)
    assert engine.phase == 1

    # Result should be None or tuple of ints
    if result is not None:
        assert len(result) == 2
        assert all(isinstance(op, int) for op in result)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_integration_complete.py -v -s
    pytest.main([__file__, "-v", "-s"])
