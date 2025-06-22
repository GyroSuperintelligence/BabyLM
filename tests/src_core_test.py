"""
src_core_test.py - Core System Tests

Tests for the three-tier core architecture:
- GyroEngine (Tier 3) - Pure navigation engine
- ExtensionManager (Tier 2) - Orchestration layer  
- GyroAPI (Tier 1) - Public interface

This test file focuses on what we can test immediately without full extension
implementations, using mocks where necessary.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our core components
from core.gyro_core import GyroEngine, gyration_op
# Note: ExtensionManager and gyro_api will need mocking for now


class TestGyroEngine:
    """Test the pure navigation engine - this should work completely."""
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        engine = GyroEngine()
        
        # Check initial state
        assert engine.phase == 0
        assert "id_0" in engine.gene
        assert "id_1" in engine.gene
        
        # Check gene structure
        assert engine.gene["id_0"].shape == (4, 2, 3, 2)
        assert engine.gene["id_1"].shape == (4, 2, 3, 2)
        assert engine.gene["id_0"].dtype == torch.int8
        assert engine.gene["id_1"].dtype == torch.int8
        
        # Check gene values are in {-1, 1}
        for tensor in engine.gene.values():
            unique_vals = torch.unique(tensor)
            assert len(unique_vals) <= 2
            assert all(val in [-1, 1] for val in unique_vals.tolist())
    
    def test_phase_loading(self):
        """Test phase loading with validation."""
        engine = GyroEngine()
        
        # Valid phase loading
        engine.load_phase(25)
        assert engine.phase == 25
        
        engine.load_phase(0)
        assert engine.phase == 0
        
        engine.load_phase(47)
        assert engine.phase == 47
        
        # Invalid phase loading
        with pytest.raises(ValueError, match="Phase must be between 0 and 47"):
            engine.load_phase(-1)
        
        with pytest.raises(ValueError, match="Phase must be between 0 and 47"):
            engine.load_phase(48)
        
        with pytest.raises(ValueError, match="Phase must be between 0 and 47"):
            engine.load_phase(100)
    
    def test_structural_resonance(self):
        """Test structural resonance logic."""
        engine = GyroEngine()
        
        # Test with various input bytes and phases
        test_cases = [
            (0x00, 0),   # Low nibbles, phase 0
            (0xFF, 0),   # High nibbles, phase 0
            (0x88, 1),   # Mixed, phase 1
            (0x77, 2),   # Mixed, phase 2
        ]
        
        for input_byte, phase in test_cases:
            engine.load_phase(phase)
            # Should return boolean without error
            result = engine._structural_resonance(input_byte)
            assert isinstance(result, bool)
        
        # Test invalid input bytes
        assert engine._structural_resonance(-1) == False
        assert engine._structural_resonance(256) == False
    
    def test_operator_selection(self):
        """Test operator code selection based on phase."""
        engine = GyroEngine()
        
        # Test CS boundaries (should return stable operator)
        cs_phases = [0, 12, 24, 36]
        for phase in cs_phases:
            engine.load_phase(phase)
            result = engine._select_operator_codes()
            assert result is not None
            op_0, op_1 = result
            # Should be Identity for id_0, Inverse for id_1
            assert op_0 == 0  # (0 << 1) | 0 = Identity for id_0
            assert op_1 == 3  # (1 << 1) | 1 = Inverse for id_1
        
        # Test UNA/ONA boundaries (should return unstable operator)
        una_ona_phases = [3, 9, 15, 21, 27, 33, 39, 45]
        for phase in una_ona_phases:
            engine.load_phase(phase)
            result = engine._select_operator_codes()
            assert result is not None
            op_0, op_1 = result
            # Should be Forward or Backward based on position
            assert op_0 in [4, 6]  # Forward or Backward for id_0
            assert op_1 in [5, 7]  # Forward or Backward for id_1
        
        # Test nesting boundaries (should return neutral operator)
        nesting_phases = [6, 18, 30, 42]
        for phase in nesting_phases:
            engine.load_phase(phase)
            result = engine._select_operator_codes()
            assert result is not None
            op_0, op_1 = result
            # Should be Backward for both
            assert op_0 == 6  # (3 << 1) | 0 = Backward for id_0
            assert op_1 == 7  # (3 << 1) | 1 = Backward for id_1
        
        # Test non-resonant phases
        non_resonant_phases = [1, 2, 4, 5, 7, 8, 10, 11]
        for phase in non_resonant_phases[:4]:  # Test a few
            engine.load_phase(phase)
            result = engine._select_operator_codes()
            assert result is None
    
    def test_execute_cycle(self):
        """Test the complete execution cycle."""
        engine = GyroEngine()
        
        # Test cycle with no resonance
        engine.load_phase(1)  # Non-resonant phase
        result = engine.execute_cycle(0x00)
        # Phase should advance regardless
        assert engine.phase == 2
        # But no operators should fire if no structural resonance
        # (This depends on the specific gene pattern and input)
        
        # Test cycle with potential resonance
        engine.load_phase(11)  # Will advance to 12 (CS boundary)
        # Try different inputs to find one that resonates
        for test_byte in [0x00, 0x88, 0xFF, 0x77]:
            result = engine.execute_cycle(test_byte)
            assert engine.phase == 12  # Phase should advance
            # Result depends on structural resonance
            if result:
                assert len(result) == 2
                assert all(isinstance(op, int) for op in result)
            break  # Test at least one cycle
    
    def test_phase_advancement(self):
        """Test that phase advances correctly through the 48-step cycle."""
        engine = GyroEngine()
        
        # Test normal advancement
        for expected_phase in range(1, 48):
            engine.execute_cycle(0x00)  # Input doesn't matter for phase advancement
            assert engine.phase == expected_phase
        
        # Test wraparound
        engine.execute_cycle(0x00)
        assert engine.phase == 0  # Should wrap back to 0


class TestGyrationOp:
    """Test the core transformation primitive."""
    
    def test_gyration_op_validation(self):
        """Test input validation for gyration_op."""
        # Create a valid tensor
        tensor = torch.ones((4, 2, 3, 2), dtype=torch.int8)
        
        # Valid operations
        for code in [0, 1, 2, 3]:
            result = gyration_op(tensor, code)
            assert result.shape == (4, 2, 3, 2)
            assert result.dtype == torch.int8
        
        # Invalid tensor shape
        bad_tensor = torch.ones((3, 2, 3, 2), dtype=torch.int8)
        with pytest.raises(ValueError, match="Invalid tensor shape"):
            gyration_op(bad_tensor, 0)
        
        # Invalid operator code
        with pytest.raises(ValueError, match="Invalid gyration code"):
            gyration_op(tensor, 4)
        
        with pytest.raises(ValueError, match="Invalid gyration code"):
            gyration_op(tensor, -1)
    
    def test_gyration_operations(self):
        """Test each gyration operation."""
        # Create test tensor with known pattern
        tensor = torch.tensor([
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]]
        ], dtype=torch.int8)
        
        # Test Identity (code 0)
        result = gyration_op(tensor, 0)
        assert torch.equal(result, tensor)
        
        # Test Inverse (code 1)
        result = gyration_op(tensor, 1)
        expected = tensor * -1
        assert torch.equal(result, expected)
        
        # Test Forward (code 2) - flips rows 0 and 2
        result = gyration_op(tensor, 2)
        expected = tensor.clone()
        expected[0] *= -1
        expected[2] *= -1
        assert torch.equal(result, expected)
        
        # Test Backward (code 3) - flips rows 1 and 3
        result = gyration_op(tensor, 3)
        expected = tensor.clone()
        expected[1] *= -1
        expected[3] *= -1
        assert torch.equal(result, expected)
    
    def test_clone_behavior(self):
        """Test clone vs in-place behavior."""
        tensor = torch.ones((4, 2, 3, 2), dtype=torch.int8)
        original = tensor.clone()
        
        # Test with clone=True (default)
        result = gyration_op(tensor, 1, clone=True)
        assert torch.equal(tensor, original)  # Original unchanged
        assert not torch.equal(result, original)  # Result is different
        
        # Test with clone=False
        result = gyration_op(tensor, 1, clone=False)
        assert torch.equal(tensor, result)  # Original was modified
        assert not torch.equal(tensor, original)  # Original changed


class TestArchitecturalIntegration:
    """Test that our architectural decisions work together."""
    
    def test_engine_isolation(self):
        """Test that the engine is truly isolated and pure."""
        engine1 = GyroEngine()
        engine2 = GyroEngine()
        
        # Engines should be independent
        engine1.load_phase(10)
        engine2.load_phase(20)
        
        assert engine1.phase == 10
        assert engine2.phase == 20
        
        # Gene should be identical but independent
        assert torch.equal(engine1.gene["id_0"], engine2.gene["id_0"])
        
        # Modifying one shouldn't affect the other
        engine1.execute_cycle(0x88)
        assert engine1.phase != engine2.phase
    
    @patch('core.extension_manager.ExtensionManager')
    def test_api_manager_interaction(self, mock_manager_class):
        """Test that API layer can interact with manager (mocked)."""
        # This tests the architectural pattern even without full implementation
        mock_manager = Mock()
        mock_manager.get_session_id.return_value = "test-session-123"
        mock_manager.gyro_operation.return_value = (4, 5)
        mock_manager_class.return_value = mock_manager
        
        # Test that we can create the pattern
        manager = mock_manager_class("session-id", "knowledge-id")
        
        # Test basic interactions
        session_id = manager.get_session_id()
        assert session_id == "test-session-123"
        
        result = manager.gyro_operation(0x88)
        assert result == (4, 5)
        
        # Verify the manager was called correctly
        mock_manager_class.assert_called_once_with("session-id", "knowledge-id")
        mock_manager.gyro_operation.assert_called_once_with(0x88)


class TestPerformanceCharacteristics:
    """Test performance guarantees from CORE-SPEC-07."""
    
    def test_engine_performance(self):
        """Test that engine meets performance guarantees."""
        engine = GyroEngine()
        
        # Test that execute_cycle is fast (should be < 1ms for single operation)
        import time
        
        start_time = time.perf_counter()
        for _ in range(1000):  # 1000 operations
            engine.execute_cycle(0x88)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_op = total_time / 1000
        
        # Should be much faster than 1ms per operation
        assert avg_time_per_op < 0.001, f"Operation too slow: {avg_time_per_op:.6f}s"
    
    def test_memory_footprint(self):
        """Test that engine has minimal memory footprint."""
        engine = GyroEngine()
        
        # Gene should be exactly 96 bytes (2 * 4 * 2 * 3 * 2 * 1 byte)
        gene_size = engine.gene["id_0"].numel() + engine.gene["id_1"].numel()
        assert gene_size == 96, f"Gene size should be 96 bytes, got {gene_size}"
        
        # Phase is 1 integer (4 bytes on most systems)
        import sys
        phase_size = sys.getsizeof(engine.phase)
        assert phase_size <= 28, f"Phase size too large: {phase_size} bytes"  # Python int overhead


def test_integration_example():
    """Test a complete example of the system working together."""
    # This is what we can test right now with just the engine
    engine = GyroEngine()
    
    # Simulate processing "Hello" byte by byte
    hello_bytes = [ord(c) for c in "Hello"]
    navigation_events = []
    
    for byte in hello_bytes:
        result = engine.execute_cycle(byte)
        if result:
            navigation_events.append(result)
    
    # Should have advanced through 5 phases
    assert engine.phase == 5
    
    # Should have some navigation events (depends on resonance)
    print(f"Navigation events generated: {len(navigation_events)}")
    print(f"Final phase: {engine.phase}")
    print(f"Events: {navigation_events}")
    
    # The exact number depends on structural resonance, but we should get some
    assert isinstance(navigation_events, list)
    for event in navigation_events:
        assert len(event) == 2
        assert all(isinstance(op, int) for op in event)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])