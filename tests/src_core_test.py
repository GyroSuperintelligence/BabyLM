"""
src_core_test.py - Core System Tests

Tests for the implemented core components:
- GyroEngine (Tier 3) - Pure navigation engine
- gyration_op - Core transformation primitive

This test file focuses on what we can test immediately without dependencies.
"""

import pytest
import torch
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only what we have implemented
from core.gyro_core import GyroEngine, gyration_op


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
    
    def test_gene_constant_correctness(self):
        """Test that the gene constant matches CORE-SPEC-02."""
        engine = GyroEngine()
        
        # Verify the exact pattern from CORE-SPEC-02
        expected_pattern = [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
        ]
        expected_tensor = torch.tensor(expected_pattern, dtype=torch.int8)
        
        # Both id_0 and id_1 should match the expected pattern
        assert torch.equal(engine.gene["id_0"], expected_tensor)
        assert torch.equal(engine.gene["id_1"], expected_tensor)
    
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
            (0xAB, 5),   # Random case
            (0x12, 10),  # Another random case
        ]
        
        for input_byte, phase in test_cases:
            engine.load_phase(phase)
            # Should return boolean without error
            result = engine._structural_resonance(input_byte)
            assert isinstance(result, bool)
        
        # Test invalid input bytes
        assert engine._structural_resonance(-1) == False
        assert engine._structural_resonance(256) == False
        assert engine._structural_resonance(1000) == False
    
    def test_structural_resonance_deterministic(self):
        """Test that structural resonance is deterministic."""
        engine = GyroEngine()
        
        # Same input and phase should always give same result
        test_cases = [(0x88, 5), (0xFF, 12), (0x00, 25)]
        
        for input_byte, phase in test_cases:
            engine.load_phase(phase)
            result1 = engine._structural_resonance(input_byte)
            result2 = engine._structural_resonance(input_byte)
            assert result1 == result2
    
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
        non_resonant_phases = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]
        for phase in non_resonant_phases[:6]:  # Test a few
            engine.load_phase(phase)
            result = engine._select_operator_codes()
            assert result is None
    
    def test_execute_cycle_phase_advancement(self):
        """Test that execute_cycle always advances phase correctly."""
        engine = GyroEngine()
        
        # Test normal advancement
        for expected_phase in range(1, 48):
            engine.execute_cycle(0x00)  # Input doesn't matter for phase advancement
            assert engine.phase == expected_phase
        
        # Test wraparound
        engine.execute_cycle(0x00)
        assert engine.phase == 0  # Should wrap back to 0
    
    def test_execute_cycle_with_resonance(self):
        """Test execute_cycle with different inputs to find resonance."""
        engine = GyroEngine()
        
        # Test at a CS boundary where we know operators will fire
        engine.load_phase(11)  # Will advance to 12 (CS boundary)
        
        # Try different inputs to see resonance behavior
        test_inputs = [0x00, 0x88, 0xFF, 0x77, 0xAA, 0x55]
        results = []
        
        for test_byte in test_inputs:
            engine.load_phase(11)  # Reset to same starting phase
            result = engine.execute_cycle(test_byte)
            results.append((test_byte, result))
            assert engine.phase == 12  # Phase should always advance
        
        # At least some inputs should produce different results
        unique_results = set(results)
        
        # Verify result format when not None
        for test_byte, result in results:
            if result is not None:
                assert len(result) == 2
                assert all(isinstance(op, int) for op in result)
                assert all(0 <= op <= 15 for op in result)  # 4-bit values
    
    def test_engine_independence(self):
        """Test that multiple engines are independent."""
        engine1 = GyroEngine()
        engine2 = GyroEngine()
        
        # Engines should start identically
        assert engine1.phase == engine2.phase == 0
        assert torch.equal(engine1.gene["id_0"], engine2.gene["id_0"])
        
        # Modifying one shouldn't affect the other
        engine1.load_phase(10)
        engine2.load_phase(20)
        
        assert engine1.phase == 10
        assert engine2.phase == 20
        
        # Execute cycles independently
        result1 = engine1.execute_cycle(0x88)
        result2 = engine2.execute_cycle(0x77)
        
        assert engine1.phase == 11
        assert engine2.phase == 21


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
    
    def test_gyration_reversibility(self):
        """Test that some operations are reversible."""
        tensor = torch.ones((4, 2, 3, 2), dtype=torch.int8)
        
        # Identity should be reversible with itself
        result = gyration_op(tensor, 0)
        result = gyration_op(result, 0, clone=False)
        assert torch.equal(result, tensor)
        
        # Inverse should be reversible with itself
        result = gyration_op(tensor, 1)
        result = gyration_op(result, 1, clone=False)
        assert torch.equal(result, tensor)


class TestPerformanceCharacteristics:
    """Test performance guarantees from CORE-SPEC-07."""
    
    def test_engine_performance(self):
        """Test that engine meets performance guarantees."""
        engine = GyroEngine()
        
        # Test that execute_cycle is fast
        start_time = time.perf_counter()
        for i in range(1000):  # 1000 operations
            engine.execute_cycle(i % 256)  # Vary input
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_op = total_time / 1000
        
        # Should be much faster than 1ms per operation
        assert avg_time_per_op < 0.001, f"Operation too slow: {avg_time_per_op:.6f}s"
    
    def test_memory_footprint(self):
        """Test that engine has minimal memory footprint."""
        engine = GyroEngine()
        
        # Gene should be exactly 96 elements (2 * 4 * 2 * 3 * 2)
        gene_elements = engine.gene["id_0"].numel() + engine.gene["id_1"].numel()
        assert gene_elements == 96, f"Gene should have 96 elements, got {gene_elements}"
        
        # Each element is 1 byte (int8), so total gene size is 96 bytes
        gene_bytes = gene_elements * 1  # int8 = 1 byte each
        assert gene_bytes == 96, f"Gene should be 96 bytes, got {gene_bytes}"
        
        # Phase is a single integer
        assert isinstance(engine.phase, int)
        assert 0 <= engine.phase < 48


def test_integration_example():
    """Test a complete example of the system working together."""
    engine = GyroEngine()
    
    # Simulate processing "Hello" byte by byte
    hello_bytes = [ord(c) for c in "Hello"]
    navigation_events = []
    phases_visited = []
    
    for i, byte in enumerate(hello_bytes):
        initial_phase = engine.phase
        result = engine.execute_cycle(byte)
        final_phase = engine.phase
        
        phases_visited.append((initial_phase, final_phase))
        
        if result:
            navigation_events.append((byte, result))
        
    # Should have advanced through 5 phases
    assert engine.phase == 5
    
    # Should have visited phases 0→1, 1→2, 2→3, 3→4, 4→5
    expected_phases = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    assert phases_visited == expected_phases
    
    # Verify event format
    for byte, event in navigation_events:
        assert len(event) == 2
        assert all(isinstance(op, int) for op in event)
        assert all(0 <= op <= 15 for op in event)  # 4-bit packed values


def test_comprehensive_phase_cycle():
    """Test a complete 48-phase cycle to verify all boundaries."""
    engine = GyroEngine()
    
    operator_counts = {
        'stable': 0,
        'unstable': 0, 
        'neutral': 0,
        'none': 0
    }
    
    # Go through complete cycle
    for phase in range(48):
        engine.load_phase(phase)
        result = engine._select_operator_codes()
        
        if result:
            op_0, op_1 = result
            # Classify the operator type based on the codes
            # FIX: Check for neutral *before* unstable
            if op_0 == 0 and op_1 == 3:  # Identity + Inverse
                operator_counts['stable'] += 1
            elif op_0 == 6 and op_1 == 7:  # Both Backward
                operator_counts['neutral'] += 1
            elif op_0 in [4, 6] and op_1 in [5, 7]:  # Forward/Backward
                operator_counts['unstable'] += 1
        else:
            operator_counts['none'] += 1
    
    # Verify expected counts based on CORE-SPEC-08
    assert operator_counts['stable'] == 4    # CS boundaries: 0,12,24,36
    assert operator_counts['unstable'] == 8  # UNA/ONA: 3,9,15,21,27,33,39,45
    assert operator_counts['neutral'] == 4   # Nesting: 6,18,30,42
    assert operator_counts['none'] == 32     # Remaining phases


def test_structural_resonance_coverage():
    """Test structural resonance across different tensor positions."""
    engine = GyroEngine()
    
    # Test resonance at different phases to cover different tensor positions
    resonance_results = {}
    
    for phase in range(0, 48, 6):  # Test every 6th phase
        engine.load_phase(phase)
        
        # Test with different bit patterns
        test_patterns = [0x00, 0x0F, 0xF0, 0xFF, 0x88, 0x77, 0xAA, 0x55]
        phase_results = []
        
        for pattern in test_patterns:
            resonates = engine._structural_resonance(pattern)
            phase_results.append((pattern, resonates))
        
        resonance_results[phase] = phase_results
    
    # Should have some resonance somewhere
    total_resonances = sum(sum(1 for _, r in results if r) 
                          for results in resonance_results.values())
    assert total_resonances > 0, "No structural resonance found in any test"


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    engine = GyroEngine()
    
    # Test at phase boundaries
    boundary_phases = [0, 47]  # Start and end of cycle
    
    for phase in boundary_phases:
        # Should handle all input bytes without error
        for test_byte in [0, 1, 127, 128, 254, 255]:
            # FIX: Reset the phase for each independent test
            engine.load_phase(phase)
            
            result = engine.execute_cycle(test_byte)
            # Should always advance phase
            expected_phase = (phase + 1) % 48
            assert engine.phase == expected_phase
    
    # Test rapid phase changes
    engine.load_phase(0)
    for _ in range(100):  # More than one complete cycle
        engine.execute_cycle(0x88)
    
    # Should be at phase 4 (100 % 48 = 4)
    assert engine.phase == 4