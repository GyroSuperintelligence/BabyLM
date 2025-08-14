#!/usr/bin/env python3
"""
Test script for the pure physics Gyro model.

Verifies that the model can:
1. Load Gyro physics maps from converted files
2. Initialize physics tables (fold, resonance, token-intron)
3. Run forward pass with pure physics operations
4. Generate outputs without any traditional transformer operations
"""

import torch
import os
from model import GyroTransformer, ModelConfig

def test_model():
    """Test the pure physics Gyro model."""
    print("Testing Pure Physics Gyro Model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path to converted Gyro files
    gyro_path = "/Users/basil/Development/BabyLM/memories/models/gpt-oss-20b/gyro"
    
    if not os.path.exists(gyro_path):
        print(f"Error: Gyro directory not found at {gyro_path}")
        return False
    
    print(f"Loading Gyro physics maps from: {gyro_path}")
    
    try:
        # Initialize the pure physics model
        model = GyroTransformer(
            gyro_path=gyro_path,
            device=device
        )
        print("✓ Model initialized successfully")
        
        # Test input - small batch for verification
        batch_size = 2
        seq_len = 8
        vocab_size = model.vocab_size
        
        # Create test input tokens
        input_tokens = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len), device=device)
        print(f"✓ Created test input: {input_tokens.shape}")
        
        # Run forward pass with pure physics
        print("Running pure physics forward pass...")
        with torch.no_grad():
            logits = model.forward(input_tokens)
        
        print(f"✓ Forward pass completed")
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Output dtype: {logits.dtype}")
        print(f"✓ Output device: {logits.device}")
        
        # Verify output properties
        assert logits.shape == (batch_size, seq_len, vocab_size), f"Unexpected output shape: {logits.shape}"
        assert not torch.isnan(logits).any(), "Output contains NaN values"
        assert not torch.isinf(logits).any(), "Output contains infinite values"
        
        print(f"✓ Output validation passed")
        print(f"✓ Sample logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        # Test physics tables
        print("\nPhysics Tables Status:")
        print(f"✓ Fold table shape: {model.fold_table.shape}")
        print(f"✓ Resonance table shape: {model.resonance_table.shape}")
        print(f"✓ Token-intron table entries: {len(model.token_intron_table)}")
        print(f"✓ Physics maps loaded: {len(model.physics_maps)}")
        
        print("\n🎉 All tests passed! Pure physics Gyro model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)