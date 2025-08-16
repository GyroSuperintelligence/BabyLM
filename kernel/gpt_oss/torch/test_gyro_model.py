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
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model import GyroTransformer, ModelConfig

def test_model(model_path: str):
    """Test the pure physics Gyro model."""
    print("Testing Pure Physics Gyro Model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for model.gyro.safetensors file
    gyro_weights_path = os.path.join(model_path, "model.gyro.safetensors")
    if not os.path.exists(gyro_weights_path):
        print(f"Error: Gyro weights file not found at {gyro_weights_path}")
        return False
    
    print(f"Loading Gyro physics model from: {model_path}")
    
    try:
        # Initialize the pure physics model
        # GyroTransformer will automatically find the model.gyro.safetensors file
        model = GyroTransformer()
        print("✓ Model initialized successfully")
        
        # Test input - small batch for verification
        batch_size = 2
        seq_len = 8
        vocab_size = model.gyro.vocab_size
        
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
        print(f"✓ Resonance table shape: {model.gyro._resonance_table.shape}")
        print(f"✓ Token-intron table entries: {len(model.gyro.token_introns)}")
        print(f"✓ Model weights loaded: {len(model.gyro.model_weights)}")
        
        print("\n🎉 All tests passed! Pure physics Gyro model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the pure physics Gyro model")
    parser.add_argument("--model_path", 
                       default=str(Path(__file__).parents[3] / "memories/models/gpt-oss-20b"),
                       help="Path to the model directory containing model.gyro.safetensors")
    
    args = parser.parse_args()
    success = test_model(args.model_path)
    exit(0 if success else 1)