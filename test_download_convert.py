#!/usr/bin/env python3
"""
Test script to download and convert gpt-oss-20b model with new structure.
"""

import os
import sys
from pathlib import Path

# Add kernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kernel"))

def test_download():
    """Test downloading the model."""
    print("=== Testing Model Download ===")
    
    from codecs.gyrowt import download_model
    
    model_dir = "memories/models/gpt-oss-20b"
    
    try:
        download_model("openai/gpt-oss-20b", model_dir)
        
        # Verify files exist
        config_path = Path(model_dir) / "config.json"
        index_path = Path(model_dir) / "model.safetensors.index.json"
        
        if config_path.exists():
            print("✓ config.json downloaded")
        else:
            print("✗ config.json missing")
            
        if index_path.exists():
            print("✓ model.safetensors.index.json downloaded")
            
            # Check for actual model files
            import json
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            model_files = list(set(index_data.get("weight_map", {}).values()))
            print(f"✓ Found {len(model_files)} model files: {model_files}")
            
            for model_file in model_files:
                file_path = Path(model_dir) / model_file
                if file_path.exists():
                    print(f"✓ {model_file} exists")
                else:
                    print(f"✗ {model_file} missing")
        else:
            print("✗ model.safetensors.index.json missing")
            
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def test_conversion():
    """Test converting the model to gyro format."""
    print("\n=== Testing Model Conversion ===")
    
    from codecs.gyrowt import convert_checkpoint_dir_to_gyro
    
    model_dir = "memories/models/gpt-oss-20b"
    
    try:
        output_dir = convert_checkpoint_dir_to_gyro(model_dir, "gyro")
        print(f"✓ Conversion completed to: {output_dir}")
        
        # Check if gyro files were created
        gyro_dir = Path(output_dir)
        if gyro_dir.exists():
            gyro_files = list(gyro_dir.glob("*.safetensors"))
            print(f"✓ Created {len(gyro_files)} gyro files")
            
            # Show a few examples
            for i, gyro_file in enumerate(gyro_files[:3]):
                print(f"  - {gyro_file.name}")
            if len(gyro_files) > 3:
                print(f"  ... and {len(gyro_files) - 3} more")
        else:
            print("✗ Gyro directory not created")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("Testing new download and conversion process...")
    
    download_ok = test_download()
    if not download_ok:
        print("\n❌ Download test failed")
        return 1
        
    convert_ok = test_conversion()
    if not convert_ok:
        print("\n❌ Conversion test failed")
        return 1
        
    print("\n🎉 All tests passed!")
    print("\nTo download the model manually, run:")
    print("  huggingface-cli download openai/gpt-oss-20b --local-dir memories/models/gpt-oss-20b --exclude 'original/*' --exclude 'metal/*' --resume-download")
    print("\nTo convert to gyro format, run:")
    print("  python -m kernel.codecs.gyrowt --input memories/models/gpt-oss-20b --output gyro")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())