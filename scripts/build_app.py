#!/usr/bin/env python3
"""
Build GyroSI Baby ML app with custom icon and bundle configuration
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Sized


def build_app():
    """Build the Flet app with custom configuration"""

    # Paths
    project_root = Path(__file__).parent.parent
    main_script = project_root / "src" / "main.py"
    icon_path = project_root / "src" / "frontend" / "assets" / "icons" / "mingcute--baby-fill.png"
    assets_path = project_root / "src" / "frontend" / "assets"

    # Verify files exist
    if not main_script.exists():
        print(f"✗ Main script not found: {main_script}")
        return False

    if not icon_path.exists():
        print(f"✗ Icon not found: {icon_path}")
        return False

    # Build command
    cmd = [
        "flet",
        "pack",
        str(main_script),
        "--icon",
        str(icon_path),
        "--add-data",
        f"{assets_path}:assets",
        "--product-name",
        "GyroSI Baby ML",
        "--product-version",
        "1.0.0",
        "--copyright",
        "© 2024 GyroSI Baby ML",
        "--bundle-id",
        "com.gyrosi.babylm",
        "--name",
        "GyroSI Baby ML",
    ]

    print("🚀 Building GyroSI Baby ML app...")
    print(f"📁 Main script: {main_script}")
    print(f"🎨 Icon: {icon_path}")
    print(f"📦 Assets: {assets_path}")
    print()

    try:
        # Run the build command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Build completed successfully!")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ 'flet' command not found. Make sure Flet is installed:")
        print("pip install flet")
        return False


def main():
    """Main entry point"""
    if build_app():
        print("\n🎉 GyroSI Baby ML app built successfully!")
        print("📱 You can find the app in the 'dist' directory")
    else:
        print("\n❌ Build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
