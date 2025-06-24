#!/usr/bin/env python3
"""
Convert SVG icon to PNG for app bundle icon
"""

import os
import sys
from pathlib import Path
from cairosvg import svg2png


def convert_svg_to_png(svg_path: str, png_path: str, size: int = 512):
    """Convert SVG to PNG with specified size"""
    try:
        # Read SVG file
        with open(svg_path, "rb") as svg_file:
            svg_data = svg_file.read()

        # Convert to PNG
        png_data = svg2png(
            bytestring=svg_data,
            output_width=size,
            output_height=size,
            background_color="transparent",
        )

        # Write PNG file
        with open(png_path, "wb") as png_file:
            if isinstance(png_data, bytes):
                png_file.write(png_data)
            else:
                raise TypeError("PNG data is not bytes")

        print(f"✓ Converted {svg_path} to {png_path} ({size}x{size})")
        return True

    except Exception as e:
        print(f"✗ Error converting SVG: {e}")
        return False


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    svg_path = project_root / "src" / "frontend" / "assets" / "icons" / "mingcute--baby-fill.svg"
    png_path = project_root / "src" / "frontend" / "assets" / "icons" / "app_icon.png"

    # Ensure output directory exists
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert SVG to PNG
    if convert_svg_to_png(str(svg_path), str(png_path), 512):
        print(f"✓ App icon ready: {png_path}")
        print("\nTo use this icon when packaging:")
        print(f"flet pack src/main.py --icon {png_path}")
    else:
        print("✗ Failed to create app icon")
        sys.exit(1)


if __name__ == "__main__":
    main()
