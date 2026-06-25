#!/usr/bin/env python3
"""
Example script showing programmatic usage of the crater annotation tool.
This demonstrates how to create annotations programmatically and export them.
"""

import torch
import numpy as np
from pathlib import Path
from crater_annotator import Ellipse
import pandas as pd
from PIL import Image


def create_sample_annotations():
    """Create sample crater annotations."""
    # Example: Define some crater ellipses
    craters = [
        Ellipse(center_x=100, center_y=150, radius_x=30, radius_y=25, label="small_crater"),
        Ellipse(center_x=300, center_y=200, radius_x=60, radius_y=55, label="medium_crater"),
        Ellipse(center_x=450, center_y=100, radius_x=80, radius_y=70, label="large_crater"),
    ]
    
    return craters


def save_to_csv(craters, output_path):
    """Save crater annotations to CSV format."""
    data = [crater.to_dict() for crater in craters]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(craters)} craters to {output_path}")


def create_mask_from_ellipses(craters, image_height, image_width, output_path):
    """
    Create a label PNG from ellipse annotations.

    - Shape: (H, W)
    - Values: 0 = background, 1 = crater 1, 2 = crater 2, ...
    - Format: single-channel PNG
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    y_coords, x_coords = np.meshgrid(
        np.arange(image_height), np.arange(image_width), indexing='ij'
    )

    for i, crater in enumerate(craters, start=1):
        ellipse_eq = (
            ((x_coords - crater.center_x) ** 2 / (crater.radius_x ** 2)) +
            ((y_coords - crater.center_y) ** 2 / (crater.radius_y ** 2))
        )
        mask[ellipse_eq <= 1] = i

    Image.fromarray(mask, mode='L').save(output_path)
    print(f"✓ Saved mask with shape {mask.shape} to {output_path} ({len(craters)} craters)")

    return mask


def load_and_verify_mask(mask_path):
    """Load and verify a mask file."""
    mask = np.array(Image.open(mask_path))
    print(mask.shape)
    print(f"\nMask verification:")
    print(f"  Shape: {mask.shape}")
    print(f"  Number of craters: {len(np.unique(mask[mask > 0]))}")
    print(f"  Image dimensions: {mask.shape[0]} x {mask.shape[1]}")
    print(f"  Data type: {mask.dtype}")
    
    # Calculate crater areas
    for i in range(1, len(np.unique(mask[mask > 0])) + 1):
        area = (mask == i).sum().item()
        print(f"  Crater {i} area: {area} pixels")
    
    return mask


def main():
    """Main example workflow."""
    print("=" * 60)
    print("Crater Annotation Programmatic Usage Example")
    print("=" * 60)
    
    # Create sample annotations
    print("\n1. Creating sample crater annotations...")
    craters = create_sample_annotations()
    print(f"✓ Created {len(craters)} crater ellipses")
    
    # Save to CSV
    print("\n2. Saving to CSV format...")
    csv_path = "example_annotations.csv"
    save_to_csv(craters, csv_path)
    
    # Create and save mask
    print("\n3. Creating PyTorch mask...")
    image_height, image_width = 512, 512  # Example image dimensions
    mask_path = "example_mask.png"
    mask = create_mask_from_ellipses(craters, image_height, image_width, mask_path)
    
    # Verify the mask
    print("\n4. Verifying saved mask...")
    loaded_masks = load_and_verify_mask(mask_path)
    
    # Show how to use in training
    print("\n5. Usage in training pipeline:")
    print("   # Load mask")
    print(f"   sparse_mask = torch.load('{mask_path}')")
    print("   masks = sparse_mask.to_dense().to(torch.uint8)")
    print("   ")
    print("   # Convert to single binary mask (all craters combined)")
    print("   binary_mask = (masks.sum(dim=0) > 0).float()")
    print("   ")
    print("   # Or work with individual crater masks")
    print("   for i, crater_mask in enumerate(masks):")
    print("       # Process each crater separately")
    print("       pass")
    
    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - {csv_path}")
    print(f"  - {mask_path}")
    print("\nTo use the GUI application:")
    print("  python crater_annotator.py")


if __name__ == "__main__":
    main()
