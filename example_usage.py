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
    Create PyTorch mask from ellipse annotations.
    
    This matches the format used by the training pipeline:
    - Shape: (N, H, W) where N is number of craters
    - Values: Binary (0 or 1)
    - Format: Sparse tensor saved as .pt file
    """
    sparse_masks = []
    
    for crater in craters:
        # Create dense mask for this crater
        mask_dense = torch.zeros((image_height, image_width), dtype=torch.uint8)
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height), 
            torch.arange(image_width), 
            indexing='ij'
        )
        
        # Compute ellipse equation
        ellipse_eq = (
            ((x_coords - crater.center_x) ** 2 / (crater.radius_x ** 2)) +
            ((y_coords - crater.center_y) ** 2 / (crater.radius_y ** 2))
        )
        
        # Set pixels inside ellipse to 1
        mask_dense[ellipse_eq <= 1] = 1
        
        # Convert to sparse
        sparse_masks.append(mask_dense.to_sparse())
    
    # Stack all masks
    dense_stacked = torch.stack([m.to_dense() for m in sparse_masks], dim=0)
    mask = dense_stacked.to_sparse()
    
    # Save
    torch.save(mask, output_path)
    print(f"✓ Saved mask with shape {dense_stacked.shape} to {output_path}")
    
    return mask


def load_and_verify_mask(mask_path):
    """Load and verify a mask file."""
    sparse_mask = torch.load(mask_path)
    masks = sparse_mask.to_dense().to(torch.uint8)
    
    print(f"\nMask verification:")
    print(f"  Shape: {masks.shape}")
    print(f"  Number of craters: {masks.shape[0]}")
    print(f"  Image dimensions: {masks.shape[1]} x {masks.shape[2]}")
    print(f"  Data type: {masks.dtype}")
    
    # Calculate crater areas
    for i in range(masks.shape[0]):
        area = masks[i].sum().item()
        print(f"  Crater {i+1} area: {area} pixels")
    
    return masks


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
    mask_path = "example_mask.pt"
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
