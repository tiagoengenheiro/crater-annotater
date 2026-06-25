#!/usr/bin/env python3
"""Quick test to verify the crater annotator can be imported."""

import sys

try:
    # Test imports
    print("Testing imports...")
    from crater_annotator import Ellipse, ImageCanvas, CraterAnnotatorApp
    print("✓ All imports successful")
    
    # Test Ellipse class
    print("\nTesting Ellipse class...")
    ellipse = Ellipse(100, 100, 50, 30)
    assert ellipse.center_x == 100
    assert ellipse.center_y == 100
    assert ellipse.radius_x == 50
    assert ellipse.radius_y == 30
    print("✓ Ellipse class works")
    
    # Test to_dict/from_dict
    print("\nTesting serialization...")
    data = ellipse.to_dict()
    ellipse2 = Ellipse.from_dict(data)
    assert ellipse2.center_x == ellipse.center_x
    assert ellipse2.center_y == ellipse.center_y
    print("✓ Serialization works")
    
    print("\n✅ All tests passed! The application is ready to use.")
    print("\nTo start the application, run:")
    print("  python crater_annotator.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
