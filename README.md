# Mars Crater Annotation Tool

A desktop application for annotating craters on Mars images using ellipses. This tool allows you to create, edit, and export crater annotations that are compatible with PyTorch-based training pipelines.

## Features

- **Interactive Ellipse Drawing**: Click and drag to draw ellipses around craters
- **Edit Annotations**: Move, resize, and delete ellipses with intuitive controls
- **Precise Adjustments**: Fine-tune ellipse parameters using numerical input boxes
- **Multiple Export Formats**:
  - CSV files with ellipse parameters (xc, yc, rx, ry)
  - PyTorch tensor masks (.png) compatible with training pipelines
- **Import Existing Annotations**: Load previously saved CSV or JSON annotations
- **Real-time Preview**: See annotations overlaid on the image as you work

## Installation

### 1. Clone or navigate to the repository

```bash
cd /home/isradmin/Documents/mars/crater-annotater
```

### 2. Create and activate virtual environment (if not already done)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
python crater_annotator.py
```

### Workflow

1. **Load Image**: Click "Load Image" and select a Mars surface image (PNG, JPG, TIFF)

2. **Draw Ellipses**:
   - Click and drag on the image to draw an ellipse
   - Press `Ctrl` while clicking to force drawing a new ellipse (when clicking near existing ones)

3. **Edit Ellipses**:
   - **Select**: Click on an ellipse to select it (turns yellow)
   - **Move**: Click and drag a selected ellipse to reposition it
   - **Resize**: Use the property spinboxes on the right panel for precise adjustments
   - **Delete**: Right-click on an ellipse or select it and press `Delete` key

4. **Save Your Work**:
   - **Save Annotations (CSV)**: Export ellipse parameters to CSV format
   - **Export as Mask (.png)**: Create PNG masks for training

### Keyboard Shortcuts

- `Ctrl + Click & Drag`: Force draw new ellipse
- `Delete`: Remove selected ellipse
- `Esc`: Cancel current drawing operation
- `Right-Click`: Delete ellipse at cursor

### Controls Summary

| Action | Method |
|--------|--------|
| Draw ellipse | Click & drag on empty space |
| Select ellipse | Click on ellipse |
| Move ellipse | Click & drag selected ellipse |
| Delete ellipse | Right-click or select + Delete key |
| Adjust size/position | Use property spinboxes (right panel) |
| Clear all | Click "Clear All Ellipses" button |

## Export Formats

### 1. CSV Format

Saves ellipse parameters in a CSV file with columns:
- `xc`: X-coordinate of ellipse center
- `yc`: Y-coordinate of ellipse center
- `rx`: X-radius (semi-major axis)
- `ry`: Y-radius (semi-minor axis)
- `rotation`: Rotation angle (currently always 0)
- `label`: Annotation label (default: "crater")

### 2. PNG Mask Format

- 0 represents the background
- 1 represents pixels from crater 1, 2 from crater 2, etc... N represents pixels from crater N.

Number of craters: Calculate the max of the image array

## File Structure

```
crater-annotater/
├── crater_annotator.py    # Main application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .venv/                 # Virtual environment
```

## Tips

- **Drawing Precision**: For small craters, zoom in on your display or use the property spinboxes for fine adjustments
- **Overlapping Ellipses**: The tool supports overlapping annotations - each crater is stored separately
- **Save Frequently**: Save your work regularly using "Save Annotations (CSV)" to avoid losing progress
- **Undo**: Currently, there's no undo feature - delete and redraw if you make mistakes, or use the property spinboxes to adjust

## Troubleshooting

### Application won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that your virtual environment is activated

### Image won't load
- Supported formats: PNG, JPG, JPEG, TIF, TIFF
- Try converting your image to PNG if it's in an unusual format

### Mask export fails
- Ensure you've drawn at least one ellipse
- Check that you have write permissions in the target directory

## Future Enhancements

Potential features for future versions:
- Rotation support for ellipses
- Undo/redo functionality
- Zoom and pan controls
- Batch processing multiple images
- Auto-detection suggestions
- Classification labels (small/medium/large craters)

## License

This tool is part of the Mars crater detection project.
