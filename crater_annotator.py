#!/usr/bin/env python3
"""
Mars Crater Annotation Tool
A desktop application for annotating craters on Mars images using ellipses.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, QMessageBox,
    QInputDialog, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QCheckBox, QSplitter
)
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QBrush,
    QKeySequence, QCursor
)

import numpy as np
from PIL import Image


class AnnotationListItem(QWidget):
    """Custom widget for displaying annotation in list with Delete button."""
    
    clicked = pyqtSignal(int)  # index
    delete_clicked = pyqtSignal(int)  # index
    
    def __init__(self, index: int, ellipse: 'Ellipse', parent=None):
        super().__init__(parent)
        self.index = index
        self.ellipse = ellipse
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Label with ellipse info (clickable)
        label_text = f"{index + 1}. Ellipse at ({ellipse.center_x:.1f}, {ellipse.center_y:.1f})"
        self.label = QLabel(label_text)
        self.label.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.label, 1)
        
        # Delete button
        btn_delete = QPushButton("Delete")
        btn_delete.setMaximumWidth(60)
        btn_delete.clicked.connect(lambda: self.delete_clicked.emit(self.index))
        layout.addWidget(btn_delete)
    
    def mousePressEvent(self, event):
        """Handle click on annotation item."""
        self.clicked.emit(self.index)


class Ellipse:
    """Represents an annotated crater ellipse."""
    
    def __init__(self, center_x: float, center_y: float, 
                 radius_x: float, radius_y: float, 
                 rotation: float = 0.0, label: str = "crater"):
        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.rotation = rotation
        self.label = label
        self.selected = False
    
    def contains_point(self, x: float, y: float, tolerance: float = 10.0) -> bool:
        """Check if a point is near the ellipse boundary."""
        # Simplified hit test - check if point is within tolerance of ellipse
        dx = (x - self.center_x) / max(self.radius_x, 1)
        dy = (y - self.center_y) / max(self.radius_y, 1)
        distance = np.sqrt(dx**2 + dy**2)
        return abs(distance - 1.0) < tolerance / max(self.radius_x, self.radius_y)
    
    def is_inside(self, x: float, y: float) -> bool:
        """Check if point is inside the ellipse."""
        dx = (x - self.center_x) / max(self.radius_x, 1)
        dy = (y - self.center_y) / max(self.radius_y, 1)
        return dx**2 + dy**2 <= 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'xc': self.center_x,
            'yc': self.center_y,
            'rx': self.radius_x,
            'ry': self.radius_y,
            'rotation': self.rotation,
            'label': self.label
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Ellipse':
        """Create ellipse from dictionary."""
        return Ellipse(
            center_x=data['xc'],
            center_y=data['yc'],
            radius_x=data['rx'],
            radius_y=data['ry'],
            rotation=data.get('rotation', 0.0),
            label=data.get('label', 'crater')
        )


class ImageCanvas(QLabel):
    """Canvas widget for displaying image and drawing ellipses."""
    
    ellipse_added = pyqtSignal(Ellipse)
    ellipse_modified = pyqtSignal(Ellipse)
    ellipse_selected = pyqtSignal(object)  # None or Ellipse
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Image data
        self.original_image: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.image_path: Optional[str] = None
        
        # Ellipse storage
        self.ellipses: List[Ellipse] = []
        self.selected_ellipse: Optional[Ellipse] = None
        
        # Drawing state
        self.drawing_mode = False
        self.drawing_start: Optional[QPoint] = None
        self.drawing_current: Optional[QPoint] = None
        
        # Editing state
        self.dragging_ellipse: Optional[Ellipse] = None
        self.resizing_ellipse: Optional[Ellipse] = None
        self.drag_offset: Optional[QPoint] = None
        
        # Display settings
        self.show_ellipses = True
        self.ellipse_color = QColor(0, 255, 0, 180)
        self.selected_color = QColor(255, 255, 0, 200)
        self.drawing_color = QColor(255, 0, 0, 150)
    
    def load_image(self, image_path: str) -> bool:
        """Load an image file."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False
        
        self.original_image = pixmap
        self.image_path = image_path
        self.ellipses.clear()
        self.selected_ellipse = None
        self.update_display()
        return True
    
    def update_display(self):
        """Redraw the canvas with image and ellipses."""
        if self.original_image is None:
            return
        
        # Create a copy of the image to draw on
        self.display_pixmap = self.original_image.copy()
        
        if self.show_ellipses:
            painter = QPainter(self.display_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw all ellipses
            for ellipse in self.ellipses:
                color = self.selected_color if ellipse.selected else self.ellipse_color
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
                
                painter.drawEllipse(
                    QRectF(
                        ellipse.center_x - ellipse.radius_x,
                        ellipse.center_y - ellipse.radius_y,
                        2 * ellipse.radius_x,
                        2 * ellipse.radius_y
                    )
                )
                
                # Draw center point
                painter.drawEllipse(QRectF(
                    ellipse.center_x - 3, ellipse.center_y - 3, 6, 6
                ))
            
            # Draw current drawing ellipse
            if self.drawing_mode and self.drawing_start and self.drawing_current:
                pen = QPen(self.drawing_color, 2, Qt.DashLine)
                painter.setPen(pen)
                
                x1, y1 = self.drawing_start.x(), self.drawing_start.y()
                x2, y2 = self.drawing_current.x(), self.drawing_current.y()
                
                rx = abs(x2 - x1) / 2
                ry = abs(y2 - y1) / 2
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                painter.drawEllipse(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
            
            painter.end()
        
        self.setPixmap(self.display_pixmap)
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.original_image is None:
            return
        
        pos = event.pos()
        
        if event.button() == Qt.LeftButton:
            # Check if clicking on an existing ellipse
            clicked_ellipse = self.get_ellipse_at(pos.x(), pos.y())
            
            if event.modifiers() & Qt.ControlModifier:
                # Ctrl+Click: Start drawing new ellipse
                self.drawing_mode = True
                self.drawing_start = pos
                self.drawing_current = pos
                self.deselect_all()
            elif clicked_ellipse:
                # Click on ellipse: Select and prepare to drag
                self.select_ellipse(clicked_ellipse)
                self.dragging_ellipse = clicked_ellipse
                self.drag_offset = QPoint(
                    pos.x() - int(clicked_ellipse.center_x),
                    pos.y() - int(clicked_ellipse.center_y)
                )
            else:
                # Click on empty space: Start drawing
                self.drawing_mode = True
                self.drawing_start = pos
                self.drawing_current = pos
                self.deselect_all()
        
        elif event.button() == Qt.RightButton:
            # Right click: Delete ellipse
            clicked_ellipse = self.get_ellipse_at(pos.x(), pos.y())
            if clicked_ellipse:
                self.ellipses.remove(clicked_ellipse)
                if self.selected_ellipse == clicked_ellipse:
                    self.selected_ellipse = None
                    self.ellipse_selected.emit(None)
                self.update_display()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.original_image is None:
            return
        
        pos = event.pos()
        
        # Update cursor based on position
        if self.get_ellipse_at(pos.x(), pos.y()):
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CrossCursor))
        
        if self.drawing_mode and self.drawing_start:
            # Update drawing preview
            self.drawing_current = pos
            self.update_display()
        
        elif self.dragging_ellipse and self.drag_offset:
            # Drag ellipse to new position
            self.dragging_ellipse.center_x = pos.x() - self.drag_offset.x()
            self.dragging_ellipse.center_y = pos.y() - self.drag_offset.y()
            self.update_display()
            self.ellipse_modified.emit(self.dragging_ellipse)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.drawing_mode and self.drawing_start and self.drawing_current:
                # Finalize ellipse drawing
                x1, y1 = self.drawing_start.x(), self.drawing_start.y()
                x2, y2 = self.drawing_current.x(), self.drawing_current.y()
                
                rx = abs(x2 - x1) / 2
                ry = abs(y2 - y1) / 2
                
                if rx > 1 and ry > 1:  # Minimum size threshold
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    new_ellipse = Ellipse(cx, cy, rx, ry)
                    self.ellipses.append(new_ellipse)
                    self.ellipse_added.emit(new_ellipse)
                
                self.drawing_mode = False
                self.drawing_start = None
                self.drawing_current = None
                self.update_display()
            
            elif self.dragging_ellipse:
                self.dragging_ellipse = None
                self.drag_offset = None
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Delete and self.selected_ellipse:
            # Delete selected ellipse
            self.ellipses.remove(self.selected_ellipse)
            self.selected_ellipse = None
            self.ellipse_selected.emit(None)
            self.update_display()
        
        elif event.key() == Qt.Key_Escape:
            # Cancel drawing
            if self.drawing_mode:
                self.drawing_mode = False
                self.drawing_start = None
                self.drawing_current = None
                self.update_display()
    
    def get_ellipse_at(self, x: float, y: float) -> Optional[Ellipse]:
        """Find ellipse at given coordinates."""
        # Check in reverse order (top to bottom)
        for ellipse in reversed(self.ellipses):
            if ellipse.is_inside(x, y):
                return ellipse
        return None
    
    def select_ellipse(self, ellipse: Optional[Ellipse]):
        """Select an ellipse."""
        self.deselect_all()
        if ellipse:
            ellipse.selected = True
            self.selected_ellipse = ellipse
            self.ellipse_selected.emit(ellipse)
        self.update_display()
    
    def deselect_all(self):
        """Deselect all ellipses."""
        for ellipse in self.ellipses:
            ellipse.selected = False
        self.selected_ellipse = None
        self.ellipse_selected.emit(None)
    
    def clear_ellipses(self):
        """Remove all ellipses."""
        self.ellipses.clear()
        self.selected_ellipse = None
        self.ellipse_selected.emit(None)
        self.update_display()
    
    def export_to_csv(self, output_path: str) -> bool:
        """Export ellipses to CSV format."""
        if not self.ellipses:
            return False
        
        data = [e.to_dict() for e in self.ellipses]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return True
    

    def export_to_mask(self, output_path: str) -> bool:
        """Export ellipses to a label PNG (0=background, 1=crater 1, 2=crater 2, ...)."""
        if not self.ellipses or self.original_image is None:
            return False

        H = self.original_image.height()
        W = self.original_image.width()

        # Single-channel label map
        mask = np.zeros((H, W), dtype=np.uint8)

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        for i, ellipse in enumerate(self.ellipses, start=1):
            ellipse_eq = (
                ((x_coords - ellipse.center_x) ** 2 / (ellipse.radius_x ** 2)) +
                ((y_coords - ellipse.center_y) ** 2 / (ellipse.radius_y ** 2))
            )
            mask[ellipse_eq <= 1] = i

        Image.fromarray(mask, mode='L').save(output_path)
        return True
    
    def load_annotations(self, annotation_path: str) -> bool:
        """Load annotations from CSV file."""
        try:
            if annotation_path.endswith('.csv'):
                df = pd.read_csv(annotation_path)
                self.ellipses.clear()
                
                for _, row in df.iterrows():
                    ellipse = Ellipse(
                        center_x=row['xc'],
                        center_y=row['yc'],
                        radius_x=row['rx'],
                        radius_y=row['ry'],
                        rotation=row.get('rotation', 0.0),
                        label=row.get('label', 'crater')
                    )
                    self.ellipses.append(ellipse)
                
                self.update_display()
                return True
            
            elif annotation_path.endswith('.json'):
                with open(annotation_path, 'r') as f:
                    data = json.load(f)
                    self.ellipses = [Ellipse.from_dict(e) for e in data]
                    self.update_display()
                    return True
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False
        
        return False


class CraterAnnotatorApp(QMainWindow):
    """Main application window for crater annotation."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mars Crater Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize UI
        self.init_ui()
        
        # Current state
        self.current_image_path: Optional[str] = None
        self.current_annotation_path: Optional[str] = None
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Canvas
        self.canvas = ImageCanvas()
        self.canvas.ellipse_added.connect(self.on_ellipse_added)
        self.canvas.ellipse_modified.connect(self.on_ellipse_modified)
        self.canvas.ellipse_selected.connect(self.on_ellipse_selected)
        splitter.addWidget(self.canvas)
        
        # Right panel: Controls
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Set splitter sizes (canvas gets more space)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def create_control_panel(self) -> QWidget:
        """Create the control panel with buttons and settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        btn_load_image = QPushButton("Load Image")
        btn_load_image.clicked.connect(self.load_image)
        file_layout.addWidget(btn_load_image)
        
        btn_load_annotations = QPushButton("Load Annotations")
        btn_load_annotations.clicked.connect(self.load_annotations)
        file_layout.addWidget(btn_load_annotations)
        
        btn_save_annotations = QPushButton("Save Annotations (CSV)")
        btn_save_annotations.clicked.connect(self.save_annotations_csv)
        file_layout.addWidget(btn_save_annotations)
        
        btn_export_mask = QPushButton("Export as Mask (.png)")
        btn_export_mask.clicked.connect(self.export_mask)
        file_layout.addWidget(btn_export_mask)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Drawing instructions group
        inst_group = QGroupBox("Instructions")
        inst_layout = QVBoxLayout()
        
        instructions = QLabel(
            "<b>Controls:</b><br>"
            "• Click & Drag: Draw ellipse<br>"
            "• Ctrl+Click & Drag: Force new ellipse<br>"
            "• Click on ellipse: Select<br>"
            "• Drag ellipse: Move<br>"
            "• Right-click: Delete ellipse<br>"
            "• Delete key: Remove selected<br>"
            "• Esc: Cancel drawing"
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignLeft)
        inst_layout.addWidget(instructions)
        
        inst_group.setLayout(inst_layout)
        layout.addWidget(inst_group)
        
        # Ellipse properties group
        self.props_group = QGroupBox("Selected Ellipse Properties")
        props_layout = QFormLayout()
        
        self.spin_center_x = QDoubleSpinBox()
        self.spin_center_x.setRange(0, 10000)
        self.spin_center_x.setDecimals(1)
        self.spin_center_x.valueChanged.connect(self.update_selected_ellipse)
        props_layout.addRow("Center X:", self.spin_center_x)
        
        self.spin_center_y = QDoubleSpinBox()
        self.spin_center_y.setRange(0, 10000)
        self.spin_center_y.setDecimals(1)
        self.spin_center_y.valueChanged.connect(self.update_selected_ellipse)
        props_layout.addRow("Center Y:", self.spin_center_y)
        
        self.spin_radius_x = QDoubleSpinBox()
        self.spin_radius_x.setRange(1, 5000)
        self.spin_radius_x.setDecimals(1)
        self.spin_radius_x.valueChanged.connect(self.update_selected_ellipse)
        props_layout.addRow("Radius X:", self.spin_radius_x)
        
        self.spin_radius_y = QDoubleSpinBox()
        self.spin_radius_y.setRange(1, 5000)
        self.spin_radius_y.setDecimals(1)
        self.spin_radius_y.valueChanged.connect(self.update_selected_ellipse)
        props_layout.addRow("Radius Y:", self.spin_radius_y)
        
        self.props_group.setLayout(props_layout)
        self.props_group.setEnabled(False)
        layout.addWidget(self.props_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.lbl_count = QLabel("Ellipses: 0")
        stats_layout.addWidget(self.lbl_count)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        btn_clear = QPushButton("Clear All Ellipses")
        btn_clear.clicked.connect(self.clear_all)
        action_layout.addWidget(btn_clear)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Annotations list group
        annot_group = QGroupBox("Annotations")
        annot_layout = QVBoxLayout()
        
        self.list_annotations = QListWidget()
        annot_layout.addWidget(self.list_annotations)
        
        annot_group.setLayout(annot_layout)
        layout.addWidget(annot_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return panel
    
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        
        if file_path:
            if self.canvas.load_image(file_path):
                self.current_image_path = file_path
                self.setWindowTitle(f"Mars Crater Annotation Tool - {Path(file_path).name}")
                self.update_statistics()
            else:
                QMessageBox.warning(self, "Error", "Failed to load image.")
    
    def load_annotations(self):
        """Load existing annotations."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Annotations",
            "",
            "Annotation Files (*.csv *.json);;All Files (*)"
        )
        
        if file_path:
            if self.canvas.load_annotations(file_path):
                self.current_annotation_path = file_path
                self.update_statistics()
                QMessageBox.information(self, "Success", f"Loaded {len(self.canvas.ellipses)} annotations.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load annotations.")
    
    def save_annotations_csv(self):
        """Save annotations to CSV file."""
        if not self.canvas.ellipses:
            QMessageBox.warning(self, "Warning", "No ellipses to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotations",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            
            if self.canvas.export_to_csv(file_path):
                self.current_annotation_path = file_path
                QMessageBox.information(self, "Success", f"Saved {len(self.canvas.ellipses)} annotations to CSV.")
            else:
                QMessageBox.warning(self, "Error", "Failed to save annotations.")
    
    def export_mask(self):
        """Export annotations as PyTorch mask."""
        if not self.canvas.ellipses:
            QMessageBox.warning(self, "Warning", "No ellipses to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mask",
            "",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.png'):
                file_path += '.png'
            
            if self.canvas.export_to_mask(file_path):
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Exported {len(self.canvas.ellipses)} craters as mask.\n"
                    f"Mask shape: ({len(self.canvas.ellipses)}, "
                    f"{self.canvas.original_image.height()}, "
                    f"{self.canvas.original_image.width()})"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to export mask.")
    
    def clear_all(self):
        """Clear all ellipses after confirmation."""
        if not self.canvas.ellipses:
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            f"Are you sure you want to delete all {len(self.canvas.ellipses)} ellipses?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.clear_ellipses()
            self.update_statistics()
    
    def on_ellipse_added(self, ellipse: Ellipse):
        """Handle new ellipse addition."""
        self.update_statistics()
    
    def on_ellipse_modified(self, ellipse: Ellipse):
        """Handle ellipse modification."""
        self.update_ellipse_properties()
    
    def on_ellipse_selected(self, ellipse: Optional[Ellipse]):
        """Handle ellipse selection change."""
        if ellipse:
            self.props_group.setEnabled(True)
            self.update_ellipse_properties()
            # Update list selection
            for i, e in enumerate(self.canvas.ellipses):
                if e == ellipse:
                    self.list_annotations.setCurrentRow(i)
                    break
        else:
            self.props_group.setEnabled(False)
    
    def update_ellipse_properties(self):
        """Update property spinboxes with selected ellipse values."""
        if self.canvas.selected_ellipse:
            # Block signals to prevent recursive updates
            self.spin_center_x.blockSignals(True)
            self.spin_center_y.blockSignals(True)
            self.spin_radius_x.blockSignals(True)
            self.spin_radius_y.blockSignals(True)
            
            self.spin_center_x.setValue(self.canvas.selected_ellipse.center_x)
            self.spin_center_y.setValue(self.canvas.selected_ellipse.center_y)
            self.spin_radius_x.setValue(self.canvas.selected_ellipse.radius_x)
            self.spin_radius_y.setValue(self.canvas.selected_ellipse.radius_y)
            
            self.spin_center_x.blockSignals(False)
            self.spin_center_y.blockSignals(False)
            self.spin_radius_x.blockSignals(False)
            self.spin_radius_y.blockSignals(False)
    
    def update_selected_ellipse(self):
        """Update selected ellipse from property spinboxes."""
        if self.canvas.selected_ellipse:
            self.canvas.selected_ellipse.center_x = self.spin_center_x.value()
            self.canvas.selected_ellipse.center_y = self.spin_center_y.value()
            self.canvas.selected_ellipse.radius_x = self.spin_radius_x.value()
            self.canvas.selected_ellipse.radius_y = self.spin_radius_y.value()
            self.canvas.update_display()
    
    def update_statistics(self):
        """Update statistics display."""
        self.lbl_count.setText(f"Ellipses: {len(self.canvas.ellipses)}")
        self.update_annotation_list()
    
    def update_annotation_list(self):
        """Refresh the annotation list."""
        self.list_annotations.clear()
        
        for i, ellipse in enumerate(self.canvas.ellipses):
            # Create custom item widget
            item_widget = AnnotationListItem(i, ellipse)
            item_widget.clicked.connect(self.on_annotation_edit)
            item_widget.delete_clicked.connect(self.on_annotation_delete)
            
            # Add to list
            list_item = QListWidgetItem(self.list_annotations)
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_annotations.addItem(list_item)
            self.list_annotations.setItemWidget(list_item, item_widget)
    
    def on_annotation_edit(self, index: int):
        """Handle edit button click for annotation."""
        if 0 <= index < len(self.canvas.ellipses):
            ellipse = self.canvas.ellipses[index]
            self.canvas.select_ellipse(ellipse)
    
    def on_annotation_delete(self, index: int):
        """Handle delete button click for annotation."""
        if 0 <= index < len(self.canvas.ellipses):
            ellipse = self.canvas.ellipses[index]
            self.canvas.ellipses.remove(ellipse)
            if self.canvas.selected_ellipse == ellipse:
                self.canvas.selected_ellipse = None
                self.canvas.ellipse_selected.emit(None)
            self.canvas.update_display()
            self.update_statistics()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = CraterAnnotatorApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
