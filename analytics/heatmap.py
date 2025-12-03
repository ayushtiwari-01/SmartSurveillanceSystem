from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional


class HeatmapGenerator:
    """Generate heatmap visualization of person movement and density."""
    
    def __init__(self, frame_shape: Tuple[int, int], decay_factor: float = 0.995) -> None:
        """
        Args:
            frame_shape: (height, width) of video frames
            decay_factor: How quickly old activity fades (0.99 = slow, 0.95 = fast)
        """
        self.height, self.width = frame_shape
        self.decay_factor = decay_factor
        
        # Accumulation map for heatmap (float32 for precision)
        self.heatmap_accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Track total frames processed
        self.frame_count = 0
    
    def update(self, detections: list) -> None:
        """Update heatmap with new detections.
        
        Args:
            detections: List of tracks with 'bbox' key containing (x1, y1, x2, y2)
        """
        # Apply decay to previous heatmap (makes old activity fade)
        self.heatmap_accumulator *= self.decay_factor
        
        # Add new detections
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            x2 = max(0, min(x2, self.width - 1))
            y2 = max(0, min(y2, self.height - 1))
            
            # Add heat to the bounding box area
            # Use center point with gaussian-like distribution
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Create a small gaussian-like blob around the person
            radius = max((x2 - x1), (y2 - y1)) // 2
            radius = max(20, min(radius, 100))  # Clamp radius
            
            # Draw filled circle on accumulator
            cv2.circle(self.heatmap_accumulator, (center_x, center_y), 
                      radius, 1.0, -1)
        
        self.frame_count += 1
    
    def generate_heatmap_overlay(self, background_frame: np.ndarray, 
                                 alpha: float = 0.6) -> np.ndarray:
        """Generate heatmap overlay on video frame.
        
        Args:
            background_frame: Original video frame
            alpha: Transparency of heatmap (0.0 = invisible, 1.0 = opaque)
            
        Returns:
            Frame with heatmap overlay
        """
        # Normalize heatmap to 0-255 range
        if self.heatmap_accumulator.max() > 0:
            normalized = (self.heatmap_accumulator / self.heatmap_accumulator.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.heatmap_accumulator, dtype=np.uint8)
        
        # Apply Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(normalized, (51, 51), 0)
        
        # Apply colormap (COLORMAP_JET: blue=cold, red=hot)
        heatmap_colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        
        # Blend with original frame
        output = cv2.addWeighted(background_frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return output
    
    def save_heatmap(self, filepath: str, background_frame: Optional[np.ndarray] = None) -> None:
        """Save heatmap to file.
        
        Args:
            filepath: Path to save image (e.g., 'heatmap.png')
            background_frame: Optional background frame, if None uses black background
        """
        if background_frame is None:
            # Create black background
            background_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Generate heatmap overlay
        heatmap_image = self.generate_heatmap_overlay(background_frame, alpha=0.7)
        
        # Add title and statistics
        cv2.putText(heatmap_image, "Activity Heatmap", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(heatmap_image, f"Frames analyzed: {self.frame_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add color legend
        legend_height = 30
        legend_width = 200
        legend = np.linspace(0, 255, legend_width).astype(np.uint8)
        legend = np.repeat(legend.reshape(1, -1), legend_height, axis=0)
        legend_colored = cv2.applyColorMap(legend, cv2.COLORMAP_JET)
        
        # Place legend on image
        y_pos = self.height - 60
        x_pos = 20
        heatmap_image[y_pos:y_pos+legend_height, x_pos:x_pos+legend_width] = legend_colored
        
        # Add legend labels
        cv2.putText(heatmap_image, "Low", (x_pos, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(heatmap_image, "High", (x_pos + legend_width - 30, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save to file
        cv2.imwrite(filepath, heatmap_image)
        print(f"\n✓ Heatmap saved to: {filepath}")
    
    def get_heatmap_array(self) -> np.ndarray:
        """Get raw heatmap accumulator array."""
        return self.heatmap_accumulator.copy()
    
    def reset(self) -> None:
        """Reset heatmap accumulator."""
        self.heatmap_accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        self.frame_count = 0
