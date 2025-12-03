from typing import List, Tuple
import cv2
import numpy as np


class ZoneDrawer:
    """Interactive tool to draw zones on video frames."""
    
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.current_points: List[Tuple[int, int]] = []
        self.zones: List[List[Tuple[int, int]]] = []
        self.zone_names: List[str] = []
        self.is_drawing = False
        
    def mouse_callback(self, event, x, y, flags, param) -> None:
        """Handle mouse events for drawing zones."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.current_points.append((x, y))
            self.is_drawing = True
            self.redraw()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current zone
            if len(self.current_points) >= 3:
                zone_name = f"Zone_{len(self.zones) + 1}"
                self.zones.append(self.current_points.copy())
                self.zone_names.append(zone_name)
                print(f"Zone '{zone_name}' created with {len(self.current_points)} points")
                self.current_points = []
                self.is_drawing = False
                self.redraw()
            else:
                print("Need at least 3 points to create a zone")
    
    def redraw(self) -> None:
        """Redraw frame with zones and current points."""
        self.display_frame = self.frame.copy()
        
        # Draw completed zones
        for i, points in enumerate(self.zones):
            pts = np.array(points, dtype=np.int32)
            color = (0, 0, 255)  # Red for restricted zones
            
            # Fill zone with transparency
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            
            # Draw border
            cv2.polylines(self.display_frame, [pts], isClosed=True, 
                         color=color, thickness=2)
            
            # Draw zone name
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(self.display_frame, self.zone_names[i], tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current points being drawn
        if len(self.current_points) > 0:
            for point in self.current_points:
                cv2.circle(self.display_frame, point, 5, (0, 255, 0), -1)
            
            # Draw lines connecting points
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], isClosed=False,
                            color=(0, 255, 0), thickness=2)
    
    def draw(self, window_name: str = "Define Restricted Zones") -> List[Tuple[str, List[Tuple[int, int]]]]:
        """Start interactive zone drawing.
        
        Returns:
            List of (zone_name, points) tuples
        """
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        instructions = [
            "Left Click: Add point to zone",
            "Right Click: Finish current zone",
            "Press 'c': Clear current zone",
            "Press 'u': Undo last completed zone",
            "Press 's': Save and exit",
            "Press 'q': Exit without saving"
        ]
        
        print("\n=== Zone Drawing Instructions ===")
        for inst in instructions:
            print(inst)
        print("=" * 35 + "\n")
        
        while True:
            # Add instructions to frame
            display = self.display_frame.copy()
            y_offset = 30
            for inst in instructions[:3]:  # Show first 3 instructions on frame
                cv2.putText(display, inst, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Clear current zone
                if len(self.current_points) > 0:
                    self.current_points = []
                    self.is_drawing = False
                    self.redraw()
                    print("Current zone cleared")
            
            elif key == ord('u'):  # Undo last zone
                if len(self.zones) > 0:
                    removed_zone = self.zone_names.pop()
                    self.zones.pop()
                    self.redraw()
                    print(f"Removed zone: {removed_zone}")
            
            elif key == ord('s'):  # Save and exit
                if len(self.current_points) >= 3:
                    # Auto-save current zone
                    self.zones.append(self.current_points.copy())
                    self.zone_names.append(f"Zone_{len(self.zones)}")
                
                cv2.destroyWindow(window_name)
                result = [(name, points) for name, points in zip(self.zone_names, self.zones)]
                print(f"\nSaved {len(result)} zone(s)")
                return result
            
            elif key == ord('q'):  # Quit without saving
                cv2.destroyWindow(window_name)
                print("\nExited without saving")
                return []
        
        return []
