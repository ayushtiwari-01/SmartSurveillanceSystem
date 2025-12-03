from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import cv2
import json


class Zone:
    """Represents a restricted zone with polygon coordinates."""
    
    def __init__(self, zone_id: int, name: str, points: List[Tuple[int, int]], 
                 color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        self.zone_id = zone_id
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color
        self.alpha = 0.3  # Transparency for overlay
        
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside the zone using cv2.pointPolygonTest."""
        result = cv2.pointPolygonTest(self.points, (float(x), float(y)), False)
        return result >= 0
    
    def contains_bbox(self, x1: int, y1: int, x2: int, y2: int, 
                      threshold: float = 0.5) -> bool:
        """Check if bounding box overlaps with zone.
        
        Args:
            threshold: Percentage of bbox center or area that must be in zone
        """
        # Check if bbox center is in zone
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return self.contains_point(center_x, center_y)
    
    def draw(self, frame: np.ndarray, filled: bool = True) -> np.ndarray:
        """Draw zone on frame."""
        overlay = frame.copy()
        
        if filled:
            cv2.fillPoly(overlay, [self.points], self.color)
            cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        
        # Draw border
        cv2.polylines(frame, [self.points], isClosed=True, 
                     color=self.color, thickness=2)
        
        # Draw zone name
        centroid = np.mean(self.points, axis=0).astype(int)
        cv2.putText(frame, self.name, tuple(centroid), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def to_dict(self) -> Dict:
        """Convert zone to dictionary for storage."""
        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'points': self.points.tolist(),
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> Zone:
        """Create zone from dictionary."""
        return cls(
            zone_id=data['zone_id'],
            name=data['name'],
            points=data['points'],
            color=tuple(data['color'])
        )


class ZoneManager:
    """Manages multiple restricted zones and detects violations."""
    
    def __init__(self) -> None:
        self.zones: List[Zone] = []
        self.violations: Dict[int, set] = {}  # track_id -> set of zone_ids violated
        
    def add_zone(self, name: str, points: List[Tuple[int, int]], 
                 color: Tuple[int, int, int] = (0, 0, 255)) -> Zone:
        """Add a new restricted zone."""
        zone_id = len(self.zones) + 1
        zone = Zone(zone_id, name, points, color)
        self.zones.append(zone)
        return zone
    
    def remove_zone(self, zone_id: int) -> bool:
        """Remove a zone by ID."""
        for i, zone in enumerate(self.zones):
            if zone.zone_id == zone_id:
                self.zones.pop(i)
                return True
        return False
    
    def check_violations(self, track_id: int, bbox: Tuple[int, int, int, int]) -> List[Zone]:
        """Check if a tracked person violates any restricted zones.
        
        Returns:
            List of zones that are violated
        """
        x1, y1, x2, y2 = bbox
        violated_zones = []
        
        for zone in self.zones:
            if zone.contains_bbox(x1, y1, x2, y2):
                violated_zones.append(zone)
                
                # Track violations per track
                if track_id not in self.violations:
                    self.violations[track_id] = set()
                
                # New violation detected
                if zone.zone_id not in self.violations[track_id]:
                    self.violations[track_id].add(zone.zone_id)
        
        return violated_zones
    
    def clear_violations(self, track_id: int) -> None:
        """Clear violation history for a track."""
        if track_id in self.violations:
            del self.violations[track_id]
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame."""
        for zone in self.zones:
            frame = zone.draw(frame, filled=True)
        return frame
    
    def save_zones(self, filepath: str) -> None:
        """Save zones to JSON file."""
        data = [zone.to_dict() for zone in self.zones]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_zones(self, filepath: str) -> None:
        """Load zones from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.zones = [Zone.from_dict(z) for z in data]
        except FileNotFoundError:
            print(f"Zone file {filepath} not found. Starting with no zones.")
        except json.JSONDecodeError:
            print(f"Error reading zone file {filepath}.")
