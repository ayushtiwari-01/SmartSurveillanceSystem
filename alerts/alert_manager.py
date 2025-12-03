from __future__ import annotations
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import json
import cv2


class AlertType(Enum):
    """Types of security alerts."""
    ZONE_VIOLATION = "Zone Violation"
    UNKNOWN_PERSON = "Unknown Person"
    CROWD_THRESHOLD = "Crowd Threshold"
    LOITERING = "Loitering"


class Alert:
    """Represents a security alert."""
    
    def __init__(self, alert_type: AlertType, message: str, 
                 track_id: Optional[int] = None, 
                 zone_name: Optional[str] = None,
                 person_name: Optional[str] = None) -> None:
        self.alert_type = alert_type
        self.message = message
        self.track_id = track_id
        self.zone_name = zone_name
        self.person_name = person_name
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.alert_type.value}: {self.message}"
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'type': self.alert_type.value,
            'message': self.message,
            'track_id': self.track_id,
            'zone_name': self.zone_name,
            'person_name': self.person_name,
            'timestamp': self.timestamp.isoformat()
        }


class AlertManager:
    """Manages security alerts and notifications."""
    
    def __init__(self, max_alerts: int = 100, notifier=None) -> None:
        self.alerts: List[Alert] = []
        self.max_alerts = max_alerts
        self.active_violations: Dict[int, set] = {}
        self.notifier = notifier
        
    def add_alert(self, alert: Alert) -> None:
        """Add a new alert."""
        self.alerts.append(alert)
        print(alert)  # Print to console
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    def create_zone_violation_alert(self, track_id: int, zone_name: str, 
                                   person_name: Optional[str] = None) -> Optional[Alert]:
        """Create and add zone violation alert with email notification."""
        # Check if this is a new violation
        if track_id not in self.active_violations:
            self.active_violations[track_id] = set()
        
        zone_hash = hash(zone_name)
        if zone_hash in self.active_violations[track_id]:
            return None  # Already alerted
        
        self.active_violations[track_id].add(zone_hash)
        
        person_str = f" ({person_name})" if person_name else ""
        message = f"Track {track_id}{person_str} entered restricted zone: {zone_name}"
        
        alert = Alert(
            alert_type=AlertType.ZONE_VIOLATION,
            message=message,
            track_id=track_id,
            zone_name=zone_name,
            person_name=person_name
        )
        
        self.add_alert(alert)
        
        # Send email notification
        if self.notifier:
            self.notifier.send_zone_violation(track_id, zone_name, person_name)
        
        return alert
    
    def clear_track_violations(self, track_id: int) -> None:
        """Clear violation history when track is lost."""
        if track_id in self.active_violations:
            del self.active_violations[track_id]
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get most recent alerts."""
        return self.alerts[-count:]
    
    def save_alerts(self, filepath: str) -> None:
        """Save alerts to JSON file."""
        data = [alert.to_dict() for alert in self.alerts]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def draw_alerts(self, frame, max_display: int = 5) -> None:
        """Draw recent alerts on frame."""
        recent = self.get_recent_alerts(max_display)
        
        y_offset = frame.shape[0] - 30 - (len(recent) * 25)
        for alert in recent:
            time_str = alert.timestamp.strftime("%H:%M:%S")
            text = f"[{time_str}] {alert.message}"
            
            # Choose color based on alert type
            color = (0, 0, 255) if alert.alert_type == AlertType.ZONE_VIOLATION else (0, 255, 255)
            
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
