import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class VideoConfig:
    source: str = "0"
    width: int = 1280
    height: int = 720
    display: bool = True
    save_output_path: str | None = None


@dataclass
class DetectionConfig:
    model_name: str = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5


@dataclass
class TrackingConfig:
    max_age: int = 30
    n_init: int = 3
    max_cosine_distance: float = 0.2


@dataclass
class RecognitionConfig:
    face_detection_image_size: int = 160
    face_match_threshold: float = 0.75


@dataclass
class DatabaseConfig:
    path: str = os.environ.get("DB_PATH", "surveillance.db")


@dataclass
class ZoneConfig:
    enabled: bool = True
    zones_file: str = "zones.json"
    alert_cooldown: int = 30
    violation_color: Tuple[int, int, int] = (0, 0, 255)


@dataclass
class NotificationConfig:
    sender_email: str = os.getenv('SENDER_EMAIL', '')
    sender_password: str = os.getenv('SENDER_PASSWORD', '')
    recipient_email: str = os.getenv('RECIPIENT_EMAIL', '')
    
    @property
    def enabled(self) -> bool:
        return bool(self.sender_email and self.sender_password and self.recipient_email)


# Instantiate all configs
video_config = VideoConfig()
detection_config = DetectionConfig()
tracking_config = TrackingConfig()
recognition_config = RecognitionConfig()
database_config = DatabaseConfig()
zone_config = ZoneConfig()
notification_config = NotificationConfig()
