import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
import os

from config import (detection_config, tracking_config, recognition_config, 
                   database_config, video_config, zone_config, notification_config)

from detectors.yolo_detector import YoloPersonDetector
from tracking.deepsort_tracker import DeepSortTracker
from recognition.facenet_recognizer import FaceNetRecognizer
from db.database import Database
from zones.zone_manager import ZoneManager
from alerts.alert_manager import AlertManager
from alerts.notifier import EmailNotifier
from analytics.heatmap import HeatmapGenerator


# Page configuration
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False
    st.session_state.running = False
    st.session_state.detector = None
    st.session_state.tracker = None
    st.session_state.recognizer = None
    st.session_state.zone_manager = None
    st.session_state.alert_manager = None
    st.session_state.heatmap = None
    st.session_state.cap = None
    st.session_state.stats = {
        'active_tracks': 0,
        'total_alerts': 0,
        'violations': 0,
        'known_persons': 0
    }
    st.session_state.id_to_name = {}


def initialize_models():
    """Initialize all models."""
    try:
        # Detector
        st.session_state.detector = YoloPersonDetector(
            detection_config.model_name,
            detection_config.conf_threshold,
            detection_config.iou_threshold
        )
        
        # Tracker
        st.session_state.tracker = DeepSortTracker(
            tracking_config.max_age,
            tracking_config.n_init,
            tracking_config.max_cosine_distance
        )
        
        # Recognizer
        st.session_state.recognizer = FaceNetRecognizer(
            image_size=recognition_config.face_detection_image_size,
            match_threshold=recognition_config.face_match_threshold
        )
        
        # Load known faces
        db = Database(database_config.path)
        labels, embs = db.get_all_embeddings()
        if len(embs) > 0:
            embs = np.array(embs, dtype=np.float32)
            st.session_state.recognizer.set_known(labels, embs)
            st.session_state.stats['known_persons'] = len(labels)
        else:
            st.session_state.stats['known_persons'] = 0
        
        # Zone manager
        st.session_state.zone_manager = ZoneManager()
        if os.path.exists(zone_config.zones_file):
            st.session_state.zone_manager.load_zones(zone_config.zones_file)
        
        # Email notifier
        notifier = None
        if notification_config.enabled:
            notifier = EmailNotifier(
                smtp_server='smtp.gmail.com',
                smtp_port=587,
                sender_email=notification_config.sender_email,
                sender_password=notification_config.sender_password,
                recipient_email=notification_config.recipient_email
            )
        
        # Alert manager
        st.session_state.alert_manager = AlertManager(notifier=notifier)
        
        st.session_state.models_initialized = True
        return True
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return False


def start_surveillance():
    """Start surveillance."""
    try:
        st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Use same resolution as main.py (1280x720)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.width)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.height)
        
        if not st.session_state.cap.isOpened():
            st.error("Failed to open camera")
            return False
        
        # Initialize heatmap with actual frame dimensions
        actual_height = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_width = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        st.session_state.heatmap = HeatmapGenerator((actual_height, actual_width), decay_factor=0.995)
        
        st.session_state.running = True
        return True
    except Exception as e:
        st.error(f"Error starting surveillance: {str(e)}")
        return False


def stop_surveillance():
    """Stop surveillance."""
    st.session_state.running = False
    
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    
    # Save heatmap
    if st.session_state.heatmap and st.session_state.heatmap.frame_count > 0:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.height)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.flip(frame, 1)
            st.session_state.heatmap.save_heatmap("heatmap_output.png", frame)
    
    # Save alerts
    if st.session_state.alert_manager:
        st.session_state.alert_manager.save_alerts("alerts_log.json")


def process_frame(show_heatmap=False):
    """Process single frame."""
    if not st.session_state.cap or not st.session_state.cap.isOpened():
        return None
    
    ret, frame = st.session_state.cap.read()
    if not ret:
        return None
    
    frame = cv2.flip(frame, 1)
    
    # Detection
    people = st.session_state.detector.detect(frame)
    
    # Tracking
    tracks = st.session_state.tracker.update(people, frame)
    
    # Update heatmap
    st.session_state.heatmap.update(tracks)
    
    # Update stats
    st.session_state.stats['active_tracks'] = len(tracks)
    
    # Process tracks
    for t in tracks:
        tid = t['track_id']
        x1, y1, x2, y2 = t['bbox']
        
        # Zone violation check
        violated_zones = st.session_state.zone_manager.check_violations(tid, (x1, y1, x2, y2))
        
        if len(violated_zones) > 0:
            for zone in violated_zones:
                person_name = st.session_state.id_to_name.get(tid)
                st.session_state.alert_manager.create_zone_violation_alert(tid, zone.name, person_name)
            bbox_color = (0, 0, 255)  # Red
        else:
            bbox_color = (0, 255, 0)  # Green
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
        label = f"ID {tid}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
    
    # Apply heatmap overlay
    if show_heatmap and st.session_state.heatmap:
        frame = st.session_state.heatmap.generate_heatmap_overlay(frame, alpha=0.4)
    
    # Draw zones
    frame = st.session_state.zone_manager.draw_zones(frame)
    
    # Draw crowd count
    cv2.putText(frame, f"Crowd: {len(tracks)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Update alert count
    st.session_state.stats['total_alerts'] = len(st.session_state.alert_manager.alerts)
    st.session_state.stats['violations'] = sum(1 for a in st.session_state.alert_manager.alerts 
                                               if a.alert_type.value == "Zone Violation")
    
    return frame


# Main UI
st.title("🎥 Smart Surveillance Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Initialize button
    if st.button("🔄 Initialize Models", disabled=st.session_state.models_initialized):
        with st.spinner("Loading models... This may take 30-60 seconds"):
            if initialize_models():
                st.success("✅ Models loaded!")
                st.rerun()
    
    if st.session_state.models_initialized:
        st.success("✅ Models Ready")
    
    st.divider()
    
    # Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.running or not st.session_state.models_initialized):
            if start_surveillance():
                st.success("Started!")
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.running):
            stop_surveillance()
            st.info("Stopped!")
            st.rerun()
    
    st.divider()
    
    # Settings
    st.header("🎛️ Settings")
    show_heatmap = st.checkbox("Show Heatmap", value=False)
    auto_refresh = st.checkbox("Auto Refresh", value=True, help="Uncheck to pause updates")
    
    st.divider()
    
    # Info
    st.header("📋 Info")
    st.metric("Known Persons", st.session_state.stats['known_persons'])
    if st.session_state.zone_manager:
        st.metric("Zones", len(st.session_state.zone_manager.zones))
    st.metric("Email", "✅" if notification_config.enabled else "❌")
    
    # Display actual resolution
    if st.session_state.cap:
        actual_w = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.info(f"Resolution: {actual_w}x{actual_h}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()
    
    if st.session_state.running:
        frame = process_frame(show_heatmap=show_heatmap)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # FIXED: Changed use_column_width to use_container_width
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        else:
            video_placeholder.error("Failed to read frame")
    else:
        video_placeholder.info("📷 Click 'Start' to begin surveillance")

with col2:
    # Statistics
    st.subheader("📊 Statistics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Active", st.session_state.stats['active_tracks'])
        st.metric("Violations", st.session_state.stats['violations'])
    with col_b:
        st.metric("Alerts", st.session_state.stats['total_alerts'])
    
    st.divider()
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    if st.session_state.alert_manager and len(st.session_state.alert_manager.alerts) > 0:
        recent = st.session_state.alert_manager.get_recent_alerts(5)
        for alert in reversed(recent):
            ts = alert.timestamp.strftime("%H:%M:%S")
            if alert.alert_type.value == "Zone Violation":
                st.error(f"🚨 [{ts}] {alert.message}")
            else:
                st.info(f"ℹ️ [{ts}] {alert.message}")
    else:
        st.info("No alerts")
    
    st.divider()
    
    # Heatmap
    st.subheader("🔥 Heatmap")
    if os.path.exists("heatmap_output.png"):
        # FIXED: Changed use_column_width to use_container_width
        st.image("heatmap_output.png", use_container_width=True)
    else:
        st.info("Run and stop surveillance to generate")
    
    # Downloads
    st.divider()
    st.subheader("📥 Downloads")
    
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json") as f:
            st.download_button(
                "📄 Download Alerts Log", 
                f.read(), 
                "alerts.json",
                use_container_width=True
            )
    
    if os.path.exists("heatmap_output.png"):
        with open("heatmap_output.png", "rb") as f:
            st.download_button(
                "🖼️ Download Heatmap", 
                f.read(), 
                "heatmap.png",
                mime="image/png",
                use_container_width=True
            )

# Auto-refresh (only if enabled)
if st.session_state.running and auto_refresh:
    time.sleep(0.033)  # 30 FPS
    st.rerun()
