import argparse
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save', type=str, default=video_config.save_output_path,
                   help='Optional path to save annotated video')
    p.add_argument('--zones', type=str, default=zone_config.zones_file,
                   help='Path to zones JSON file')
    p.add_argument('--show-heatmap', action='store_true',
                   help='Show real-time heatmap overlay')
    return p.parse_args()


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union


def main():
    args = parse_args()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if video_config.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.width)
    if video_config.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.height)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open default webcam (index 0)")
    
    ok, frame = cap.read()
    retry = 0
    while (not ok or frame is None or frame.size == 0) and retry < 10:
        ok, frame = cap.read()
        retry += 1
    
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("Could not read initial frame from webcam")
    
    # Initialize components
    print("Loading models...")
    detector = YoloPersonDetector(detection_config.model_name,
                                 detection_config.conf_threshold,
                                 detection_config.iou_threshold)
    
    tracker = DeepSortTracker(tracking_config.max_age,
                             tracking_config.n_init,
                             tracking_config.max_cosine_distance)
    
    recognizer = FaceNetRecognizer(image_size=recognition_config.face_detection_image_size,
                                  match_threshold=recognition_config.face_match_threshold)
    
    db = Database(database_config.path)
    labels, embs = db.get_all_embeddings()
    
    # Convert list of embeddings to numpy array
    if len(embs) > 0:
        embs = np.array(embs, dtype=np.float32)
    else:
        embs = None
    
    recognizer.set_known(labels, embs)
    print(f"Loaded {len(labels)} known person(s) from database")
    
    # Initialize zone manager
    zone_manager = ZoneManager()
    zone_manager.load_zones(args.zones)
    print(f"Loaded {len(zone_manager.zones)} restricted zone(s)")
    
    # Initialize email notifier
    notifier = None
    if notification_config.enabled:
        notifier = EmailNotifier(
            smtp_server='smtp.gmail.com',
            smtp_port=587,
            sender_email=notification_config.sender_email,
            sender_password=notification_config.sender_password,
            recipient_email=notification_config.recipient_email
        )
        print(f"✓ Email notifications enabled -> {notification_config.recipient_email}")
    else:
        print("⚠ Email notifications disabled (check .env file)")
    
    # Initialize alert manager with notifier
    alert_manager = AlertManager(notifier=notifier)
    
    # Initialize heatmap generator
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heatmap = HeatmapGenerator((frame_height, frame_width), decay_factor=0.995)
    print("Heatmap tracking enabled")
    
    # Video writer setup
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
    
    # Track state
    id_to_name = {}
    id_to_match_score = {}
    active_tracks = set()
    show_heatmap = args.show_heatmap
    
    frame_idx = 0
    
    print("\nStarting surveillance... Press ESC to quit, 'h' to toggle heatmap\n")
    
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        
        frame = cv2.flip(frame, 1)  # Flip horizontally
        
        # Detection
        people = detector.detect(frame)
        
        # Tracking
        tracks = tracker.update(people, frame)
        
        # Update heatmap
        heatmap.update(tracks)
        
        # Track current active track IDs
        current_tracks = {t['track_id'] for t in tracks}
        
        # Clear violations for lost tracks
        lost_tracks = active_tracks - current_tracks
        for tid in lost_tracks:
            alert_manager.clear_track_violations(tid)
            zone_manager.clear_violations(tid)
        
        active_tracks = current_tracks
        
        # Face recognition (sampled)
        do_face = (frame_idx % 10 == 0)
        
        for t in tracks:
            tid = t['track_id']
            x1, y1, x2, y2 = t['bbox']
            
            # Find best matching detection confidence
            best_conf = None
            for dx1, dy1, dx2, dy2, dconf in people:
                if iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2)) > 0.3:
                    if best_conf is None or dconf > best_conf:
                        best_conf = dconf
            
            # Face recognition
            if do_face:
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size != 0:
                    emb, face_box = recognizer.compute_embedding(crop)
                    if emb is not None:
                        match_result = recognizer.match(emb)
                        
                        if match_result is not None:
                            if len(match_result) == 3:
                                pid, name, score = match_result
                            elif len(match_result) == 2:
                                name, score = match_result
                                pid = None
                            else:
                                name = match_result
                                score = None
                                pid = None
                            
                            if name is not None:
                                if tid not in id_to_name or id_to_name[tid] != name:
                                    score_str = f"{score:.3f}" if score is not None else "N/A"
                                    print(f"NEW MATCH: Track {tid} -> {name} (score={score_str})")
                                id_to_name[tid] = name
                                if score is not None:
                                    id_to_match_score[tid] = score
            
            # Zone violation check
            person_name = id_to_name.get(tid)
            violated_zones = zone_manager.check_violations(tid, (x1, y1, x2, y2))
            
            if len(violated_zones) > 0:
                for zone in violated_zones:
                    alert_manager.create_zone_violation_alert(tid, zone.name, person_name)
                bbox_color = (0, 0, 255)
            else:
                bbox_color = (0, 255, 0)
            
            # Build label
            label = f"ID {tid}"
            if tid in id_to_name:
                label += f" | {id_to_name[tid]}"
            if tid in id_to_match_score:
                label += f" | sim:{id_to_match_score[tid]:.2f}"
            if best_conf is not None:
                label += f" | det:{best_conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        # Apply heatmap overlay if enabled
        if show_heatmap:
            frame = heatmap.generate_heatmap_overlay(frame, alpha=0.4)
        
        # Draw zones
        frame = zone_manager.draw_zones(frame)
        
        # Draw crowd count
        crowd_count = len(tracks)
        cv2.putText(frame, f"Crowd: {crowd_count}", (12, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Draw recent alerts
        alert_manager.draw_alerts(frame, max_display=3)
        
        # Draw heatmap indicator
        if show_heatmap:
            cv2.putText(frame, "HEATMAP ON", (frame.shape[1] - 180, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save and display
        if writer is not None:
            writer.write(frame)
        
        cv2.imshow("Smart Surveillance - Zone Monitoring", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('h'):  # Toggle heatmap
            show_heatmap = not show_heatmap
            print(f"Heatmap overlay: {'ON' if show_heatmap else 'OFF'}")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    # Save alerts log
    alert_manager.save_alerts("alerts_log.json")
    print(f"\nAlerts saved to alerts_log.json")
    print(f"Total alerts: {len(alert_manager.alerts)}")
    
    # Save heatmap
    heatmap.save_heatmap("heatmap_output.png", frame)
    print(f"Heatmap analyzed {heatmap.frame_count} frames")


if __name__ == '__main__':
    main()
