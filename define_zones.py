"""Standalone tool to define restricted zones on a video frame."""

import argparse
import cv2
from zones.zone_drawer import ZoneDrawer
from zones.zone_manager import ZoneManager
from config import video_config, zone_config


def parse_args():
    p = argparse.ArgumentParser(description="Define restricted zones for surveillance")
    p.add_argument('--source', type=str, default=str(video_config.source),
                   help='Video source (0 for webcam, or video file path)')
    p.add_argument('--output', type=str, default=zone_config.zones_file,
                   help='Output JSON file to save zones')
    return p.parse_args()


def main():
    args = parse_args()
    
    # Try to convert source to integer if it's a number
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Try multiple camera backends and indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any Available")
    ]
    
    cap = None
    for backend, backend_name in backends:
        print(f"Trying {backend_name} backend...")
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, backend)
        else:
            cap = cv2.VideoCapture(source)
        
        if cap.isOpened():
            print(f"✓ Successfully opened camera with {backend_name}")
            break
        cap.release()
    
    # If still not opened, try different camera indices
    if cap is None or not cap.isOpened():
        print("\nTrying different camera indices...")
        for idx in [0, 1, 2]:
            print(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"✓ Successfully opened camera at index {idx}")
                source = idx
                break
            cap.release()
    
    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {args.source}\n"
            "Troubleshooting:\n"
            "1. Check if your webcam is connected and not used by another app\n"
            "2. Try: python define_zones.py --source 1\n"
            "3. Or use a video file: python define_zones.py --source path/to/video.mp4"
        )
    
    # SET THE SAME RESOLUTION AS MAIN.PY (ADD THESE LINES)
    if video_config.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.width)
    if video_config.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.height)
    
    # Read frame
    ret, frame = cap.read()
    retry = 0
    while (not ret or frame is None) and retry < 10:
        ret, frame = cap.read()
        retry += 1
    
    cap.release()
    
    if not ret or frame is None:
        raise RuntimeError("Could not read frame from video source")
    
    # Flip the frame to match main.py
    frame = cv2.flip(frame, 1)
    
    print(f"\n✓ Captured frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]} (matches main.py)")
    
    # Start zone drawing
    drawer = ZoneDrawer(frame)
    zones_data = drawer.draw()
    
    if len(zones_data) == 0:
        print("No zones defined. Exiting.")
        return
    
    # Save zones
    zone_manager = ZoneManager()
    for name, points in zones_data:
        zone_manager.add_zone(name, points)
    
    zone_manager.save_zones(args.output)
    print(f"\n✓ Zones saved to: {args.output}")


if __name__ == '__main__':
    main()
