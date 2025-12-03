from typing import Tuple, Optional
import cv2


def draw_bbox_with_label(frame, bbox: Tuple[int, int, int, int], label: str, color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bg_w = tw + 10
    bg_h = th + 8
    cv2.rectangle(frame, (x1, max(0, y1 - bg_h)), (x1 + bg_w, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def put_hud(frame, crowd_count: int) -> None:
    cv2.putText(frame, f"Crowd: {crowd_count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
