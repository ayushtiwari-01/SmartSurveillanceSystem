from typing import List, Tuple
import numpy as np

from ultralytics import YOLO


class YoloPersonDetector:
    """YOLOv8-based person detector returning person bounding boxes.

    detect(frame) -> List[Tuple[x1, y1, x2, y2, conf]]
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.4, iou_threshold: float = 0.5) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(source=frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        people: List[Tuple[int, int, int, int, float]] = []
        if not results:
            return people
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return people
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if k == 0:  # person class
                people.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return people
