from typing import List, Dict, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """Wrapper around deep-sort-realtime to track person detections.

    update(detections, frame) -> List[Dict]: returns list of tracks with id and bbox.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.2) -> None:
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance)

    def update(self, detections: List[Tuple[int, int, int, int, float]], frame) -> List[Dict]:
        # Convert detections to expected format: [ [x1,y1,x2,y2], conf, class ]
        ds_dets = [([x1, y1, x2, y2], conf, 0) for x1, y1, x2, y2, conf in detections]
        tracks = self.tracker.update_tracks(ds_dets, frame=frame)
        output = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            output.append({
                "track_id": t.track_id,
                "bbox": (x1, y1, x2, y2),
            })
        return output
