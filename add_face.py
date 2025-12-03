import argparse
import cv2

from utils.warnings_control import suppress_pkg_resources_warnings
suppress_pkg_resources_warnings()

from db.database import Database
from recognition.facenet_recognizer import FaceNetRecognizer
from config import database_config, recognition_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, type=str)
    p.add_argument('--image', required=True, type=str)
    return p.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    recognizer = FaceNetRecognizer(image_size=recognition_config.face_detection_image_size, match_threshold=recognition_config.face_match_threshold)
    emb, bbox = recognizer.compute_embedding(img)
    if emb is None:
        raise RuntimeError("No face detected in the provided image")

    db = Database(database_config.path)
    pid = db.add_person_with_embedding(args.name, emb)
    print(f"Added/updated person '{args.name}' with id {pid}")


if __name__ == '__main__':
    main()
