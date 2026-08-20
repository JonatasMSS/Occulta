from __future__ import annotations

import os

# RetinaFace 0.0.18 builds models with the legacy Keras API.
# This must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np

from src.core import Face, PipelineError, TrackedFace


class FaceService:
    """Small adapter around RetinaFace detection and DeepFace ArcFace embeddings."""

    def __init__(self) -> None:
        self.device = self._configure_device()

    @staticmethod
    def _configure_device() -> str:
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            return f"GPU ({gpus[0].name})" if gpus else "CPU"
        except Exception:
            return "CPU"

    def detect(self, image: np.ndarray) -> list[Face]:
        try:
            from retinaface import RetinaFace

            detected = RetinaFace.detect_faces(image, threshold=0.7, allow_upscaling=True)
        except Exception as error:
            raise PipelineError("face detection", str(error)) from error

        if not isinstance(detected, dict):
            return []

        height, width = image.shape[:2]
        faces: list[Face] = []
        for value in detected.values():
            area = value.get("facial_area")
            if not area or len(area) != 4:
                continue
            x1, y1, x2, y2 = (int(point) for point in area)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                faces.append(Face((x1, y1, x2, y2), float(value.get("score", 0.0))))
        return faces

    def embeddings(self, image: np.ndarray, faces: list[Face]) -> list[np.ndarray]:
        crops = []
        for face in faces:
            x1, y1, x2, y2 = face.box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise PipelineError("face embedding", "could not crop detected face")
            crops.append(crop)
        try:
            from deepface import DeepFace

            result = DeepFace.represent(
                img_path=crops,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
            )
        except Exception as error:
            raise PipelineError("face embedding", str(error)) from error
        if not result:
            raise PipelineError("face embedding", "ArcFace did not return an embedding")
        return [np.asarray(item[0]["embedding"], dtype=np.float32) for item in result]

    def embedding(self, image: np.ndarray, face: Face) -> np.ndarray:
        return self.embeddings(image, [face])[0]

    @staticmethod
    def create_tracks(image: np.ndarray, faces: list[Face]) -> list[TrackedFace]:
        tracks = []
        for face in faces:
            x1, y1, x2, y2 = face.box
            tracker = cv2.TrackerMIL_create()
            tracker.init(image, (x1, y1, x2 - x1, y2 - y1))
            tracks.append(TrackedFace(tracker, face.similarity or 0.0, face.confidence))
        return tracks

    @staticmethod
    def update_tracks(image: np.ndarray, tracks: list[TrackedFace]) -> list[Face] | None:
        faces = []
        height, width = image.shape[:2]
        for track in tracks:
            ok, (x, y, box_width, box_height) = track.tracker.update(image)
            if not ok:
                return None
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + box_width)), min(height, int(y + box_height))
            if x2 <= x1 or y2 <= y1:
                return None
            faces.append(Face((x1, y1, x2, y2), track.confidence, track.similarity))
        return faces

    @staticmethod
    def similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
        denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
        if denominator == 0:
            raise PipelineError("face comparison", "received a zero-length embedding")
        return float(np.dot(reference, candidate) / denominator)

    @staticmethod
    def blur(image: np.ndarray, face: Face) -> None:
        x1, y1, x2, y2 = face.box
        region = image[y1:y2, x1:x2]
        smallest_side = min(region.shape[:2]) if region.size else 0
        if smallest_side < 3:
            return
        kernel = min(99, max(3, smallest_side // 3))
        kernel += 1 - kernel % 2
        image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)

    @staticmethod
    def annotate(image: np.ndarray, faces: list[Face], threshold: float) -> np.ndarray:
        result = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face.box
            kept = face.similarity is not None and face.similarity >= threshold
            color = (0, 180, 0) if kept else (0, 0, 255)
            label = "target" if kept else "blur"
            if face.similarity is not None:
                label = f"{label} {face.similarity:.2f}"
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return result
