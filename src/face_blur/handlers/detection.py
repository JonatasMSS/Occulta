from __future__ import annotations

from typing import List

import numpy as np

from ..pipeline import FaceResult, FrameContext, Handler


class FaceDetectionHandler(Handler):
    def __init__(self, detector: object, process_every: int = 3) -> None:
        super().__init__()
        if process_every < 1:
            raise ValueError("process_every deve ser maior ou igual a 1.")
        self.detector = detector
        self.process_every = process_every
        self._cached_faces: List[FaceResult] = []

    def process(self, context: FrameContext) -> bool:
        context.analyzed = context.frame_index % self.process_every == 0
        if not context.analyzed:
            context.faces = self._cached_faces
            return True

        height, width = context.frame.shape[:2]
        detected = self.detector.get(context.frame)
        faces: List[FaceResult] = []
        for face in detected:
            bbox = np.asarray(face.bbox).round().astype(int)
            x1 = max(0, min(width, int(bbox[0])))
            y1 = max(0, min(height, int(bbox[1])))
            x2 = max(0, min(width, int(bbox[2])))
            y2 = max(0, min(height, int(bbox[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = np.asarray(face.kps, dtype=np.float32).copy()
            faces.append(
                FaceResult(
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    detection_score=float(getattr(face, "det_score", 0.0)),
                )
            )
        self._cached_faces = faces
        context.faces = faces
        return True
