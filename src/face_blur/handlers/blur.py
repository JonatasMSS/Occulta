from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..pipeline import FaceResult, FrameContext, Handler


class BlurHandler(Handler):
    def __init__(
        self,
        threshold: float = 0.40,
        debug: bool = False,
        margin: float = 0.15,
        blur_target: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.debug = debug
        self.margin = margin
        self.blur_target = blur_target

    def _padded_bbox(
        self, face: FaceResult, shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = face.bbox
        pad_x = int((x2 - x1) * self.margin)
        pad_y = int((y2 - y1) * self.margin)
        height, width = shape[:2]
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    @staticmethod
    def _blur(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return
        kernel = max(31, int(min(roi.shape[:2]) * 0.75))
        if kernel % 2 == 0:
            kernel += 1
        image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    @staticmethod
    def _annotate(
        image: np.ndarray,
        face: FaceResult,
        is_target: bool,
        should_blur: bool,
    ) -> None:
        x1, y1, x2, y2 = face.bbox
        color = (0, 0, 255) if should_blur else (0, 180, 0)
        score = "inválido" if face.similarity is None else f"{face.similarity:.3f}"
        identity = "UNKNOWN" if face.similarity is None else (
            "TARGET" if is_target else "NON_TARGET"
        )
        action = "BLUR" if should_blur else "KEEP"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{identity} {action} {score}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def process(self, context: FrameContext) -> bool:
        output = context.frame.copy()
        for face in context.faces:
            is_target = face.similarity is not None and face.similarity >= self.threshold
            face.is_target = is_target
            should_blur = face.similarity is None or (
                is_target if self.blur_target else not is_target
            )
            if should_blur:
                self._blur(output, self._padded_bbox(face, output.shape))
            if self.debug:
                self._annotate(output, face, is_target, should_blur)
        context.output_frame = output
        return True
