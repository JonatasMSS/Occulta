from __future__ import annotations

from typing import Callable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..pipeline import FrameContext, Handler


class FaceEmbeddingHandler(Handler):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        aligner: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.aligner = aligner

    @staticmethod
    def _to_tensor(aligned: np.ndarray) -> torch.Tensor:
        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        return (tensor / 255.0 - 0.5) / 0.5

    def process(self, context: FrameContext) -> bool:
        if not context.analyzed or not context.faces:
            return True

        tensors: List[torch.Tensor] = []
        face_indexes: List[int] = []
        for index, face in enumerate(context.faces):
            face.embedding = None
            face.similarity = None
            face.is_target = False
            face.error = None
            try:
                aligned = self.aligner(context.frame, face.landmarks)
                tensors.append(self._to_tensor(aligned))
                face_indexes.append(index)
            except Exception as error:  # Uma face ruim não deve liberar sua identidade.
                face.error = f"alignment:{type(error).__name__}"

        if not tensors:
            return True

        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        try:
            with torch.inference_mode():
                embeddings = F.normalize(self.model(batch), dim=1)
            for row, face_index in enumerate(face_indexes):
                context.faces[face_index].embedding = embeddings[row]
        except Exception as error:
            for face_index in face_indexes:
                context.faces[face_index].error = f"embedding:{type(error).__name__}"
                context.faces[face_index].embedding = None
        return True
