from __future__ import annotations

from typing import Callable, List, Sequence

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
    def preprocess_aligned(aligned: np.ndarray) -> torch.Tensor:
        if aligned.shape[:2] != (112, 112):
            aligned = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        return (tensor / 255.0 - 0.5) / 0.5

    _to_tensor = preprocess_aligned

    def embed_aligned(self, aligned_faces: Sequence[np.ndarray]) -> torch.Tensor:
        if not aligned_faces:
            raise ValueError("É necessário informar ao menos uma face alinhada.")
        batch = torch.stack(
            [self.preprocess_aligned(face) for face in aligned_faces]
        ).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            return F.normalize(self.model(batch), dim=1)

    def process(self, context: FrameContext) -> bool:
        if not context.analyzed or not context.faces:
            return True

        aligned_faces: List[np.ndarray] = []
        face_indexes: List[int] = []
        for index, face in enumerate(context.faces):
            face.embedding = None
            face.similarity = None
            face.is_target = False
            face.error = None
            face.matched_target = None
            try:
                aligned = self.aligner(context.frame, face.landmarks)
                aligned_faces.append(aligned)
                face_indexes.append(index)
            except Exception as error:  # Uma face ruim não deve liberar sua identidade.
                face.error = f"alignment:{type(error).__name__}"

        if not aligned_faces:
            return True

        try:
            embeddings = self.embed_aligned(aligned_faces)
            for row, face_index in enumerate(face_indexes):
                context.faces[face_index].embedding = embeddings[row]
        except Exception as error:
            for face_index in face_indexes:
                context.faces[face_index].error = f"embedding:{type(error).__name__}"
                context.faces[face_index].embedding = None
        return True
