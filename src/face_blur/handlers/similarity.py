from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from ..pipeline import FrameContext, Handler


class SimilarityHandler(Handler):
    def __init__(self, target_embedding: torch.Tensor, threshold: float = 0.40) -> None:
        super().__init__()
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold deve estar entre -1 e 1.")
        self.target_embedding = F.normalize(target_embedding.reshape(-1), dim=0)
        self.threshold = threshold

    def process(self, context: FrameContext) -> bool:
        if not context.analyzed:
            return True

        valid_indexes: List[int] = []
        valid_embeddings: List[torch.Tensor] = []
        for index, face in enumerate(context.faces):
            if face.embedding is not None:
                valid_indexes.append(index)
                valid_embeddings.append(face.embedding)

        if valid_embeddings:
            batch = torch.stack(valid_embeddings)
            target = self.target_embedding.to(batch.device)
            with torch.inference_mode():
                scores = batch @ target
            for face_index, score in zip(valid_indexes, scores):
                value = float(score.item())
                face = context.faces[face_index]
                face.similarity = value
                face.is_target = value >= self.threshold

        for face in context.faces:
            face.embedding = None
            if face.similarity is None:
                face.is_target = False
        return True
