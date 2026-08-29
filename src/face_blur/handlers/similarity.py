from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..pipeline import FrameContext, Handler


class SimilarityHandler(Handler):
    def __init__(
        self,
        target_embedding: torch.Tensor,
        threshold: float = 0.40,
        target_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold deve estar entre -1 e 1.")
        target_embeddings = target_embedding
        if target_embeddings.ndim == 1:
            target_embeddings = target_embeddings.unsqueeze(0)
        if target_embeddings.ndim != 2 or target_embeddings.shape[0] == 0:
            raise ValueError("target_embeddings deve ter shape 512 ou T×512.")
        if target_names is None:
            target_names = [f"target_{index + 1}" for index in range(len(target_embeddings))]
        if len(target_names) != len(target_embeddings):
            raise ValueError("A quantidade de nomes deve corresponder aos embeddings.")
        self.target_embeddings = F.normalize(target_embeddings, dim=1)
        self.target_names = list(target_names)
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
            targets = self.target_embeddings.to(batch.device)
            with torch.inference_mode():
                scores, matched_indexes = (batch @ targets.T).max(dim=1)
            for face_index, score, matched_index in zip(
                valid_indexes, scores, matched_indexes
            ):
                value = float(score.item())
                face = context.faces[face_index]
                face.similarity = value
                face.is_target = value >= self.threshold
                face.matched_target = self.target_names[int(matched_index.item())]

        for face in context.faces:
            face.embedding = None
            if face.similarity is None:
                face.is_target = False
                face.matched_target = None
        return True
