from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FaceResult:
    """Dados de uma face que atravessam a cadeia de processamento."""

    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray
    detection_score: float = 0.0
    embedding: Optional[Any] = None
    similarity: Optional[float] = None
    is_target: bool = False
    error: Optional[str] = None
    matched_target: Optional[str] = None


@dataclass
class FrameContext:
    """Estado mutável compartilhado por todos os handlers de um frame."""

    frame_index: int
    frame: np.ndarray
    output_frame: Optional[np.ndarray] = None
    analyzed: bool = True
    faces: List[FaceResult] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        self._next = handler
        return handler

    def handle(self, context: FrameContext) -> FrameContext:
        should_continue = self.process(context)
        if should_continue and self._next is not None:
            return self._next.handle(context)
        return context

    @abstractmethod
    def process(self, context: FrameContext) -> bool:
        """Process the frame and return whether the chain should continue."""
