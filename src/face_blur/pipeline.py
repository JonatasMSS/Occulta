from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FrameContext:
    frame_index: int
    frame: np.ndarray
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
