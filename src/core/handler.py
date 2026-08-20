from abc import ABC, abstractmethod
from typing import Any


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Handler | None = None

    def set_next(self, handler: "Handler") -> "Handler":
        self._next = handler
        return handler

    def handle(self, context: Any) -> Any:
        self.process(context)
        return self._next.handle(context) if self._next else context

    @abstractmethod
    def process(self, context: Any) -> None:
        """Process a context or raise PipelineError."""
