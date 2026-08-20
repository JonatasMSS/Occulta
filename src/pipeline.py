from src.core import Handler
from src.handlers import (
    PrepareReferenceHandler,
    RemuxAudioHandler,
    ProcessVideoHandler,
    ValidateInputHandler,
)


def build_pipeline() -> Handler:
    first = ValidateInputHandler()
    current = first
    for handler in (
        PrepareReferenceHandler(),
        ProcessVideoHandler(),
        RemuxAudioHandler(),
    ):
        current = current.set_next(handler)
    return first
