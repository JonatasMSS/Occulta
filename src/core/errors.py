class PipelineError(Exception):
    """An error that stops pipeline processing at a named stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}: {message}")
