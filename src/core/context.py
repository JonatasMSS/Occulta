from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Face:
    box: tuple[int, int, int, int]
    confidence: float
    similarity: float | None = None


@dataclass
class TrackedFace:
    tracker: Any
    similarity: float
    confidence: float


@dataclass
class FrameContext:
    frame: np.ndarray
    index: int
    writer: Any
    service: Any
    reference_embedding: np.ndarray
    similarity_threshold: float
    should_detect: bool
    tracks: list[TrackedFace]
    faces: list[Face] = field(default_factory=list)
    detected: bool = False


@dataclass
class PipelineContext:
    reference_image: Path
    input_video: Path
    output_video: Path
    similarity_threshold: float = 0.80
    detection_interval: int = 5
    debug_dir: Path | None = None
    service: Any = None
    reference_embedding: np.ndarray | None = None
    temporary_video: Path | None = None
    staging_video: Path | None = None
    total_frames: int = 0
    frames_processed: int = 0
    faces_kept: int = 0
    faces_blurred: int = 0
    frames_detected: int = 0
    frames_tracked: int = 0
    tracks: list[TrackedFace] = field(default_factory=list)
    debug_frame_written: bool = False
    history: list[str] = field(default_factory=list)

    def cleanup_temporary_files(self) -> None:
        for path in (self.temporary_video, self.staging_video):
            if path:
                path.unlink(missing_ok=True)
