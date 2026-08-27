from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..pipeline import FrameContext, Handler


class FFmpegVideoSink:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command: Sequence[str] = (
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        )
        self.width = width
        self.height = height
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError("O sink FFmpeg já foi fechado.")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame com shape {frame.shape}; esperado "
                f"({self.height}, {self.width}, 3)."
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError("O FFmpeg encerrou durante a escrita do vídeo.") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg encerrou com código {return_code}.")

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        finally:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait()


class VideoWriterHandler(Handler):
    def __init__(self, sink: object) -> None:
        super().__init__()
        self.sink = sink

    def process(self, context: FrameContext) -> bool:
        frame = context.output_frame if context.output_frame is not None else context.frame
        self.sink.write(frame)
        return True

    def close(self) -> None:
        self.sink.close()

    def abort(self) -> None:
        abort = getattr(self.sink, "abort", None)
        if abort is not None:
            abort()
