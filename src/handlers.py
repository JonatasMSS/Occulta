from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
from tqdm import tqdm

from src.core import Face, FrameContext, Handler, PipelineContext, PipelineError
from src.face_service import FaceService


class ValidateInputHandler(Handler):
    def process(self, context: PipelineContext) -> None:
        print("Etapa 1/4: validando entradas...")
        if not 0.0 <= context.similarity_threshold <= 1.0:
            raise PipelineError("validation", "similarity threshold must be between 0 and 1")
        if context.detection_interval < 1:
            raise PipelineError("validation", "detection interval must be at least 1")
        if context.output_video.suffix.lower() != ".mp4":
            raise PipelineError("validation", "output video must use the .mp4 extension")
        for label, path in (("reference image", context.reference_image), ("input video", context.input_video)):
            if not path.is_file():
                raise PipelineError("validation", f"{label} does not exist: {path}")
        if shutil.which("ffmpeg") is None:
            raise PipelineError("validation", "ffmpeg was not found on PATH")
        context.output_video.parent.mkdir(parents=True, exist_ok=True)
        if context.debug_dir:
            context.debug_dir.mkdir(parents=True, exist_ok=True)
        context.history.append("validated inputs")


class PrepareReferenceHandler(Handler):
    def process(self, context: PipelineContext) -> None:
        print("Etapa 2/4: preparando rosto de referência...")
        image = cv2.imread(str(context.reference_image))
        if image is None:
            raise PipelineError("reference", "could not read the reference image")

        context.service = FaceService()
        print(f"Backend de inferência: {context.service.device}")
        faces = context.service.detect(image)
        if len(faces) != 1:
            raise PipelineError("reference", f"expected exactly one face, found {len(faces)}")
        context.reference_embedding = context.service.embedding(image, faces[0])

        if context.debug_dir:
            x1, y1, x2, y2 = faces[0].box
            annotated = image.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(annotated, "reference", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)
            cv2.imwrite(str(context.debug_dir / "reference.png"), annotated)
        context.history.append("prepared reference")


class DetectOrTrackFacesHandler(Handler):
    def process(self, context: FrameContext) -> None:
        if context.should_detect:
            context.faces = context.service.detect(context.frame)
            context.detected = True
            return

        tracked_faces = context.service.update_tracks(context.frame, context.tracks)
        if tracked_faces is None:
            context.faces = context.service.detect(context.frame)
            context.detected = True
        else:
            context.faces = tracked_faces


class IdentifyTargetHandler(Handler):
    def process(self, context: FrameContext) -> None:
        if not context.detected or not context.faces:
            return
        for face, candidate in zip(context.faces, context.service.embeddings(context.frame, context.faces), strict=True):
            face.similarity = context.service.similarity(context.reference_embedding, candidate)


class UpdateTracksHandler(Handler):
    def process(self, context: FrameContext) -> None:
        if context.detected:
            context.tracks[:] = context.service.create_tracks(context.frame, context.faces)


class BlurNonTargetFacesHandler(Handler):
    def process(self, context: FrameContext) -> None:
        for face in context.faces:
            if face.similarity is None or face.similarity < context.similarity_threshold:
                context.service.blur(context.frame, face)


class WriteFrameHandler(Handler):
    def process(self, context: FrameContext) -> None:
        context.writer.write(context.frame)


class ProcessVideoHandler(Handler):
    def process(self, context: PipelineContext) -> None:
        print("Etapa 3/4: processando frames...")
        capture = cv2.VideoCapture(str(context.input_video))
        if not capture.isOpened():
            raise PipelineError("video", "could not open input video")

        writer = None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            context.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0 or fps <= 0:
                raise PipelineError("video", "input video has invalid dimensions or FPS")

            context.temporary_video = _temporary_path(context.output_video, "video")
            writer = cv2.VideoWriter(
                str(context.temporary_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise PipelineError("video", "could not create temporary video")

            chain = _build_frame_pipeline()
            with tqdm(
                total=context.total_frames or None,
                desc="Frames",
                unit="frame",
                dynamic_ncols=True,
            ) as progress:
                while True:
                    read, frame = capture.read()
                    if not read:
                        break
                    frame_context = FrameContext(
                        frame=frame,
                        index=context.frames_processed,
                        writer=writer,
                        service=context.service,
                        reference_embedding=context.reference_embedding,
                        similarity_threshold=context.similarity_threshold,
                        should_detect=context.frames_processed % context.detection_interval == 0,
                        tracks=context.tracks,
                    )
                    chain.handle(frame_context)
                    kept = sum(face.similarity is not None and face.similarity >= context.similarity_threshold for face in frame_context.faces)
                    context.faces_kept += kept
                    context.faces_blurred += len(frame_context.faces) - kept
                    context.frames_processed += 1
                    if frame_context.detected:
                        context.frames_detected += 1
                    else:
                        context.frames_tracked += 1
                    progress.update()
                    progress.set_postfix(
                        detected=context.frames_detected,
                        tracked=context.frames_tracked,
                        kept=context.faces_kept,
                        blurred=context.faces_blurred,
                        refresh=False,
                    )

                    if context.debug_dir and frame_context.faces and not context.debug_frame_written:
                        annotated = context.service.annotate(
                            frame_context.frame,
                            frame_context.faces,
                            context.similarity_threshold,
                        )
                        cv2.imwrite(str(context.debug_dir / "first_faces_frame.png"), annotated)
                        context.debug_frame_written = True

            if not context.frames_processed:
                raise PipelineError("video", "input video contains no readable frames")
            context.history.append("processed frames")
        finally:
            capture.release()
            if writer is not None:
                writer.release()


class RemuxAudioHandler(Handler):
    def process(self, context: PipelineContext) -> None:
        print("Etapa 4/4: exportando MP4 e áudio...")
        if not context.temporary_video or not context.temporary_video.is_file():
            raise PipelineError("export", "temporary video was not created")

        context.staging_video = _temporary_path(context.output_video, "final")
        copy_audio = _ffmpeg_command(context, "copy")
        result = _run_ffmpeg(copy_audio)
        if result.returncode != 0:
            context.staging_video.unlink(missing_ok=True)
            result = _run_ffmpeg(_ffmpeg_command(context, "aac"))
        if result.returncode != 0:
            context.staging_video.unlink(missing_ok=True)
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed"
            raise PipelineError("export", detail)

        try:
            context.staging_video.replace(context.output_video)
        except OSError as error:
            raise PipelineError("export", f"could not replace output video: {error}") from error
        context.history.append("exported video")


def _build_frame_pipeline() -> Handler:
    first = DetectOrTrackFacesHandler()
    current = first
    for handler in (IdentifyTargetHandler(), UpdateTracksHandler(), BlurNonTargetFacesHandler(), WriteFrameHandler()):
        current = current.set_next(handler)
    return first


def _temporary_path(output: Path, label: str) -> Path:
    return output.with_name(f".{output.stem}.{label}.{uuid.uuid4().hex}.mp4")


def _ffmpeg_command(context: PipelineContext, audio_codec: str) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(context.temporary_video),
        "-i", str(context.input_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-map_metadata", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", audio_codec,
        "-shortest", "-movflags", "+faststart",
        str(context.staging_video),
    ]


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)
