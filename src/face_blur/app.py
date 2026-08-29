from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm

from .adaface import load_adaface
from .handlers import (
    BlurHandler,
    FaceDetectionHandler,
    FaceEmbeddingHandler,
    FFmpegVideoSink,
    SimilarityHandler,
    VideoWriterHandler,
)
from .pipeline import FrameContext


DEFAULT_MODEL_PATH = Path("models/adaface_ir101/model.pt")
TARGET_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_TARGETS = 32
ProgressCallback = Callable[[int, int], None]


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_detector_provider(device: torch.device) -> str:
    if (
        device.type == "cuda"
        and "CUDAExecutionProvider" in ort.get_available_providers()
    ):
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("o valor deve ser maior ou igual a 1")
    return parsed


def _threshold(value: str) -> float:
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("o threshold deve estar entre -1 e 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anonimiza faces de um vídeo com base em rostos-alvo."
    )
    parser.add_argument("--video", type=Path, required=True, help="Vídeo de entrada.")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--target", type=Path, help="Foto frontal de um target.")
    targets.add_argument(
        "--targets-dir",
        type=Path,
        help="Pasta com uma foto por pessoa-alvo.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Vídeo MP4 de saída.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Checkpoint AdaFace IR-101 (padrão: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument("--threshold", type=_threshold, default=0.40)
    parser.add_argument("--process-every", type=_positive_int, default=3)
    parser.add_argument("--det-size", type=_positive_int, default=416)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--blur-target",
        action="store_true",
        help="Borra o target e mantém as demais faces visíveis.",
    )
    return parser


def _validate_paths(
    video_path: Path, model_path: Path, output_path: Path
) -> None:
    for label, path in (
        ("Vídeo", video_path),
        ("Modelo", model_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} não encontrado: {path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("O vídeo de saída não pode sobrescrever o vídeo de entrada.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg não foi encontrado no PATH.")


def _resolve_target_paths(
    target_path: Optional[Path], targets_dir: Optional[Path]
) -> List[Path]:
    if (target_path is None) == (targets_dir is None):
        raise ValueError("Informe exatamente um entre target_path e targets_dir.")
    if target_path is not None:
        if not target_path.is_file():
            raise FileNotFoundError(f"Target não encontrado: {target_path}")
        return [target_path]

    assert targets_dir is not None
    if not targets_dir.is_dir():
        raise FileNotFoundError(f"Pasta de targets não encontrada: {targets_dir}")
    paths = sorted(
        (
            path
            for path in targets_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TARGET_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not paths:
        raise ValueError(f"Nenhuma imagem de target encontrada em: {targets_dir}")
    if len(paths) > MAX_TARGETS:
        raise ValueError(
            f"A pasta contém {len(paths)} targets; o máximo suportado é {MAX_TARGETS}."
        )
    return paths


def _video_properties(capture: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("O vídeo não possui resolução/FPS válidos.")
    return width, height, fps, total


def create_detector(device: torch.device, det_size: int) -> FaceAnalysis:
    provider = select_detector_provider(device)
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")
    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=providers,
    )
    detector.prepare(
        ctx_id=(
            torch.cuda.current_device()
            if provider == "CUDAExecutionProvider"
            else -1
        ),
        det_size=(det_size, det_size),
    )
    return detector


_create_detector = create_detector


def _align(frame, landmarks):
    return face_align.norm_crop(frame, landmark=landmarks, image_size=112)


def _target_embeddings(target_paths, detector, embedding_handler):
    aligned_faces = []
    names = []
    detection_handler = FaceDetectionHandler(detector, process_every=1)
    for target_path in tqdm(target_paths, desc="Preparando targets", unit="target"):
        image = cv2.imread(str(target_path))
        if image is None:
            raise ValueError(f"Não foi possível ler o target: {target_path}")

        context = FrameContext(frame_index=0, frame=image)
        try:
            detection_handler.process(context)
        except Exception as error:
            raise ValueError(
                f"Não foi possível detectar a face no target {target_path}: "
                f"{type(error).__name__}"
            ) from error
        if len(context.faces) != 1:
            raise ValueError(
                f"O target {target_path} deve conter exatamente uma face; "
                f"foram encontradas {len(context.faces)}."
            )
        try:
            aligned_faces.append(
                embedding_handler.aligner(image, context.faces[0].landmarks)
            )
        except Exception as error:
            raise ValueError(
                f"Não foi possível alinhar o target {target_path}: "
                f"{type(error).__name__}"
            ) from error
        names.append(target_path.name)

    try:
        embeddings = embedding_handler.embed_aligned(aligned_faces)
    except Exception as error:
        raise ValueError(
            f"Não foi possível gerar os embeddings de {len(target_paths)} targets: "
            f"{type(error).__name__}"
        ) from error
    return embeddings.detach().clone(), names


def create_runtime(
    model_path: Path, det_size: int
) -> Tuple[torch.device, FaceAnalysis, FaceEmbeddingHandler]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if det_size < 1:
        raise ValueError("det_size deve ser maior ou igual a 1.")

    device = select_device()
    detector = create_detector(device, det_size)
    return device, detector, create_embedding_handler(model_path, device)


def create_embedding_handler(
    model_path: Path, device: Optional[torch.device] = None
) -> FaceEmbeddingHandler:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    selected_device = device or select_device()
    model = load_adaface(model_path, selected_device)
    return FaceEmbeddingHandler(model, selected_device, _align)


def process_video(
    video_path: Path,
    output_path: Path,
    detector: object,
    embedding_handler: FaceEmbeddingHandler,
    target_embeddings: torch.Tensor,
    target_names: Sequence[str],
    threshold: float = 0.40,
    process_every: int = 3,
    debug: bool = False,
    blur_target: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    if process_every < 1:
        raise ValueError("process_every deve ser maior ou igual a 1.")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre -1 e 1.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("O vídeo de saída não pode sobrescrever o vídeo de entrada.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg não foi encontrado no PATH.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    writer: Optional[VideoWriterHandler] = None
    succeeded = False
    try:
        width, height, fps, total = _video_properties(capture)
        sink = FFmpegVideoSink(video_path, output_path, width, height, fps)
        writer = VideoWriterHandler(sink)

        first = FaceDetectionHandler(detector, process_every)
        first.set_next(embedding_handler).set_next(
            SimilarityHandler(
                target_embeddings.to(embedding_handler.device),
                threshold,
                target_names,
            )
        ).set_next(
            BlurHandler(threshold=threshold, debug=debug, blur_target=blur_target)
        ).set_next(writer)

        frame_index = 0
        if progress_callback is not None:
            progress_callback(0, total)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first.handle(FrameContext(frame_index=frame_index, frame=frame))
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, total)
        writer.close()
        succeeded = True
    finally:
        capture.release()
        if writer is not None and not succeeded:
            writer.abort()
        if not succeeded:
            output_path.unlink(missing_ok=True)

    return output_path


def run(
    video_path: Path,
    target_path: Optional[Path],
    output_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.40,
    process_every: int = 3,
    det_size: int = 416,
    debug: bool = False,
    blur_target: bool = False,
    targets_dir: Optional[Path] = None,
) -> None:
    _validate_paths(video_path, model_path, output_path)
    target_paths = _resolve_target_paths(target_path, targets_dir)
    if process_every < 1 or det_size < 1:
        raise ValueError("process_every e det_size devem ser maiores ou iguais a 1.")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre -1 e 1.")

    device, detector, embedding_handler = create_runtime(model_path, det_size)
    print(
        f"AdaFace: {device}; detector: {select_detector_provider(device)} "
        f"{det_size}x{det_size}"
    )
    target_embeddings, target_names = _target_embeddings(
        target_paths, detector, embedding_handler
    )

    progress: Optional[tqdm] = None

    def report_progress(processed: int, total: int) -> None:
        nonlocal progress
        if progress is None:
            progress = tqdm(total=total or None, desc="Anonimizando", unit="frame")
        progress.update(processed - progress.n)

    try:
        process_video(
            video_path=video_path,
            output_path=output_path,
            detector=detector,
            embedding_handler=embedding_handler,
            target_embeddings=target_embeddings,
            target_names=target_names,
            threshold=threshold,
            process_every=process_every,
            debug=debug,
            blur_target=blur_target,
            progress_callback=report_progress,
        )
    finally:
        if progress is not None:
            progress.close()

    print(f"Vídeo final: {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        video_path=args.video,
        target_path=args.target,
        output_path=args.output,
        model_path=args.model,
        threshold=args.threshold,
        process_every=args.process_every,
        det_size=args.det_size,
        debug=args.debug,
        blur_target=args.blur_target,
        targets_dir=args.targets_dir,
    )
    return 0
