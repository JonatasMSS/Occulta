from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
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
        description="Anonimiza faces de um vídeo com base em um rosto-alvo."
    )
    parser.add_argument("--video", type=Path, required=True, help="Vídeo de entrada.")
    parser.add_argument("--target", type=Path, required=True, help="Foto frontal alvo.")
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
    video_path: Path, target_path: Path, model_path: Path, output_path: Path
) -> None:
    for label, path in (
        ("Vídeo", video_path),
        ("Target", target_path),
        ("Modelo", model_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} não encontrado: {path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("O vídeo de saída não pode sobrescrever o vídeo de entrada.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg não foi encontrado no PATH.")


def _video_properties(capture: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("O vídeo não possui resolução/FPS válidos.")
    return width, height, fps, total


def _create_detector(device: torch.device, det_size: int) -> FaceAnalysis:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=providers,
    )
    detector.prepare(
        ctx_id=torch.cuda.current_device() if device.type == "cuda" else -1,
        det_size=(det_size, det_size),
    )
    return detector


def _align(frame, landmarks):
    return face_align.norm_crop(frame, landmark=landmarks, image_size=112)


def _target_embedding(target_path, detector, embedding_handler):
    image = cv2.imread(str(target_path))
    if image is None:
        raise ValueError(f"Não foi possível ler a foto-alvo: {target_path}")

    context = FrameContext(frame_index=0, frame=image)
    FaceDetectionHandler(detector, process_every=1).process(context)
    if len(context.faces) != 1:
        raise ValueError(
            "A foto-alvo deve conter exatamente uma face detectável; "
            f"foram encontradas {len(context.faces)}."
        )
    embedding_handler.process(context)
    embedding = context.faces[0].embedding
    if embedding is None:
        detail = context.faces[0].error or "erro desconhecido"
        raise ValueError(f"Não foi possível gerar o embedding do target: {detail}")
    context.faces[0].embedding = None
    return embedding.detach().clone()


def run(
    video_path: Path,
    target_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.40,
    process_every: int = 3,
    det_size: int = 416,
    debug: bool = False,
    blur_target: bool = False,
) -> None:
    _validate_paths(video_path, target_path, model_path, output_path)
    if process_every < 1 or det_size < 1:
        raise ValueError("process_every e det_size devem ser maiores ou iguais a 1.")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre -1 e 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}; detector: buffalo_l {det_size}x{det_size}")
    detector = _create_detector(device, det_size)
    model = load_adaface(model_path, device)
    embedding_handler = FaceEmbeddingHandler(model, device, _align)
    target_embedding = _target_embedding(target_path, detector, embedding_handler)

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
            SimilarityHandler(target_embedding, threshold)
        ).set_next(
            BlurHandler(threshold=threshold, debug=debug, blur_target=blur_target)
        ).set_next(writer)

        frame_index = 0
        with tqdm(total=total or None, desc="Anonimizando", unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                first.handle(FrameContext(frame_index=frame_index, frame=frame))
                frame_index += 1
                progress.update(1)
        writer.close()
        succeeded = True
    finally:
        capture.release()
        if writer is not None and not succeeded:
            writer.abort()

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
    )
    return 0
