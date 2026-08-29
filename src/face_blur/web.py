from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_image_coordinates import streamlit_image_coordinates

from .app import (
    DEFAULT_MODEL_PATH,
    MAX_TARGETS,
    create_detector,
    create_embedding_handler,
    process_video,
    select_device,
    select_detector_provider,
)
from .handlers import FaceDetectionHandler, FaceEmbeddingHandler
from .pipeline import FaceResult, FrameContext


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int


@dataclass
class SelectedTarget:
    name: str
    frame_index: int
    bbox: Tuple[int, int, int, int]
    aligned_rgb: np.ndarray
    embedding: torch.Tensor
    normalized_min: float
    normalized_max: float


def video_info(video_path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Não foi possível abrir o MP4 enviado.")
    try:
        info = VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()
    if min(info.width, info.height, info.total_frames) <= 0 or info.fps <= 0:
        raise ValueError("O vídeo não possui resolução, FPS ou frames válidos.")
    return info


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Não foi possível abrir o MP4 enviado.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Não foi possível ler o frame {frame_index}.")
    return frame


def detect_faces(frame: np.ndarray, detector: object) -> List[FaceResult]:
    context = FrameContext(frame_index=0, frame=frame)
    FaceDetectionHandler(detector, process_every=1).process(context)
    return sorted(context.faces, key=lambda face: (face.bbox[0], face.bbox[1]))


def target_name(frame_index: int, face_index: int) -> str:
    return f"frame_{frame_index:06d}_face_{face_index + 1:02d}"


def render_detection_frame(
    frame: np.ndarray,
    faces: Sequence[FaceResult],
    selected_names: Sequence[str],
    frame_index: int,
    max_width: int = 900,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    display = cv2.resize(frame, (display_width, display_height))
    display_boxes: List[Tuple[int, int, int, int]] = []
    selected = set(selected_names)

    for index, face in enumerate(faces):
        box = tuple(round(value * scale) for value in face.bbox)
        display_boxes.append(box)
        color = (
            (0, 180, 0)
            if target_name(frame_index, index) in selected
            else (255, 140, 0)
        )
        x1, y1, x2, y2 = box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display,
            f"Face {index + 1}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), display_boxes


def find_face_at(
    x: int, y: int, boxes: Sequence[Tuple[int, int, int, int]]
) -> int | None:
    matches = [
        (index, (x2 - x1) * (y2 - y1))
        for index, (x1, y1, x2, y2) in enumerate(boxes)
        if x1 <= x <= x2 and y1 <= y <= y2
    ]
    return min(matches, key=lambda match: match[1])[0] if matches else None


def toggle_target(
    targets: List[SelectedTarget], target: SelectedTarget
) -> str:
    for index, current in enumerate(targets):
        if current.name == target.name:
            targets.pop(index)
            return "removed"
    if len(targets) >= MAX_TARGETS:
        raise ValueError(f"O limite é de {MAX_TARGETS} targets.")
    targets.append(target)
    return "added"


def prepare_target(
    frame: np.ndarray,
    face: FaceResult,
    frame_index: int,
    face_index: int,
    embedding_handler: FaceEmbeddingHandler,
) -> SelectedTarget:
    aligned = embedding_handler.aligner(frame, face.landmarks)
    normalized = embedding_handler.preprocess_aligned(aligned)
    embedding = embedding_handler.embed_aligned([aligned])[0].detach().cpu()
    return SelectedTarget(
        name=target_name(frame_index, face_index),
        frame_index=frame_index,
        bbox=face.bbox,
        aligned_rgb=cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
        embedding=embedding,
        normalized_min=float(normalized.min().item()),
        normalized_max=float(normalized.max().item()),
    )


@st.cache_resource(scope="session", show_spinner="Carregando AdaFace...")
def _embedding_runtime(model_path: str) -> Tuple[torch.device, FaceEmbeddingHandler]:
    device = select_device()
    return device, create_embedding_handler(Path(model_path), device)


@st.cache_resource(scope="session", show_spinner="Carregando detector...")
def _detector_runtime(device_type: str, det_size: int):
    return create_detector(torch.device(device_type), det_size)


def _clear_video_state() -> None:
    temp_dir = st.session_state.get("temp_dir")
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    for key in (
        "temp_dir",
        "video_path",
        "video_info",
        "preview_key",
        "preview_frame",
        "preview_faces",
        "targets",
        "last_click",
        "result_path",
        "frame_index",
    ):
        st.session_state.pop(key, None)


def _save_upload(uploaded) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="face_blur_"))
    video_path = temp_dir / "input.mp4"
    try:
        with video_path.open("wb") as destination:
            destination.write(uploaded.getbuffer())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    st.session_state.temp_dir = str(temp_dir)
    st.session_state.video_path = str(video_path)
    return video_path


def _current_preview(
    video_path: Path, frame_index: int, det_size: int, detector: object
) -> Tuple[np.ndarray, List[FaceResult]]:
    key = (str(video_path), frame_index, det_size)
    if st.session_state.get("preview_key") != key:
        frame = read_frame(video_path, frame_index)
        st.session_state.preview_key = key
        st.session_state.preview_frame = frame
        st.session_state.preview_faces = detect_faces(frame, detector)
    return st.session_state.preview_frame, st.session_state.preview_faces


def _output_path(original_name: str) -> Path:
    output_dir = Path(os.getenv("FACE_BLUR_OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")
    safe_stem = safe_stem or "video"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{safe_stem}_{stamp}_{uuid4().hex[:8]}.mp4"


def _sidebar_controls() -> Tuple[bool, float, int, int, bool]:
    st.sidebar.header("Configuração")
    mode = st.sidebar.radio(
        "Ação",
        ("Manter targets e borrar outros", "Borrar targets e manter outros"),
    )
    threshold = st.sidebar.slider("Threshold", -1.0, 1.0, 0.40, 0.01)
    process_every = st.sidebar.number_input(
        "Processar a cada X frames", min_value=1, value=3, step=1
    )
    det_size = st.sidebar.select_slider(
        "Tamanho do detector", options=(320, 416, 512, 640), value=416
    )
    debug = st.sidebar.checkbox("Debug no vídeo final", value=False)
    return (
        mode == "Borrar targets e manter outros",
        float(threshold),
        int(process_every),
        int(det_size),
        debug,
    )


def _show_targets(targets: List[SelectedTarget]) -> None:
    st.subheader(f"Targets selecionados ({len(targets)}/{MAX_TARGETS})")
    if not targets:
        st.info("Clique dentro de uma caixa facial para selecionar um target.")
        return
    for target in list(targets):
        st.image(target.aligned_rgb, caption=target.name, width=112)
        st.caption(
            f"frame={target.frame_index} · bbox={target.bbox} · "
            f"shape=3×112×112 · normalizado="
            f"[{target.normalized_min:.3f}, {target.normalized_max:.3f}]"
        )
        if st.button("Remover", key=f"remove_{target.name}"):
            targets.remove(target)
            st.rerun()


def _show_result() -> None:
    result = st.session_state.get("result_path")
    if not result or not Path(result).is_file():
        return
    result_path = Path(result)
    st.subheader("Resultado")
    st.video(str(result_path))
    with result_path.open("rb") as video_file:
        st.download_button(
            "Baixar vídeo",
            data=video_file,
            file_name=result_path.name,
            mime="video/mp4",
        )


def main() -> None:
    st.set_page_config(page_title="Face Blur", layout="wide")
    st.title("Anonimização seletiva de rostos")
    st.caption("Envie um MP4, navegue até um frame e clique nas faces-alvo.")

    blur_target, threshold, process_every, det_size, debug = _sidebar_controls()
    uploaded = st.file_uploader(
        "Vídeo MP4",
        type=("mp4",),
        max_upload_size=500,
        key="video_upload",
        on_change=_clear_video_state,
    )
    if uploaded is None:
        st.info("Envie um vídeo para começar.")
        st.button("Processar vídeo", type="primary", disabled=True)
        return

    try:
        video_path = Path(st.session_state.get("video_path") or _save_upload(uploaded))
        info = st.session_state.get("video_info")
        if info is None:
            info = video_info(video_path)
            st.session_state.video_info = info
        model_path = Path(os.getenv("FACE_BLUR_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        device, embedding_handler = _embedding_runtime(str(model_path))
        detector = _detector_runtime(device.type, det_size)
    except Exception as error:
        st.error(str(error))
        return

    duration = info.total_frames / info.fps
    st.caption(
        f"{info.width}×{info.height} · {info.fps:.2f} FPS · "
        f"{info.total_frames} frames · {duration:.1f} s · "
        f"AdaFace {device} · detector {select_detector_provider(device)}"
    )
    frame_index = st.slider(
        "Frame para selecionar targets",
        min_value=0,
        max_value=info.total_frames - 1,
        value=0,
        key="frame_index",
    )
    targets: List[SelectedTarget] = st.session_state.setdefault("targets", [])

    try:
        frame, faces = _current_preview(video_path, frame_index, det_size, detector)
    except Exception as error:
        st.error(str(error))
        return

    left, right = st.columns((3, 1))
    with left:
        st.subheader(f"Frame {frame_index}")
        selected_names = [target.name for target in targets]
        display, boxes = render_detection_frame(
            frame, faces, selected_names, frame_index
        )
        click = streamlit_image_coordinates(
            display,
            width=display.shape[1],
            key=f"frame_{Path(st.session_state.temp_dir).name}_{frame_index}_{det_size}",
            cursor="pointer",
        )
        if not faces:
            st.warning(
                "Nenhuma face detectada. Escolha outro frame ou aumente o detector."
            )
        if click and click.get("unix_time") != st.session_state.get("last_click"):
            st.session_state.last_click = click.get("unix_time")
            face_index = find_face_at(int(click["x"]), int(click["y"]), boxes)
            if face_index is None:
                st.warning("O clique não ficou dentro de uma caixa facial.")
            else:
                name = target_name(frame_index, face_index)
                existing = next((item for item in targets if item.name == name), None)
                try:
                    candidate = existing or prepare_target(
                        frame,
                        faces[face_index],
                        frame_index,
                        face_index,
                        embedding_handler,
                    )
                    toggle_target(targets, candidate)
                    st.rerun()
                except Exception as error:
                    st.error(f"Não foi possível selecionar {name}: {error}")
    with right:
        _show_targets(targets)

    if st.button("Processar vídeo", type="primary", disabled=not targets):
        output_path = _output_path(uploaded.name)
        progress = st.progress(0.0, text="Preparando processamento...")
        status = st.status("Anonimizando vídeo...", expanded=False)
        last_percent = -1

        def report(processed: int, total: int) -> None:
            nonlocal last_percent
            percent = min(100, round(processed * 100 / total)) if total else 0
            if percent != last_percent:
                last_percent = percent
                progress.progress(
                    percent / 100,
                    text=f"Anonimizando: {processed}/{total or '?'} frames",
                )

        try:
            process_video(
                video_path=video_path,
                output_path=output_path,
                detector=detector,
                embedding_handler=embedding_handler,
                target_embeddings=torch.stack([item.embedding for item in targets]),
                target_names=[item.name for item in targets],
                threshold=threshold,
                process_every=process_every,
                debug=debug,
                blur_target=blur_target,
                progress_callback=report,
            )
            st.session_state.result_path = str(output_path)
            status.update(label="Vídeo concluído", state="complete")
            progress.progress(1.0, text="Processamento concluído.")
        except Exception as error:
            output_path.unlink(missing_ok=True)
            status.update(label="Falha no processamento", state="error", expanded=True)
            st.error(str(error))

    _show_result()
