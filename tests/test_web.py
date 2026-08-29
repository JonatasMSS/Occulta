import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from face_blur.app import MAX_TARGETS, process_video
from face_blur.handlers import FaceEmbeddingHandler
from face_blur.pipeline import FaceResult
from face_blur.web import (
    SelectedTarget,
    find_face_at,
    prepare_target,
    read_frame,
    render_detection_frame,
    toggle_target,
)


LANDMARKS = np.array(
    [[30, 35], [70, 35], [50, 55], [35, 75], [65, 75]], dtype=np.float32
)


class FakeModel(torch.nn.Module):
    def forward(self, batch):
        result = torch.zeros((batch.shape[0], 2), device=batch.device)
        result[:, 0] = 1.0
        return result


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.set_calls = []
        self.read_calls = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        self.read_calls += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 6,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 2,
        }.get(prop, 0)

    def release(self):
        self.released = True


class FakeDetector:
    def get(self, frame):
        return []


class FakeSink:
    def __init__(self, *args):
        self.frames = []
        self.aborted = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        pass

    def abort(self):
        self.aborted = True


def selected(name):
    return SelectedTarget(
        name=name,
        frame_index=0,
        bbox=(0, 0, 2, 2),
        aligned_rgb=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=torch.tensor([1.0, 0.0]),
        normalized_min=-1.0,
        normalized_max=1.0,
    )


class PreviewTests(unittest.TestCase):
    def test_read_frame_seeks_and_reads_only_requested_frame(self):
        capture = FakeCapture([np.ones((6, 8, 3), dtype=np.uint8)])

        with mock.patch("face_blur.web.cv2.VideoCapture", return_value=capture):
            frame = read_frame(Path("video.mp4"), 30)

        self.assertEqual(capture.set_calls, [(cv2.CAP_PROP_POS_FRAMES, 30)])
        self.assertEqual(capture.read_calls, 1)
        self.assertTrue(capture.released)
        self.assertEqual(frame.shape, (6, 8, 3))

    def test_hit_test_chooses_smallest_overlapping_box(self):
        boxes = [(0, 0, 100, 100), (25, 25, 50, 50)]

        self.assertEqual(find_face_at(30, 30, boxes), 1)
        self.assertIsNone(find_face_at(110, 110, boxes))

    def test_detection_frame_scales_boxes(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        face = FaceResult((20, 10, 100, 90), LANDMARKS.copy())

        image, boxes = render_detection_frame(frame, [face], [], 0, max_width=100)

        self.assertEqual(image.shape, (50, 100, 3))
        self.assertEqual(boxes, [(10, 5, 50, 45)])


class TargetSelectionTests(unittest.TestCase):
    def test_targets_accumulate_and_click_again_removes(self):
        targets = []

        self.assertEqual(toggle_target(targets, selected("frame_1_face_1")), "added")
        self.assertEqual(toggle_target(targets, selected("frame_2_face_1")), "added")
        self.assertEqual(len(targets), 2)
        self.assertEqual(toggle_target(targets, selected("frame_1_face_1")), "removed")
        self.assertEqual([target.name for target in targets], ["frame_2_face_1"])

    def test_target_limit_is_enforced(self):
        targets = [selected(f"target_{index}") for index in range(MAX_TARGETS)]

        with self.assertRaisesRegex(ValueError, str(MAX_TARGETS)):
            toggle_target(targets, selected("one_more"))

    def test_selected_target_uses_shared_preprocessing(self):
        handler = FaceEmbeddingHandler(
            FakeModel(),
            torch.device("cpu"),
            lambda frame, landmarks: np.zeros((112, 112, 3), dtype=np.uint8),
        )
        face = FaceResult((1, 2, 20, 30), LANDMARKS.copy())

        target = prepare_target(
            np.zeros((40, 40, 3), dtype=np.uint8), face, 30, 1, handler
        )

        self.assertEqual(target.name, "frame_000030_face_02")
        self.assertEqual(tuple(target.embedding.shape), (2,))
        self.assertEqual(target.normalized_min, -1.0)
        self.assertEqual(target.normalized_max, -1.0)


class SharedProcessingTests(unittest.TestCase):
    def test_progress_reports_each_processed_frame(self):
        frames = [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(2)]
        capture = FakeCapture(frames)
        sink = FakeSink()
        progress = []
        embedding_handler = FaceEmbeddingHandler(
            FakeModel(), torch.device("cpu"), lambda frame, landmarks: frame
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.touch()
            output_path = Path(temp_dir) / "out.mp4"
            with (
                mock.patch("face_blur.app.cv2.VideoCapture", return_value=capture),
                mock.patch("face_blur.app.FFmpegVideoSink", return_value=sink),
                mock.patch("face_blur.app.shutil.which", return_value="ffmpeg"),
            ):
                process_video(
                    video_path,
                    output_path,
                    FakeDetector(),
                    embedding_handler,
                    torch.tensor([[1.0, 0.0]]),
                    ["target"],
                    progress_callback=lambda done, total: progress.append((done, total)),
                )

        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(len(sink.frames), 2)

    def test_partial_output_is_removed_after_failure(self):
        class BrokenSink(FakeSink):
            def write(self, frame):
                raise RuntimeError("falha simulada")

        capture = FakeCapture([np.zeros((6, 8, 3), dtype=np.uint8)])
        sink = BrokenSink()
        embedding_handler = FaceEmbeddingHandler(
            FakeModel(), torch.device("cpu"), lambda frame, landmarks: frame
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.touch()
            output_path = Path(temp_dir) / "out.mp4"
            output_path.write_bytes(b"partial")
            with (
                mock.patch("face_blur.app.cv2.VideoCapture", return_value=capture),
                mock.patch("face_blur.app.FFmpegVideoSink", return_value=sink),
                mock.patch("face_blur.app.shutil.which", return_value="ffmpeg"),
            ):
                with self.assertRaisesRegex(RuntimeError, "falha simulada"):
                    process_video(
                        video_path,
                        output_path,
                        FakeDetector(),
                        embedding_handler,
                        torch.tensor([[1.0, 0.0]]),
                        ["target"],
                    )

            self.assertFalse(output_path.exists())
            self.assertTrue(sink.aborted)


class StreamlitSmokeTests(unittest.TestCase):
    def test_page_loads_without_video_or_model(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app = AppTest.from_file(app_path).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Anonimização seletiva de rostos")
        self.assertIn("Envie um vídeo", app.info[0].value)
        self.assertEqual(app.sidebar.radio[0].value, "Manter targets e borrar outros")
        self.assertEqual(app.sidebar.slider[0].value, 0.40)
        self.assertTrue(app.button[0].disabled)



if __name__ == "__main__":
    unittest.main()
