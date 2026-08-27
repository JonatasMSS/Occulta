import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from face_blur import FaceResult, FrameContext
from face_blur.app import build_parser
from face_blur.handlers import (
    BlurHandler,
    FaceDetectionHandler,
    FaceEmbeddingHandler,
    SimilarityHandler,
    VideoWriterHandler,
)


LANDMARKS = np.array(
    [[30, 35], [70, 35], [50, 55], [35, 75], [65, 75]], dtype=np.float32
)


def fake_face(x1=10, y1=10, x2=80, y2=90):
    return SimpleNamespace(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        kps=LANDMARKS.copy(),
        det_score=0.95,
    )


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        return self.faces


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_shapes = []

    def forward(self, batch):
        self.calls += 1
        self.batch_shapes.append(tuple(batch.shape))
        result = torch.zeros((batch.shape[0], 2), device=batch.device)
        result[:, 0] = 1.0
        return result


class FakeSink:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class DetectionHandlerTests(unittest.TestCase):
    def test_detector_is_called_only_on_analysis_frames(self):
        detector = FakeDetector([fake_face()])
        handler = FaceDetectionHandler(detector, process_every=3)
        contexts = []

        for index in range(4):
            context = FrameContext(index, np.zeros((100, 100, 3), dtype=np.uint8))
            handler.process(context)
            contexts.append(context)

        self.assertEqual(detector.calls, 2)
        self.assertTrue(contexts[0].analyzed)
        self.assertFalse(contexts[1].analyzed)
        self.assertFalse(contexts[2].analyzed)
        self.assertTrue(contexts[3].analyzed)
        self.assertIs(contexts[1].faces, contexts[0].faces)
        self.assertEqual(contexts[0].faces[0].bbox, (10, 10, 80, 90))


class EmbeddingHandlerTests(unittest.TestCase):
    def test_all_faces_use_one_model_batch(self):
        model = FakeModel()
        align_calls = []

        def align(frame, landmarks):
            align_calls.append(landmarks)
            return np.zeros((112, 112, 3), dtype=np.uint8)

        context = FrameContext(
            0,
            np.zeros((100, 100, 3), dtype=np.uint8),
            faces=[
                FaceResult((0, 0, 40, 40), LANDMARKS.copy()),
                FaceResult((50, 50, 90, 90), LANDMARKS.copy()),
            ],
        )
        handler = FaceEmbeddingHandler(model, torch.device("cpu"), align)

        handler.process(context)

        self.assertEqual(len(align_calls), 2)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.batch_shapes, [(2, 3, 112, 112)])
        self.assertTrue(all(face.embedding is not None for face in context.faces))

    def test_skipped_frame_does_not_call_aligner_or_model(self):
        model = FakeModel()

        def unexpected_align(frame, landmarks):
            self.fail("O alinhador não deveria ser chamado.")

        context = FrameContext(
            1,
            np.zeros((10, 10, 3), dtype=np.uint8),
            analyzed=False,
            faces=[FaceResult((0, 0, 5, 5), LANDMARKS.copy())],
        )

        FaceEmbeddingHandler(model, torch.device("cpu"), unexpected_align).process(context)

        self.assertEqual(model.calls, 0)

    def test_alignment_failure_keeps_face_without_embedding(self):
        def broken_align(frame, landmarks):
            raise ValueError("landmarks ruins")

        face = FaceResult((1, 1, 8, 8), LANDMARKS.copy())
        context = FrameContext(0, np.zeros((10, 10, 3), dtype=np.uint8), faces=[face])

        FaceEmbeddingHandler(
            FakeModel(), torch.device("cpu"), broken_align
        ).process(context)

        self.assertEqual(context.faces[0].bbox, (1, 1, 8, 8))
        self.assertIsNone(context.faces[0].embedding)
        self.assertTrue(context.faces[0].error.startswith("alignment:"))


class SimilarityHandlerTests(unittest.TestCase):
    def test_scores_threshold_and_embedding_discard(self):
        faces = [
            FaceResult((0, 0, 5, 5), LANDMARKS.copy(), embedding=torch.tensor([1.0, 0.0])),
            FaceResult((5, 5, 9, 9), LANDMARKS.copy(), embedding=torch.tensor([0.4, 0.916515])),
            FaceResult((1, 1, 3, 3), LANDMARKS.copy(), embedding=None),
        ]
        context = FrameContext(0, np.zeros((10, 10, 3), dtype=np.uint8), faces=faces)

        SimilarityHandler(torch.tensor([1.0, 0.0]), threshold=0.40).process(context)

        self.assertAlmostEqual(faces[0].similarity, 1.0)
        self.assertTrue(faces[0].is_target)
        self.assertAlmostEqual(faces[1].similarity, 0.4, places=5)
        self.assertTrue(faces[1].is_target)
        self.assertIsNone(faces[2].similarity)
        self.assertFalse(faces[2].is_target)
        self.assertTrue(all(face.embedding is None for face in faces))


class BlurHandlerTests(unittest.TestCase):
    def test_below_threshold_and_invalid_faces_are_blurred(self):
        grid = np.indices((100, 100)).sum(axis=0) % 2 * 255
        frame = np.repeat(grid[:, :, None], 3, axis=2).astype(np.uint8)
        faces = [
            FaceResult((10, 10, 40, 40), LANDMARKS.copy(), similarity=0.39),
            FaceResult((60, 60, 90, 90), LANDMARKS.copy(), similarity=None),
        ]
        context = FrameContext(0, frame, faces=faces)

        BlurHandler(threshold=0.40).process(context)

        self.assertFalse(np.array_equal(context.output_frame[10:40, 10:40], frame[10:40, 10:40]))
        self.assertFalse(np.array_equal(context.output_frame[60:90, 60:90], frame[60:90, 60:90]))

    def test_inverse_mode_blurs_target_and_keeps_non_target(self):
        grid = np.indices((100, 100)).sum(axis=0) % 2 * 255
        frame = np.repeat(grid[:, :, None], 3, axis=2).astype(np.uint8)
        faces = [
            FaceResult((10, 10, 40, 40), LANDMARKS.copy(), similarity=0.40),
            FaceResult((60, 60, 90, 90), LANDMARKS.copy(), similarity=0.39),
        ]
        context = FrameContext(0, frame, faces=faces)

        BlurHandler(threshold=0.40, blur_target=True).process(context)

        self.assertFalse(np.array_equal(context.output_frame[10:40, 10:40], frame[10:40, 10:40]))
        self.assertTrue(np.array_equal(context.output_frame[60:90, 60:90], frame[60:90, 60:90]))
        self.assertTrue(faces[0].is_target)
        self.assertFalse(faces[1].is_target)

    def test_invalid_face_is_blurred_in_inverse_mode(self):
        grid = np.indices((100, 100)).sum(axis=0) % 2 * 255
        frame = np.repeat(grid[:, :, None], 3, axis=2).astype(np.uint8)
        context = FrameContext(
            0,
            frame,
            faces=[FaceResult((20, 20, 80, 80), LANDMARKS.copy(), similarity=None)],
        )

        BlurHandler(blur_target=True).process(context)

        self.assertFalse(np.array_equal(context.output_frame[20:80, 20:80], frame[20:80, 20:80]))

    def test_debug_labels_show_identity_and_action(self):
        context = FrameContext(
            0,
            np.zeros((100, 100, 3), dtype=np.uint8),
            faces=[
                FaceResult((5, 5, 25, 25), LANDMARKS.copy(), similarity=0.8),
                FaceResult((35, 35, 55, 55), LANDMARKS.copy(), similarity=0.2),
                FaceResult((65, 65, 85, 85), LANDMARKS.copy(), similarity=None),
            ],
        )

        with mock.patch("face_blur.handlers.blur.cv2.putText") as put_text:
            BlurHandler(debug=True, blur_target=True).process(context)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(
            labels,
            ["TARGET BLUR 0.800", "NON_TARGET KEEP 0.200", "UNKNOWN BLUR inválido"],
        )


class CliTests(unittest.TestCase):
    @staticmethod
    def _args(*extra):
        return build_parser().parse_args(
            ["--video", "video.mp4", "--target", "target.png", "--output", "out.mp4", *extra]
        )

    def test_blur_target_is_disabled_by_default(self):
        self.assertFalse(self._args().blur_target)

    def test_blur_target_flag_enables_inverse_mode(self):
        self.assertTrue(self._args("--blur-target").blur_target)


class WriterHandlerTests(unittest.TestCase):
    def test_writer_uses_altered_frame(self):
        sink = FakeSink()
        handler = VideoWriterHandler(sink)
        context = FrameContext(
            0,
            np.zeros((2, 2, 3), dtype=np.uint8),
            output_frame=np.ones((2, 2, 3), dtype=np.uint8),
        )

        handler.process(context)
        handler.close()

        self.assertEqual(int(sink.frames[0].sum()), 12)
        self.assertTrue(sink.closed)


if __name__ == "__main__":
    unittest.main()
