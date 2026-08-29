import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from face_blur import FaceResult, FrameContext
from face_blur.app import (
    MAX_TARGETS,
    _resolve_target_paths,
    _target_embeddings,
    build_parser,
    create_detector,
    select_device,
)
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


class RuntimeTests(unittest.TestCase):
    def test_device_selects_cuda_automatically(self):
        with mock.patch("face_blur.app.torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device().type, "cuda")

    def test_detector_uses_cuda_provider_when_available(self):
        analysis = mock.Mock()
        with (
            mock.patch(
                "face_blur.app.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            mock.patch("face_blur.app.FaceAnalysis", return_value=analysis) as factory,
            mock.patch("face_blur.app.torch.cuda.current_device", return_value=2),
        ):
            create_detector(torch.device("cuda"), 416)

        factory.assert_called_once_with(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        analysis.prepare.assert_called_once_with(ctx_id=2, det_size=(416, 416))

    def test_detector_falls_back_cleanly_when_cuda_provider_is_absent(self):
        analysis = mock.Mock()
        with (
            mock.patch(
                "face_blur.app.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            mock.patch("face_blur.app.FaceAnalysis", return_value=analysis) as factory,
        ):
            create_detector(torch.device("cuda"), 416)

        self.assertEqual(
            factory.call_args.kwargs["providers"], ["CPUExecutionProvider"]
        )
        analysis.prepare.assert_called_once_with(ctx_id=-1, det_size=(416, 416))


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

    def test_multiple_targets_use_one_model_batch(self):
        model = FakeModel()
        detector = FakeDetector([fake_face()])
        handler = FaceEmbeddingHandler(
            model,
            torch.device("cpu"),
            lambda frame, landmarks: np.zeros((112, 112, 3), dtype=np.uint8),
        )

        with (
            mock.patch(
                "face_blur.app.cv2.imread",
                return_value=np.zeros((100, 100, 3), dtype=np.uint8),
            ),
            mock.patch("face_blur.app.tqdm", side_effect=lambda items, **kwargs: items),
        ):
            embeddings, names = _target_embeddings(
                [Path("ana.jpg"), Path("bia.png")], detector, handler
            )

        self.assertEqual(names, ["ana.jpg", "bia.png"])
        self.assertEqual(tuple(embeddings.shape), (2, 2))
        self.assertEqual(detector.calls, 2)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.batch_shapes, [(2, 3, 112, 112)])

    def test_invalid_target_aborts_before_embedding(self):
        model = FakeModel()
        handler = FaceEmbeddingHandler(
            model,
            torch.device("cpu"),
            lambda frame, landmarks: frame,
        )

        with (
            mock.patch(
                "face_blur.app.cv2.imread",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ),
            mock.patch("face_blur.app.tqdm", side_effect=lambda items, **kwargs: items),
        ):
            with self.assertRaisesRegex(ValueError, "ruim.jpg.*exatamente uma face"):
                _target_embeddings([Path("ruim.jpg")], FakeDetector([]), handler)

        self.assertEqual(model.calls, 0)

    def test_unreadable_target_reports_its_path(self):
        handler = FaceEmbeddingHandler(
            FakeModel(),
            torch.device("cpu"),
            lambda frame, landmarks: frame,
        )

        with (
            mock.patch("face_blur.app.cv2.imread", return_value=None),
            mock.patch("face_blur.app.tqdm", side_effect=lambda items, **kwargs: items),
        ):
            with self.assertRaisesRegex(ValueError, "ilegivel.jpg"):
                _target_embeddings(
                    [Path("ilegivel.jpg")], FakeDetector([fake_face()]), handler
                )


class SimilarityHandlerTests(unittest.TestCase):
    def test_scores_threshold_and_embedding_discard(self):
        faces = [
            FaceResult((0, 0, 5, 5), LANDMARKS.copy(), embedding=torch.tensor([1.0, 0.0])),
            FaceResult((5, 5, 9, 9), LANDMARKS.copy(), embedding=torch.tensor([0.4, 0.916515])),
            FaceResult((1, 1, 3, 3), LANDMARKS.copy(), embedding=None),
        ]
        context = FrameContext(0, np.zeros((10, 10, 3), dtype=np.uint8), faces=faces)

        SimilarityHandler(
            target_embedding=torch.tensor([1.0, 0.0]), threshold=0.40
        ).process(context)

        self.assertAlmostEqual(faces[0].similarity, 1.0)
        self.assertTrue(faces[0].is_target)
        self.assertAlmostEqual(faces[1].similarity, 0.4, places=5)
        self.assertTrue(faces[1].is_target)
        self.assertIsNone(faces[2].similarity)
        self.assertFalse(faces[2].is_target)
        self.assertTrue(all(face.embedding is None for face in faces))

    def test_multiple_targets_keep_best_name_and_deterministic_tie(self):
        diagonal = 2 ** -0.5
        faces = [
            FaceResult((0, 0, 5, 5), LANDMARKS.copy(), embedding=torch.tensor([0.0, 1.0])),
            FaceResult(
                (5, 5, 9, 9),
                LANDMARKS.copy(),
                embedding=torch.tensor([diagonal, diagonal]),
            ),
        ]
        context = FrameContext(0, np.zeros((10, 10, 3), dtype=np.uint8), faces=faces)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        SimilarityHandler(
            targets, threshold=0.80, target_names=["ana.jpg", "bia.jpg"]
        ).process(context)

        self.assertEqual(faces[0].matched_target, "bia.jpg")
        self.assertAlmostEqual(faces[0].similarity, 1.0)
        self.assertTrue(faces[0].is_target)
        self.assertEqual(faces[1].matched_target, "ana.jpg")
        self.assertAlmostEqual(faces[1].similarity, diagonal)
        self.assertFalse(faces[1].is_target)
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
                FaceResult(
                    (5, 5, 25, 25),
                    LANDMARKS.copy(),
                    similarity=0.8,
                    matched_target="ana.jpg",
                ),
                FaceResult(
                    (35, 35, 55, 55),
                    LANDMARKS.copy(),
                    similarity=0.2,
                    matched_target="ana.jpg",
                ),
                FaceResult((65, 65, 85, 85), LANDMARKS.copy(), similarity=None),
            ],
        )

        with mock.patch("face_blur.handlers.blur.cv2.putText") as put_text:
            BlurHandler(debug=True, blur_target=True).process(context)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(
            labels,
            [
                "TARGET ana.jpg BLUR 0.800",
                "NON_TARGET best=ana.jpg KEEP 0.200",
                "UNKNOWN BLUR inválido",
            ],
        )


class TargetPathTests(unittest.TestCase):
    def test_directory_filters_top_level_images_and_sorts_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("bia.PNG", "ana.jpg", "carla.webp", "ignorar.txt"):
                (root / name).touch()
            nested = root / "subpasta"
            nested.mkdir()
            (nested / "daniela.jpeg").touch()

            paths = _resolve_target_paths(None, root)

        self.assertEqual(
            [path.name for path in paths], ["ana.jpg", "bia.PNG", "carla.webp"]
        )

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
                _resolve_target_paths(None, Path(temp_dir))

    def test_more_than_maximum_targets_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(MAX_TARGETS + 1):
                (root / f"target_{index:02}.jpg").touch()

            with self.assertRaisesRegex(ValueError, "máximo suportado"):
                _resolve_target_paths(None, root)


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

    def test_targets_directory_is_accepted(self):
        args = build_parser().parse_args(
            [
                "--video",
                "video.mp4",
                "--targets-dir",
                "targets",
                "--output",
                "out.mp4",
            ]
        )

        self.assertIsNone(args.target)
        self.assertEqual(args.targets_dir, Path("targets"))

    def test_target_and_directory_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._args("--targets-dir", "targets")

    def test_one_target_source_is_required(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--video", "video.mp4", "--output", "out.mp4"]
            )


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
