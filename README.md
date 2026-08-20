# Selective Face Anonymization in Video

A Python project that identifies one person in a video from a reference image and blurs only that person's face. The expected output is an MP4 video with the original audio preserved.

> **Status:** under development. At the moment, the repository contains only the initial structure and the V1 plan; video processing has not been implemented yet.

## V1 Goal

Accept a reference image and a video, locate the matching person in the frames, and blur only their face.

## Planned workflow

```text
Reference image
        │
        ▼
Face detection (RetinaFace) ──► Preprocessing ──► Face embedding (FaceNet)
                                                            │
Input video ──► Face detection per frame ───────────────────┤
                                                            ▼
                                                 Similarity comparison
                                                            │
                                                            ▼
                                                   Blur target face
                                                            │
                                                            ▼
                                    Final video + original audio (FFmpeg)
```

## Inputs and output

| Item | Description |
| --- | --- |
| `reference_image` | An image containing exactly one face of the person to anonymize. |
| `input_video` | The video to process. |
| `output_video` | Destination path for the generated anonymized MP4 video. |
| `similarity_threshold` | Configurable value that determines whether a face matches the reference person. |

## Planned requirements

- Python 3.13, or a version compatible with the selected libraries;
- OpenCV;
- NumPy;
- RetinaFace;
- a FaceNet implementation based on **PyTorch** or **TensorFlow** (the choice has not been made yet);
- FFmpeg, to preserve or remux the video audio.

## How to run

The processing interface is still under development. Once it is available, usage should follow this format:

```bash
python -m src.main \
  --reference-image data/reference/person.jpg \
  --input-video data/input/video.mp4 \
  --output-video data/output/anonymized_video.mp4 \
  --similarity-threshold 0.8
```

For now, the existing entry point can be run with:

```bash
python main.py
```

## Planned structure

```text
.
├── Assets/             # Supporting images
├── Notebooks/          # Experiments and exploratory tests
├── main.py             # Initial entry point
├── TODO.md             # V1 plan and acceptance criteria
├── src/                # Processing source code (to be created)
├── tests/              # Automated tests (to be created)
└── data/
    ├── reference/      # Reference images (to be created)
    ├── input/          # Input videos (to be created)
    └── output/         # Processed videos (to be created)
```

## Known limitations

Results may be affected by:

- occluded, very small, or extremely angled faces;
- poor lighting, low resolution, or intense movement;
- multiple similar-looking people in the same video;
- face detection or identification failures;
- input videos with no audio.

## Privacy and responsible use

Use images and videos only with appropriate authorization and in accordance with applicable law. Automatic anonymization can fail; review the final video before sharing it whenever identity protection is important.

## Planning

The detailed milestones, tests, and acceptance criteria are available in [TODO.md](TODO.md).
