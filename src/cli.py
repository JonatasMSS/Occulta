import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.core import PipelineContext, PipelineError
from src.pipeline import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keep one reference face visible and blur all other detected faces.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--similarity-threshold", default=0.8, type=float)
    parser.add_argument("--debug-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = PipelineContext(
        reference_image=args.reference_image,
        input_video=args.input_video,
        output_video=args.output_video,
        similarity_threshold=args.similarity_threshold,
        debug_dir=args.debug_dir,
    )
    try:
        build_pipeline().handle(context)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        context.cleanup_temporary_files()

    print(f"Completed: {context.frames_processed} frames, {context.faces_kept} target faces kept, {context.faces_blurred} faces blurred.")
    print(f"Output: {context.output_video}")
    return 0
