from pathlib import Path
import sys


def run() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from face_blur.app import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
