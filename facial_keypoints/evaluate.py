"""CLI: evaluate the trained model's pixel-space accuracy against a labeled CSV.

Requires the labeled dataset locally (not bundled in this repo -- see README
for how to obtain it). Not run in CI for that reason; the numbers in
docs/metrics.json were produced by this exact command against the held-out
test set and are documented in the README as a point-in-time result.
"""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CHECKPOINT, REPO_ROOT
from .inference import load_model
from .metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "test_frames_keypoints.csv")
    parser.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data" / "test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "metrics.json")
    args = parser.parse_args()

    model = load_model(str(args.checkpoint))
    metrics = evaluate(model, csv_file=str(args.csv), root_dir=str(args.images_dir))

    print(json.dumps(metrics.to_dict(), indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
