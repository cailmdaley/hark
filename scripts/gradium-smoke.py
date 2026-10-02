"""Exercise the real CLI against a local Gradium socket; never call the hosted API."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import numpy as np
from gradium_mock import MockGradium
from hark import cli
from hark.gradium import GradiumTrack
from hark.voice import OnlineCluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--synthetic-cluster", action="store_true")
    parser.add_argument("--drop-at", type=float)
    args = parser.parse_args()
    home = args.home or Path(tempfile.mkdtemp(prefix="hark-mock-"))
    home.mkdir(parents=True, exist_ok=True)
    cli.HOME = home.resolve()
    os.environ["GRADIUM_API_KEY"] = "local-mock-only"
    cli.credits_left = lambda key: 45000
    # Initialize numerical-library threads before installing the CLI signal watcher.
    np.dot(np.ones((256, 256)), np.ones((256, 256)))
    task_dir = Path("/proc/self/task")
    print(json.dumps({"home": str(cli.HOME),
                      "threads": len(list(task_dir.iterdir())) if task_dir.exists() else None}), flush=True)
    with MockGradium(drop_first_at=args.drop_at) as mock:
        def make_track(name, **kwargs):
            cluster = OnlineCluster(lambda _: np.array([1., 0.])) if args.synthetic_cluster else None
            options = {"phrase_seconds": 2} if args.synthetic_cluster else {}
            return GradiumTrack(name, **kwargs, url=mock.url, cluster=cluster, **options)
        cli.GradiumTrack = make_track
        options = ["--file", str(args.file)] if args.file else ["--phone"]
        result = cli.main(["--ear", "gradium", "-o", str(home / "meeting.txt")] + options)
        print(json.dumps({"connections": len(mock.connections),
                          "received_seconds": sum(c["samples"] for c in mock.connections) / 16000}), flush=True)
        return result


if __name__ == "__main__":
    sys.exit(main())
