"""Exercise real spawn startup with a fake server and no CANN or NPU imports."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FAKE_CTYPES = """
import json, os
from pathlib import Path
class Library:
    def prctl(self, *args):
        assert args == (41, 1, 0, 0, 0), args
        with (Path(os.environ["SPLITKV_SPAWN_PROBE"]) / "events.jsonl").open("a") as f:
            f.write(json.dumps({"kind": "prctl", "pid": os.getpid()}) + "\\n")
        return 0
def CDLL(*args, **kwargs):
    return Library()
def get_errno():
    return 0
"""

FAKE_SERVER = """
import json, multiprocessing as mp, os, sys
from pathlib import Path
from probe_worker import worker
if __name__ == "__main__":
    with (Path(os.environ["SPLITKV_SPAWN_PROBE"]) / "events.jsonl").open("a") as f:
        f.write(json.dumps({"kind":"launch", "pid":os.getpid(), "args":sys.argv[1:]}) + "\\n")
    proc = mp.get_context("spawn").Process(target=worker)
    proc.start()
    proc.join(10)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise RuntimeError("probe worker timed out")
    if proc.exitcode != 0:
        raise RuntimeError(f"probe worker exit {proc.exitcode}")
"""

FAKE_WORKER = """
import json, os
from pathlib import Path
def worker():
    with (Path(os.environ["SPLITKV_SPAWN_PROBE"]) / "events.jsonl").open("a") as f:
        f.write(json.dumps({"kind":"worker", "pid":os.getpid()}) + "\\n")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wrapper", type=Path, default=Path(__file__).with_name("no_thp_launch.py")
    )
    args = parser.parse_args()
    source = args.wrapper.read_bytes()
    with tempfile.TemporaryDirectory(prefix="splitkv-spawn-") as tmp:
        root = Path(tmp)
        (root / "sglang").mkdir()
        (root / "sglang/__init__.py").write_text("", encoding="utf-8")
        (root / "sglang/launch_server.py").write_text(FAKE_SERVER, encoding="utf-8")
        (root / "probe_worker.py").write_text(FAKE_WORKER, encoding="utf-8")
        (root / "ctypes.py").write_text(FAKE_CTYPES, encoding="utf-8")
        wrapper = root / "no_thp_launch.py"
        wrapper.write_bytes(source)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["SPLITKV_SPAWN_PROBE"] = str(root)
        result = subprocess.run(
            [sys.executable, str(wrapper), "--model-path", "probe-only"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        events_path = root / "events.jsonl"
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines()]
            if events_path.exists()
            else []
        )
    print(json.dumps({"exitcode": result.returncode, "events": events}, indent=2))
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            "Spawn probe failed; the wrapper may have relaunched the server"
        )
    for kind in ("prctl", "launch", "worker"):
        if sum(event["kind"] == kind for event in events) != 1:
            raise RuntimeError(f"Expected exactly one {kind} event: {events}")
    if next(event["args"] for event in events if event["kind"] == "launch") != [
        "--model-path",
        "probe-only",
    ]:
        raise RuntimeError("The wrapper changed server arguments")
    print("PASS: real spawn; one server entry, one child worker; CPU-only fake server")


if __name__ == "__main__":
    main()
