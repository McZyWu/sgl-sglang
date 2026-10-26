"""Sample device-wide npu-smi HBM by benchmark case; no NPU tensors/hooks."""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path


def parse_hbm(text):
    cards, current = {}, None
    for line in text.splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|\s*Ascend", line)
        if match:
            current = match.group(1)
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if current is not None and len(pairs) >= 2 and int(pairs[-1][1]) > 0:
            used, total = map(int, pairs[-1])
            if not 0 <= used <= total:
                raise ValueError("Invalid HBM used/total")
            cards[current] = dict(used_mib=used, total_mib=total)
            current = None
    if not cards:
        raise ValueError("npu-smi output contains no recognized HBM rows")
    return cards


def read_hbm():
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=True
    )
    return parse_hbm(result.stdout)


class Sampler:
    def __init__(
        self, output, devices=None, interval=1, label="case", performance=None
    ):
        if interval <= 0:
            raise ValueError("HBM interval must be positive")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "hbm-samples.jsonl").exists():
            raise ValueError(
                "Use a fresh HBM output directory; prior samples must not be mixed"
            )
        self.devices = [str(device) for device in devices] if devices else None
        self.interval, self.label, self.performance = interval, label, performance
        self.groups, self.errors = {}, []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = None

    def sample(self):
        label = self.label
        if self.performance and self.performance.exists():
            try:
                item = json.loads(self.performance.read_text())
                label = item.get("active_case", {}).get("name", "between_cases")
            except (OSError, ValueError):
                label = "between_cases"
        now = time.time()
        try:
            cards = read_hbm()
            devices = self.devices or list(cards)
            if any(device not in cards for device in devices):
                raise ValueError(
                    f"Missing requested NPU IDs: {devices}; observed={list(cards)}"
                )
            cards = {device: cards[device] for device in devices}
            with self.lock:
                group = self.groups.setdefault(
                    label, dict(samples=0, first_time=now, last_time=now, by_npu={})
                )
                group["samples"] += 1
                group["last_time"] = now
                for device, value in cards.items():
                    peak = group["by_npu"].setdefault(
                        device, dict(peak_used_mib=0, total_mib=value["total_mib"])
                    )
                    peak["peak_used_mib"] = max(
                        peak["peak_used_mib"], value["used_mib"]
                    )
                    peak["min_free_mib"] = value["total_mib"] - peak["peak_used_mib"]
                event = dict(time=now, case=label, by_npu=cards)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            event = dict(time=now, case=label, error=str(error))
            with self.lock:
                self.errors.append(event)
        with (self.output / "hbm-samples.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
        self.save()

    def summary(self):
        with self.lock:
            result = json.loads(json.dumps(dict(cases=self.groups, errors=self.errors)))
        result.update(
            scope="device-wide sampled HBM; includes warmup/prefill/decode; not a continuous or per-process peak",
            interval_s=self.interval,
        )
        for group in result["cases"].values():
            group["max_card_peak_mib"] = max(
                card["peak_used_mib"] for card in group["by_npu"].values()
            )
        return result

    def save(self):
        target = self.output / "hbm-peaks.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary(), indent=2) + "\n")
        temporary.replace(target)

    def start(self):
        self.sample()

        def loop():
            while not self.done.wait(self.interval):
                try:
                    self.sample()
                except Exception as error:
                    with self.lock:
                        self.errors.append(
                            dict(time=time.time(), case=self.label, error=str(error))
                        )
                    self.done.set()
                    print(f"HBM monitor failed: {error}", flush=True)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=20)
            if self.thread.is_alive():
                raise RuntimeError("HBM sampler did not stop")
        self.sample()
        return self.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance",
        type=Path,
        help="Active performance.json, including a not-yet-created one",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    sampler = Sampler(
        args.output, args.devices, args.interval, performance=args.performance
    ).start()
    print(f"Sampling HBM: {args.output}/hbm-peaks.json (Ctrl-C to finish)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sampler.stop()
