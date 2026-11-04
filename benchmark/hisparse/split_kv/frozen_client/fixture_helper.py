"""Read trusted historical pickles through a DatasetRow-compatible stub."""

import gzip
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path


def install_dataset_row_stub():
    names = (
        "sglang",
        "sglang.benchmark",
        "sglang.benchmark.datasets",
        "sglang.benchmark.datasets.common",
    )
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)

    class DatasetRow:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    DatasetRow.__module__ = "sglang.benchmark.datasets.common"
    DatasetRow.__qualname__ = "DatasetRow"
    sys.modules["sglang.benchmark.datasets.common"].DatasetRow = DatasetRow


def load_fixture(name):
    base = Path(__file__).parent
    path = base / name
    if name.endswith(".parts"):
        chunks = [p.read_bytes() for p in sorted(path.iterdir())]
        compressed = b"".join(chunks)
    else:
        compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    expected = {
        "fixtures/prefix100_warm.pkl.gz": "59706e2cf1d0ca6e3326c88c94f0d8331f975fae16d06f1e4790a951ce837432",
        "fixtures/prefix100_formal16.pkl.gz.parts": "55fec582626bfc99ea76b63d56a70db367e63a430798cfb6ba2143e809dcd223",
    }
    if hashlib.sha256(raw).hexdigest() != expected[name]:
        raise ValueError(f"historical pickle SHA256 mismatch: {name}")
    return raw


def main():
    ids_path, warm_path, formal_path, bs, work = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
        Path(sys.argv[5]),
    )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    install_dataset_row_stub()
    warm_raw = load_fixture("fixtures/prefix100_warm.pkl.gz")
    formal_raw = load_fixture("fixtures/prefix100_formal16.pkl.gz.parts")
    warm_rows = pickle.loads(warm_raw)
    formal_rows = pickle.loads(formal_raw)
    if len(warm_rows) != 1 or len(formal_rows) != 16:
        raise ValueError("frozen historical fixture row count mismatch")
    warm_exact = list(warm_rows[0].prompt) == ids
    formal_exact = all(list(row.prompt) == ids for row in formal_rows)
    warm_path.parent.mkdir(parents=True, exist_ok=True)
    if warm_exact:
        warm_path.write_bytes(warm_raw)
    else:
        row = type(warm_rows[0])(**vars(warm_rows[0]))
        row.prompt = list(ids)
        row.prompt_len = len(ids)
        row.text_prompt_len = len(ids)
        warm_path.write_bytes(pickle.dumps([row], protocol=pickle.HIGHEST_PROTOCOL))
    exact = bs == 16 and formal_exact
    if exact:
        formal_path.write_bytes(formal_raw)
    else:
        rows = []
        for _ in range(bs):
            row = type(formal_rows[0])(**vars(formal_rows[0]))
            row.prompt = list(ids)
            row.prompt_len = len(ids)
            row.text_prompt_len = len(ids)
            row.output_len = 1024
            rows.append(row)
        formal_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    (work / "fixture-helper.json").write_text(
        json.dumps(
            {
                "formal_rows": bs,
                "formal_pickle_exact_historical_bytes": exact,
                "expansion": None
                if exact
                else f"built {bs} rows from supplied token IDs using historical DatasetRow fields",
                "ids_sha256": digest,
                "warm_pickle_exact_historical_bytes": warm_exact,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
