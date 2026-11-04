"""CPU checks for the historical client assets and benchmark acceptance gates."""

import gzip
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import frozen_bench as bench


class Row:
    pass


class FixtureReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("sglang.benchmark.datasets.common", "DatasetRow"):
            return Row
        raise pickle.UnpicklingError(f"Unexpected fixture class: {module}.{name}")


class TestFrozenBench(unittest.TestCase):
    def test_source_hashes_and_exact_or_expanded_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            runtime = work / "runtime"
            runtime.mkdir()
            archive, entry, wrapper = bench._verify_assets(runtime)
            self.assertEqual(
                hashlib.sha256(archive.read_bytes()).hexdigest(),
                "9c4b0054fd80137d5b9bed6ffa26dfb5912e495fb533409c866d6fb1c993d94a",
            )
            self.assertEqual(
                hashlib.sha256(entry.read_bytes()).hexdigest(),
                "685fe36ef7762d4e43cad945b40a73ea129980469858a39eb7a49b4f0bc74965",
            )
            self.assertTrue(wrapper.is_file())
            ids = json.loads(
                gzip.decompress(
                    (
                        bench._ROOT / "fixtures/prefix100_64000_token_ids.json.gz"
                    ).read_bytes()
                )
            )
            for bs in (16, 20):
                cache = work / f"cache-{bs}"
                metadata = bench._make_cache_fixture(ids, bs, cache, work)
                self.assertEqual(
                    metadata["formal_pickle_exact_historical_bytes"], bs == 16
                )
                data = (
                    cache
                    / f"gen_shared_prefix_42_1_{bs}_64000_0_1024_ids_TokenizersBackend.pkl"
                ).read_bytes()
                rows = FixtureReader(io.BytesIO(data)).load()
                self.assertEqual(len(rows), bs)
                self.assertTrue(
                    all(row.prompt == ids and row.output_len == 1024 for row in rows)
                )
                if bs == 16:
                    self.assertEqual(
                        hashlib.sha256(data).hexdigest(),
                        "55fec582626bfc99ea76b63d56a70db367e63a430798cfb6ba2143e809dcd223",
                    )

    def test_seed_and_measurement_have_distinct_output_contracts(self):
        row = {
            "completed": 1,
            "total_input_tokens": 64000,
            "total_output_tokens": 1,
            "duration": 1.0,
            "input_lens": [64000],
            "output_lens": [1],
            "errors": [""],
            "cached_tokens": [0],
            "output_throughput": 1.0,
        }
        bench._validate_row(row, 1, None, expected_output=1)
        with self.assertRaises(ValueError):
            bench._validate_row(row, 1, 63872)
        row.update(
            total_output_tokens=1024,
            output_lens=[1024],
            cached_tokens=[63872],
            output_throughput=1024.0,
        )
        bench._validate_row(row, 1, 63872)
        with self.assertRaises(ValueError):
            bench._validate_row(row, 1, None, expected_output=1)
        row["cached_tokens"] = [64000]
        with self.assertRaises(ValueError):
            bench._validate_row(row, 1, 63872)

    def test_cache_miss_or_regeneration_is_rejected(self):
        expected = Path("/private/cache/fixture.pkl")
        marker = f"Loading cached generated input data from {expected}"
        bench._assert_cache_hit(marker, expected)
        for text in ("", marker + "\nCaching generated input data to /other/cache"):
            with self.assertRaises(RuntimeError):
                bench._assert_cache_hit(text, expected)


if __name__ == "__main__":
    unittest.main()
