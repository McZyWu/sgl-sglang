# Frozen historical benchmark client

These assets preserve the client and prefix100 inputs used by the historical
GLM-5.2 W4A8, C12288, mode1, BS16 benchmark. The historical 203.723894 output
tokens/s value came from a selected three-round subset on one TP16 host. The
new two-host PD runs retain all five formal rounds and have a different topology.

`asset_manifest.json` records each stored asset, its source hash, and the hashes
of its parts. `source_manifest.json` records the original archive inventory.
The client archive and formal fixture are split into at most 1 MiB parts to
meet this repository's file-size check. `frozen_bench.py` reconstructs and checks
them before extraction. Python entry points are stored as deterministic gzip
files; their restored bytes are checked against their original hashes.

The unmodified `bench_client.tar.gz` provides the complete old Python package.
Only the benchmark subprocess prepends it to `PYTHONPATH`; serving continues
using the current framework. The original entry point is
`bench_serving_with_acceptance.py` (SHA256
`685fe36ef7762d4e43cad945b40a73ea129980469858a39eb7a49b4f0bc74965`).
The archive SHA256 is
`9c4b0054fd80137d5b9bed6ffa26dfb5912e495fb533409c866d6fb1c993d94a`.

At BS16 with the bundled inputs, both cache pickles retain their original
bytes. `fixture_helper.py` uses a DatasetRow-compatible stub in a separate
process to read the verified pickles. A different BS or input constructs new
rows and records that distinction in `fixture-provenance.json`. The runner
requires the old client's exact cache-hit message, so a tokenizer cache-key
mismatch or regenerated input fails validation. The expected historical
tokenizer class is `TokenizersBackend`.

Each round uses a private HOME and temporary directory, runs a 64000/1 seed,
then measures BS requests with 64000/1024 tokens. The client subprocess clears
`SGLANG_IS_IN_CI` and does not pass `--flush-cache`; the controller flushes P/D
before the seed. Full stdout, stderr, command, exit code and original JSONL are
kept under each round's `seed/` and `measure/` directories. The old client does
not preserve `finish_reason`; its gate checks successful request counts,
actual lengths, errors and exactly 63872 cached tokens per measured request.
