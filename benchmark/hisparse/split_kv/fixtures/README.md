# Historical prefix100 fixture

`prefix100_64000_token_ids.json.gz` contains a JSON array of exactly 64,000 token IDs. These are the shared input IDs from the preserved GLM-5.2 W4A8 `prefix100` experiment whose selected three-round median was 203.723894 output tokens/s. The payload contains synthetic benchmark input, without model weights or generated answers.

The original one-request seed fixture and all 16 formal request prompts were checked for exact equality before this array was exported. Their original pickle SHA256 values and the compressed and decompressed JSON SHA256 values are recorded in `manifest.json`. The runner consumes JSON rather than pickle, and verifies both content hashes before sending requests.

The historical workload primes this prefix with one generated token, then runs 16 requests with the same 64,000-token input and 1,024 generated tokens each. Each formal response must report 63,872 cached input tokens. This prefix-cache count is separate from the C=12,288 resident attention KV capacity.

The JSON export preserves the input IDs. It does not make a new two-host PD measurement identical to the historical single-host experiment, or make a different client implementation the original benchmark client. The runner records its own measurement boundaries and all formal rounds.
