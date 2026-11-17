# Review source materials

The pinned upstream materials below are fixed-version copies for the local review workflow, with whitespace normalized to repository conventions only. Use them as needed through the [active review skill](../skills/npu-sglang-python-review/SKILL.md); they are not automatic execution instructions. Adapt repository-local conventions where they differ. Sample checkboxes, scores, or severity percentages in upstream material are not evidence that a check was performed or a finding is valid.

| Repository | Pinned commit | License | Source |
| --- | --- | --- | --- |
| Ascend/agent-skills | `155ac37bd169ddb89479af528297cfb2237400aa` | MulanPSL-2.0 | [Repository](https://github.com/Ascend/agent-skills/tree/155ac37bd169ddb89479af528297cfb2237400aa) |
| minhhh/agent-skills | `9f4c8c9f240d3d56463f9b826668b09e57f115f5` | MIT | [Repository](https://github.com/minhhh/agent-skills/tree/9f4c8c9f240d3d56463f9b826668b09e57f115f5) |
| getsentry/skills | `a582c51ee29d0f35e6ac321340796f77d43da733` | Apache-2.0 | [Repository](https://github.com/getsentry/skills/tree/a582c51ee29d0f35e6ac321340796f77d43da733) |
| trailofbits/skills | `149d7b21f4067765b82e67029730b42d228ef4b4` | CC-BY-SA-4.0 | [Repository](https://github.com/trailofbits/skills/tree/149d7b21f4067765b82e67029730b42d228ef4b4) |

`sources.lock.json` records repository identities, pinned commits, entrypoints, supporting paths, imported source and local paths, local SHA256 hashes, upstream SHA256 hashes, and Git blob IDs. The upstream fingerprints identify the raw pinned Git objects; local hashes identify normalized copies. The lock documents the reproducible whitespace normalization, and both sides can be verified independently. Refresh imported files directly from the pinned Git objects to avoid Windows line-ending conversion. It also records the AscendC summary-style reference that was absent at the pinned source commit. The local review skill can use a short report format when that upstream template is unavailable.
