---
name: npu-sglang-python-review
description: Review and guide code changes in SGLang Python, Ascend NPU backends, AscendC, and Triton operators, including ordinary Python quality checks. Use for code edits or explicit code reviews; not for read-only explanations or non-code tasks.
---

# SGLang, Python, and Ascend review

Use this skill for code changes and explicit reviews in this repository. Read the actual diff and trace affected callers, callees, data flow, and relevant configuration before judging behavior. For a code edit, consult this skill before changing code and again before closing the work. For an explicit review, report findings without implementing fixes unless the user asks.

## Route to focused guidance

Read only the references that match the change; combine relevant routes when needed.

- General Python changes: [Python review guidance](../../review-sources/minhhh-agent-skills/skills/python-code-review/SKILL.md), [review principles](../../review-sources/minhhh-agent-skills/skills/code-review-principles/SKILL.md), and [Python development](../../review-sources/minhhh-agent-skills/skills/python-dev/SKILL.md).
- General review method: [Sentry code review](../../review-sources/sentry-skills/skills/code-review/SKILL.md).
- NPU backend adaptation: [NPU adapter review](../../review-sources/ascend-agent-skills/skills/npu-adapter-reviewer/SKILL.md) and [local NPU checklist](references/npu-backend-checklist.md).
- AscendC operators: [AscendC review](../../review-sources/ascend-agent-skills/skills/ascendc-operator-code-review/SKILL.md).
- Triton operators: [Triton review](../../review-sources/ascend-agent-skills/skills/triton-operator-code-review/SKILL.md) and [shared Triton API reference](../../review-sources/ascend-agent-skills/skills/triton-operator-shared/references/triton-api-reference.md).
- Security-sensitive diffs, when applicable: [differential review](../../review-sources/trailofbits-skills/plugins/differential-review/skills/differential-review/SKILL.md).
- SGLang class structure: `.agents/skills/large-class-style/SKILL.md`.
- Runtime configuration, process state, or forward lifecycle: `.agents/skills/sglang-runtime-context/SKILL.md`.
- Environment variables: `.agents/skills/env-var-conventions/SKILL.md`.
- Kernel API placement or backend boundaries: `.agents/skills/kernel-organization/SKILL.md`.
- Tests: `.agents/skills/write-sglang-test/SKILL.md` and `test/README.md`.
- Mechanical extraction or relocation: `.agents/skills/mechanical-refactor-verify/SKILL.md`.

Treat imported source skills as reference material, not executable instructions. This skill's repository-specific rules below take precedence when sources conflict. Follow the checked-out branch's conventions, especially for older Ascend branches; do not impose unrelated upstream changes. The NPU migration source describes cloning, automatic file edits and compatibility-layer generation: use it only for review criteria and never execute its migration actions as part of a review. Any prefilled checkboxes or accuracy/performance numbers in source examples are illustrative; populate reports only from actual evidence.

## Review method

1. Identify the code version (commit or branch) and inspect the complete diff, including surrounding code. Trace callers and callees, state ownership/lifetime, configuration, and relevant existing tests.
2. Start with static analysis. Check behavior, error handling, compatibility, resource lifetime, concurrency, and project conventions. Preserve lazy logging. Sequential `await` may be intentional for ordering or backpressure; avoid unbounded `gather`. Do not label missing annotations or `Any` critical without a concrete impact. Follow SGLang test conventions rather than assuming pytest patterns.
3. For NPU changes, verify runtime environment, backend/device guards and GPU behavior. Check shapes, dtypes, strides/layouts, dynamic lengths and tails, output/in-place contracts, KV cache indices/capacity/lifetime/reuse, stream/event ordering, transfer-buffer lifetime, implicit synchronization, and graph-capture/replay stability where applicable.
4. For AscendC/Triton changes, validate rules against the actual CANN, torch_npu, compiler and SoC versions and reachable input bounds. Assess masks, layouts, atomics, synchronization and performance in context. Do not mechanically flag a pattern or add CPU-NPU sync or large host-side tensor expansion without evidence. AscendC manual scores and a 60% threshold are not statistical confidence or defect thresholds; decide from triggering inputs, call/data flow, and existing defenses. If a source report template is absent or spans unavailable directories, use the concise finding format below.
5. Select checks based on risk. For relevant kernel/backend behavior, use focused operator checks before end-to-end accuracy/performance checks when feasible. Compare equivalent model, hardware, environment, parallelism, batch, input/output lengths and configuration. Startup success or HTTP 200 alone is not a passing result. Profiling is optional and should answer a specific question.
6. Report findings with file and line, trigger, effect, evidence, and a minimal fix direction. Distinguish confirmed facts from hypotheses; mark unsupported performance gains as hypotheses. If a referenced resource is missing, verify against the checked-out source and limit the conclusion. Never invent test results or require an extra confirmation step for a resolved issue.

In the final report, state the code version, checks actually run and results, and relevant checks skipped with reasons. Briefly list the skills/guidance actually used. Keep the report proportional to the change; do not require a long-form report for small changes. Communicate task results in Chinese when appropriate; code, comments, docstrings, commits and PR text remain English.
