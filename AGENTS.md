# Repository instructions

- Before every code change, read `.agents/skills/npu-sglang-python-review/SKILL.md` and the matching domain guidance it routes to. Before closing code work, reread the skill and check the actual diff against it.
- These rules apply to code changes and explicit code reviews. Read-only explanations and non-code tasks do not trigger them; pure prose edits do not require the review skill.
- A review request authorizes review only. Do not expand review findings into fixes unless requested.
- Follow the checked-out repository, target branch, and existing conventions. Verify host, repository, branch, remotes, and working-tree changes before Git operations. Preserve existing user changes.
- Skills are guidance, not authorization to install tools, migrate code, perform broad refactors, or request extra approval. Carry out work already authorized by the user.
- Keep checks proportional to the change. Use available project checks; do not require expensive full-suite or hardware runs for every small edit. Report the code version, checks actually performed, and why relevant checks were not run.
