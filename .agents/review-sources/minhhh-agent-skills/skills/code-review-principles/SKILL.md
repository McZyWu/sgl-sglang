---
name: code-review-principles
description: >
  Use when a language-specific code review skill (e.g. java-code-review) references this as a Prerequisites foundation. NOT invoked directly by users — only loaded via Prerequisites by language-specific skills.
---

# Code Review Principles

Universal principles for catching critical issues before they reach production, with focus on safety violations, concurrency bugs, and silent data corruption that compilers cannot detect.

## Workflow

### Step 1 — Collect files to be reviewed

* If the user does not mention a pull request diff OR specific files OR whole code base, then assume that they want to review staged changes only and follow the steps below

```bash
git diff --staged
git diff --staged --stat
```

If nothing is staged, stop and tell the user:
> "Nothing is staged. Run `git add <files>` first."

### Step 2 — Run the review checklist

Work through each category below. For every finding, assign a severity:

| Severity | Meaning |
| --- | --- |
| 🔴 CRITICAL | Must be fixed before deploying. Blocks merge/commit. |
| 🟡 WARNING | Should be addressed; advisory only, does not block. |
| 🔵 NOTE | Minor observation or improvement suggestion. |

### Step 3 — Present findings

Group findings by severity, then by location. Format:

```
🔴 CRITICAL — [file/location]
[Description of issue and impact]

Suggested fix:
[Concrete fix suggestion]
```

After all findings, show summary:

```
Review complete: X CRITICAL, Y WARNING, Z NOTES
```

### Step 4 — Conclude

**If CRITICAL findings exist:**
> "🔴 There are CRITICAL issues that must be resolved before deploying.
> Fix them and re-run the review, or let me help you address them."

**If no CRITICAL findings:**
> "✅ No critical issues found. [N warnings / notes listed above.]
> Ready to proceed."

If an undecided possibility surfaced during the review that's not ready to act on, mention `/idea-log` as a parking option — don't invoke it automatically.

## Review Checklist

Work through each category systematically to catch critical issues:

### 🔴 Safety (always check — any violation is CRITICAL)

**Resource leaks** - streams, connections, handles, executors must be closed:

* Files, sockets, database connections opened but not closed
* Missing cleanup in exception paths
* Resources allocated but cleanup deferred beyond safe scope

**Memory leaks** - references that prevent garbage collection:

* Thread-local storage not cleaned up
* Listeners/callbacks registered but never unregistered
* Caches without eviction policies

**Silent data corruption** - exceptions silently swallowed:

* Empty catch blocks discarding exceptions
* Catch-and-log without propagation where failure should bubble up
* State mutations succeeding despite operation failures

**Deadlock risk** - multiple locks acquired without documented ordering:

* Nested lock acquisition without clear ordering rules
* Locks held while calling external code
* Lock acquisition in inconsistent order across call sites

**Null safety** - unchecked nullable values:

* Return values from external calls used without null-check
* Optional values unwrapped without presence verification
* Dependency injection points that could yield null

### 🔴 Concurrency (CRITICAL if in shared/multi-threaded code)

**Blocking on async threads** - I/O or blocking operations on event-loop threads:

* Database queries on async handler threads
* File I/O on event-loop threads
* Thread.sleep() or blocking waits on async paths

**Non-thread-safe collections** - shared mutable state without synchronization:

* HashMap, ArrayList, HashSet shared across threads
* Mutable state in singleton/shared services without synchronization
* Race conditions on shared counters or flags

**Thread-local across async boundaries** - context loss in async code:

* Thread-local values expected to propagate across thread switches
* Request context assumed available after async boundary
* User/session data accessed after CompletableFuture/callback

**Missing thread safety documentation**:

* Shared mutable state without `// NOT thread-safe` comment
* Concurrency assumptions not documented

### 🟡 Reproducibility (WARNING)

**Non-deterministic ordering** - iteration order affects output:

* Hash-based collections used in build/bootstrap code where order matters
* Non-sorted output consumed by downstream systems
* Test assertions relying on unspecified iteration order

### 🟡 Performance (WARNING in hot paths, NOTE elsewhere)

**Hot path inefficiencies** - performance issues in frequently-called code:

* Functional/stream overhead in per-request paths
* Excessive allocations inside tight loops
* Unnecessary boxing/unboxing in hot code

**Reflection overuse** - reflection where direct calls would work:

* Reflection for simple method calls
* Dynamic lookup where static typing available

### 🟡 Testing (WARNING)

**🔴 Bug fix submitted without a failing test committed before the fix** — CRITICAL.
The fix may be correct but there is no evidence the test actually detected the
bug. A test written after the fix can pass for the wrong reasons and provides
no regression protection for this specific failure mode. Flag this regardless
of language or framework.

**Mocks hiding real issues** - mocked dependencies masking integration problems:

* Database operations mocked instead of using real/in-memory DB
* External APIs mocked without contract tests
* Tests passing with mocks but failing with real implementations

**Missing test coverage** - new logic without corresponding tests:

* Non-trivial business logic added without tests
* Error handling paths untested
* Integration points untested

**Wrong test scope** - heavyweight tests for lightweight needs:

* Full application startup for unit-level tests
* Integration tests where component tests would suffice

### 🔵 Code Clarity (NOTE)

**Immutability** - unnecessary mutability:

* Parameters and variables declared mutable when immutable would work
* Missing const/final/readonly markers on values that don't change

**Unnecessary qualification** - redundant prefixes:

* this./self. prefix where disambiguation unnecessary
* Fully qualified names where imports would be cleaner

**Documentation mismatch** - missing or excessive documentation:

* Trivial methods over-documented
* Non-obvious logic under-documented
* Comments explaining what instead of why

### 🔵 Minimize Changes (NOTE)

**Unnecessary changes** - modifications unrelated to stated purpose:

* Signature changes without semantic need
* Formatting/whitespace changed on untouched lines
* Import order changed unnecessarily

### 🔵 Anti-patterns (NOTE ceiling)

Shapes that hurt readability and let changes spread. None of these can be rated CRITICAL or WARNING; they do not cause production incidents by themselves. Report each as NOTE with a concrete rewrite, not a general complaint. Skip anything the project's linter already enforces.

**Coupled guard conditions.** Two unrelated cases share one condition, so the body re-derives which case it is in. Give the exceptional case its own early return.

```text
// BAD: undefined and "allowed once" share a branch
if (choice === undefined || choice === "allowed once") {
    return choice === "allowed once";
}

// GOOD: the exceptional case exits first, the real case stands alone
if (choice === undefined) {
    return false;
}
if (choice === "allowed once") {
    return true;
}
```

**Arrow code.** Nesting grows one level per condition, pushing the real work to the right and hiding the exit points. Return early on the failures and keep the happy path at one level.

```text
// BAD
if (user != null) {
    if (user.active) {
        if (user.role === "admin") {
            doWork(user);
        }
    }
}

// GOOD
if (user == null) {
    return;
}
if (!user.active) {
    return;
}
if (user.role !== "admin") {
    return;
}
doWork(user);
```

## Severity Decision Flow

```mermaid
flowchart TD
    Finding_detected((Finding detected))
    Safety_violation_{Safety violation?}
    Concurrency_issue_in_shared_code_{Concurrency issue in shared code?}
    CRITICAL[CRITICAL]
    In_hot_path_{In hot path?}
    WARNING[WARNING]
    NOTE[NOTE]
    Finding_detected --> Safety_violation_
    Safety_violation_ -->|yes| CRITICAL
    Safety_violation_ -->|no| Concurrency_issue_in_shared_code_
    Concurrency_issue_in_shared_code_ -->|yes| CRITICAL
    Concurrency_issue_in_shared_code_ -->|no| In_hot_path_
    In_hot_path_ -->|"yes (perf/repro)"| WARNING
    In_hot_path_ -->|no| NOTE
```

## Common Pitfalls

| Mistake | Why It's Wrong | Fix |
| :--- | :--- | :--- |
| Skipping safety checks for "simple" changes | Simple code has critical bugs too | Always check safety, even for one-liners |
| Approving without understanding the change | Can't catch issues in code you don't understand | Read and understand before approving |
| Focusing only on style/formatting | Misses critical safety and concurrency issues | Safety first, style last |
| Accepting "I tested it manually" | Manual tests don't catch edge cases or regressions | Require automated tests for non-trivial changes |
| Letting CRITICAL issues through | Critical issues cause production incidents | Never approve with unresolved CRITICAL findings |
| Over-focusing on performance in cold paths | Premature optimization adds complexity | Focus performance review on actual hot paths |
