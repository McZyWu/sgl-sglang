#!/usr/bin/env bash
set -eo pipefail

readonly EXPECTED_KERNEL_BASE="b47332ce9736a7920139ad8f29364b27ef416dea"
readonly EXPECTED_PTO="af66b6ada43c40ad9583731883fb9bef65ac840a"
readonly EXPECTED_CATLASS="70eda02b1271ff0e8c47db94469fae576feba97b"

usage() {
    cat <<'EOF'
Usage:
  setup_kernel.sh [--check-only] <kernel_checkout> [prefix]

Validates the frozen split-KV ABI and exact submodules. Without --check-only,
builds from an isolated tracked-source copy and installs one wheel into an
independent pip --target prefix.

Options:
  --check-only  Validate checkout, ancestry, submodules and ABI only.

Environment:
  CANN_ENV            CANN env script (default: /usr/local/Ascend/ascend-toolkit/set_env.sh)
  ASCEND_ENV_SCRIPT   Legacy alias for CANN_ENV
  BUILD_JOBS         Build parallelism (default: 2; positive integer)
  KERNEL_SOC         Kernel target (default: 910 for A3; use 950 for A5)
  SPLITKV_BUILD_ROOT Optional new directory for isolated build files and logs.
EOF
}

check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
    check_only=1
    shift
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

kernel_checkout="$(cd "$1" 2>/dev/null && pwd -P)" || {
    echo "kernel checkout does not exist: $1" >&2
    exit 2
}
prefix_arg="${2:-}"
if [[ ! -f "$kernel_checkout/build.sh" || ( ! -d "$kernel_checkout/.git" && ! -f "$kernel_checkout/.git" ) ]]; then
    echo "not a Git kernel checkout: $kernel_checkout" >&2
    exit 3
fi

original_pythonpath="${PYTHONPATH:-}"
cann_env_script="${CANN_ENV:-${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}}"
if [[ ! -r "$cann_env_script" ]]; then
    echo "CANN environment script is not readable: $cann_env_script" >&2
    exit 4
fi
# Vendor CANN scripts may read unset shell variables; keep nounset disabled.
source "$cann_env_script"
if [[ -n "$original_pythonpath" && -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$original_pythonpath:$PYTHONPATH"
elif [[ -n "$original_pythonpath" ]]; then
    export PYTHONPATH="$original_pythonpath"
fi

head="$(git -C "$kernel_checkout" rev-parse HEAD)"
if [[ -n "$(git -C "$kernel_checkout" diff --name-only HEAD --)" ]]; then
    echo 'kernel tracked files differ from HEAD; commit them or use a clean checkout first' >&2
    exit 5
fi
if ! git -C "$kernel_checkout" merge-base --is-ancestor "$EXPECTED_KERNEL_BASE" "$head"; then
    echo "kernel HEAD must be $EXPECTED_KERNEL_BASE or a descendant; got $head" >&2
    exit 5
fi

check_gitlink() {
    local path="$1" expected="$2" entry actual submodule_root dirty
    entry="$(git -C "$kernel_checkout" ls-tree "$head" -- "$path")"
    if [[ "$entry" != "160000 commit $expected"* ]]; then
        echo "unexpected gitlink for $path: $entry" >&2
        return 1
    fi
    if [[ ! -e "$kernel_checkout/$path/.git" ]]; then
        echo "submodule is not initialized (missing $path/.git)" >&2
        return 1
    fi
    submodule_root="$(git -C "$kernel_checkout/$path" rev-parse --show-toplevel)"
    if [[ "$(cd "$submodule_root" && pwd -P)" != "$(cd "$kernel_checkout/$path" && pwd -P)" ]]; then
        echo "submodule Git root mismatch for $path: $submodule_root" >&2
        return 1
    fi
    actual="$(git -C "$kernel_checkout/$path" rev-parse HEAD)"
    if [[ "$actual" != "$expected" ]]; then
        echo "unexpected submodule HEAD for $path: expected $expected, got $actual" >&2
        return 1
    fi
    dirty="$(git -C "$kernel_checkout/$path" status --porcelain)"
    if [[ -n "$dirty" ]]; then
        echo "submodule worktree is dirty: $path" >&2
        return 1
    fi
    printf 'submodule %s HEAD=%s verified\n' "$path" "$actual"
}

check_gitlink "third_party/pto-isa" "$EXPECTED_PTO"
check_gitlink "third_party/catlass" "$EXPECTED_CATLASS"

definition="$kernel_checkout/csrc/pytorch_extensions.cpp"
implementation="$kernel_checkout/csrc/fused_scatter_copy_sparse_flash_attention/op_host/fused_copy_sfa.cpp"
if ! grep -Fq 'Tensor(a!) hbm_k_cache, Tensor(b!) hbm_v_cache' "$definition" || \
   ! grep -Fq 'int dram_k_dev_ptr, int dram_v_dev_ptr' "$definition" || \
   ! grep -Fq 'dram_k_dev_ptr, int64_t dram_v_dev_ptr' "$implementation"; then
    echo 'split-KV ABI missing: expected independent HBM K/V tensors and DRAM K/V pointers' >&2
    exit 6
fi

echo "kernel checkout: $kernel_checkout"
echo "kernel HEAD: $head"
echo 'split-KV ABI: two HBM tensors (K,V), two DRAM pointers (K,V) verified'
echo "CANN environment: $cann_env_script"
echo "PYTHONPATH after CANN source: ${PYTHONPATH:-<empty>}"
if (( check_only )); then
    exit 0
fi

build_jobs="${BUILD_JOBS:-2}"
kernel_soc="${KERNEL_SOC:-910}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "BUILD_JOBS must be a positive integer; got: $build_jobs" >&2
    exit 7
fi
if [[ -n "${SPLITKV_BUILD_ROOT:-}" ]]; then
    task_root="$(mkdir -p "$SPLITKV_BUILD_ROOT" && cd "$SPLITKV_BUILD_ROOT" && pwd -P)"
    if [[ -n "$(find "$task_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "SPLITKV_BUILD_ROOT must be new or empty: $task_root" >&2
        exit 8
    fi
else
    task_root="$(mktemp -d "${TMPDIR:-/tmp}/hisparse-split-kv-kernel.XXXXXX")"
fi

if [[ -n "$prefix_arg" ]]; then
    prefix="$(mkdir -p "$prefix_arg" && cd "$prefix_arg" && pwd -P)"
    if [[ -n "$(find "$prefix" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "install prefix must be empty: $prefix" >&2
        exit 9
    fi
else
    prefix="$task_root/install"
    mkdir -p "$prefix"
fi

readonly logs="$task_root/logs"
readonly source_copy="$task_root/source"
mkdir -p "$logs" "$source_copy"
exec > >(tee -a "$logs/setup.stdout.log") 2> >(tee -a "$logs/setup.stderr.log" >&2)
finish() {
    local rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$logs/setup.exitcode.txt"
    exit "$rc"
}
trap finish EXIT

echo "task root: $task_root"
echo "isolated source: $source_copy"
echo "install prefix: $prefix"
echo "BUILD_JOBS=$build_jobs"
echo "KERNEL_SOC=$kernel_soc"
echo 'No NPU tests or serving processes are started by this setup helper.'
if ! pip3 show wheel >/dev/null 2>&1; then
    echo 'wheel must already be installed; refusing build.sh implicit shared pip install' >&2
    exit 10
fi

git -C "$kernel_checkout" archive --format=tar "$head" | tar -xf - -C "$source_copy"
copy_submodule() {
    local path="$1" commit="$2" destination
    destination="$source_copy/$path"
    mkdir -p "$destination"
    git -C "$kernel_checkout/$path" archive --format=tar "$commit" | tar -xf - -C "$destination"
}
copy_submodule 'third_party/pto-isa' "$EXPECTED_PTO"
copy_submodule 'third_party/catlass' "$EXPECTED_CATLASS"

python3 - "$source_copy" "$logs/shell-normalization.txt" <<'PY'
import hashlib
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
rows = []
for path in sorted(root.rglob("*.sh")):
    before = path.read_bytes()
    mode = path.stat().st_mode
    normalized = before.replace(b"\r\n", b"\n")
    if normalized != before:
        path.write_bytes(normalized)
        os.chmod(path, mode)
    rows.append(
        f"{path.relative_to(root)} crlf={before.count(bytes([13, 10]))} "
        f"before_sha256={hashlib.sha256(before).hexdigest()} "
        f"after_sha256={hashlib.sha256(normalized).hexdigest()}"
    )
manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"shell normalization records: {len(rows)} ({manifest})")
PY

python3 - "$source_copy/build.sh" "$task_root/build.sh.original" "$task_root/build.sh.parallel.patch" <<'PY'
import difflib
import pathlib
import sys

build, original_path, patch_path = map(pathlib.Path, sys.argv[1:])
text = build.read_text(encoding="utf-8")
needle = 'cmake --build "$BUILD_DIR" --target install -j 16'
replacement = 'cmake --build "$BUILD_DIR" --target install --parallel "${BUILD_JOBS:-2}"'
if text.count(needle) != 1:
    raise SystemExit(f"expected exactly one hard-coded CMake -j 16 line; found {text.count(needle)}")
original_path.write_text(text, encoding="utf-8")
updated = text.replace(needle, replacement)
build.write_text(updated, encoding="utf-8")
patch_path.write_text(
    "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="build.sh.original",
            tofile="build.sh (task-local parallelism)",
        )
    ),
    encoding="utf-8",
)
PY
chmod +x "$source_copy/build.sh"
echo "task-local parallelism diff: $task_root/build.sh.parallel.patch"
echo "shell normalization manifest: $logs/shell-normalization.txt"

(
    cd "$source_copy"
    BUILD_JOBS="$build_jobs" bash ./build.sh -a kernels "$kernel_soc"
)

shopt -s nullglob
wheels=("$source_copy"/output/sgl_kernel_npu*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "expected exactly one sgl_kernel_npu wheel in $source_copy/output; found ${#wheels[@]}" >&2
    exit 10
fi
wheel="${wheels[0]}"
echo "selected wheel: $wheel"
sha256sum "$wheel"
python3 -m pip install --no-deps --target "$prefix" "$wheel"
python3 - "$prefix" <<'PY'
import importlib.util
import pathlib
import sys

prefix = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(prefix))
spec = importlib.util.find_spec("sgl_kernel_npu")
if spec is None or spec.origin is None:
    raise SystemExit(f"sgl_kernel_npu not found under isolated prefix {prefix}")
print(f"installed package path: {spec.origin}")
PY
while IFS= read -r -d '' so; do
    sha256sum "$so"
done < <(find "$prefix" -type f \( -name '*.so' -o -name '*.so.*' \) -print0)
printf 'export KERNEL_PYTHON_PATH=%q\n' "$prefix"
echo "installed into isolated prefix: $prefix"
echo 'No shared site-packages were modified; no NPU test was run.'
