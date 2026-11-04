#!/usr/bin/env bash
# Build the MF 1.3 A3 AccOffload extension in a private Python prefix.
set -eo pipefail
if [[ $# -ne 3 || $2 != --soc || $3 != A3 ]]; then
  echo "Usage: bash setup_memfabric.sh EMPTY_PREFIX --soc A3" >&2
  exit 2
fi
prefix=$1
if [[ -L "$prefix" || ( -e "$prefix" && ! -d "$prefix" ) ]]; then
  echo "Prefix must be a new or empty directory, not a symlink: $prefix" >&2
  exit 2
fi
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd -P)
if [[ -n $(find "$prefix" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "Prefix must be empty: $prefix" >&2
  exit 2
fi
exec > >(tee "$prefix/setup.stdout.log") 2> >(tee "$prefix/setup.stderr.log" >&2)
trap 'rc=$?; printf "%s\n" "$rc" > "$prefix/setup.exit"' EXIT
original_pythonpath=${PYTHONPATH:-}
cann_env=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
[[ -f "$cann_env" ]] || { echo "CANN environment not found: $cann_env" >&2; exit 2; }
set +u
source "$cann_env"
set -u
export PYTHONPATH="${PYTHONPATH:-}${original_pythonpath:+:$original_pythonpath}"
export MF_SETUP_PREFIX="$prefix" MF_SETUP_CANN_ENV="$cann_env"
python_bin=${PYTHON_BIN:-python3}
"$python_bin" - <<'PY'
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil

prefix = Path(os.environ["MF_SETUP_PREFIX"]).resolve()
dist = metadata.distribution("memfabric_hybrid")
if dist.version != "1.3.0":
    raise RuntimeError(f"Expected memfabric_hybrid 1.3.0, found {dist.version}")
spec = importlib.util.find_spec("memfabric_hybrid")
if spec is None or not spec.origin:
    raise RuntimeError("memfabric_hybrid package not found")
package = Path(spec.origin).resolve().parent
dist_dir = Path(dist._path).resolve()
if Path(dist.locate_file("memfabric_hybrid")).resolve() != package:
    raise RuntimeError("Imported package and distribution metadata do not match")
if prefix == package or prefix in package.parents or package in prefix.parents:
    raise RuntimeError("Private prefix must be independent of the installed package")
if not (package / ".acc_offload_provision_wheel_only").is_file():
    raise RuntimeError("Installed package lacks the wheel build marker")
original = package / "lib/libmf_hybm_accoffload.so"
original_bytes = original.read_bytes()
cann_home = os.environ.get("ASCEND_HOME_PATH")
version_path = Path(cann_home) / "compiler/version.info" if cann_home else None
manifest = {
    "source_package": str(package),
    "source_dist_info": str(dist_dir),
    "source_version": dist.version,
    "source_version_file": (package / "VERSION").read_text(),
    "source_library_bytes": len(original_bytes),
    "source_library_sha256": hashlib.sha256(original_bytes).hexdigest(),
    "cann_env": os.environ["MF_SETUP_CANN_ENV"],
    "ascend_home_path": cann_home,
    "cann_compiler_version": (
        version_path.read_text() if version_path and version_path.is_file() else None
    ),
    "soc": "A3",
}
shutil.copytree(package, prefix / "memfabric_hybrid", symlinks=True)
shutil.copytree(dist_dir, prefix / dist_dir.name, symlinks=True)
private = prefix / "memfabric_hybrid/lib/libmf_hybm_accoffload.so"
if private.is_symlink():
    raise RuntimeError("Private extension placeholder is a symlink")
# Always rebuild from the wheel sources for the selected CANN/torch environment.
private.unlink()
(prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY
export PYTHONPATH="$prefix${PYTHONPATH:+:$PYTHONPATH}"
torch_libs=$("$python_bin" - <<'PY'
import importlib.util
from pathlib import Path

paths = []
for name in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Required package is missing: {name}")
    paths.append(str(Path(spec.origin).parent / "lib"))
print(":".join(paths))
PY
)
export LD_LIBRARY_PATH="$torch_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir "$prefix/tmp"
export TMPDIR="$prefix/tmp"
"$python_bin" - <<'PY'
import ctypes
import hashlib
import json
import os
from pathlib import Path

import memfabric_hybrid
import torch
import torch_npu
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext

prefix = Path(os.environ["MF_SETUP_PREFIX"]).resolve()
package = Path(memfabric_hybrid.__file__).resolve().parent
if package != prefix / "memfabric_hybrid":
    raise RuntimeError(f"Unexpected imported package: {package}")
if not torch.npu.is_available():
    raise RuntimeError("An available A3 NPU is required for SoC verification")
device_name = torch.npu.get_device_name(0)
if not device_name.startswith("Ascend910_93"):
    raise RuntimeError(f"Expected an Ascend 910_93 A3 device, found {device_name}")
target = package / "lib"
if ensure_acc_offload_ext("A3", install_dir=str(target)) != str(target):
    raise RuntimeError("AccOffload build did not complete")
lib = target / "libmf_hybm_accoffload.so"
data = lib.read_bytes()
if not data.startswith(b"\x7fELF"):
    raise RuntimeError("Built AccOffload library is not ELF")
ctypes.CDLL(str(lib))
manifest_path = prefix / "manifest.json"
manifest = json.loads(manifest_path.read_text())
source = Path(manifest["source_package"]) / "lib/libmf_hybm_accoffload.so"
source_bytes = source.read_bytes()
if hashlib.sha256(source_bytes).hexdigest() != manifest["source_library_sha256"]:
    raise RuntimeError("Installed source library changed during the private build")
manifest.update({
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "device_name": device_name, "private_library": str(lib),
    "private_library_bytes": len(data),
    "private_library_sha256": hashlib.sha256(data).hexdigest(),
    "private_library_cdll_loaded": True,
})
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY
printf 'export MF_PYTHON_PATH=%q\n' "$prefix"
