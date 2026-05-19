#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lib_path="$repo_root/py-lynxes/src/lynxes/.libs/liblynxes_mojo_kernels.so"

if ! command -v mojo >/dev/null 2>&1; then
  echo "error: mojo compiler is required to verify Lynxes Mojo kernels." >&2
  echo "Install Mojo in this Linux/WSL environment, then retry the commit." >&2
  exit 1
fi

bash "$repo_root/scripts/build_mojo_kernels.sh"

if [[ ! -f "$lib_path" ]]; then
  echo "error: Mojo build did not produce $lib_path" >&2
  exit 1
fi

export LYNXES_MOJO_LIB="$lib_path"
cargo test -p lynxes-core --test mojo_structural_features -- --nocapture
