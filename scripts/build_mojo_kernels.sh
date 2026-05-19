#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$repo_root/py-lynxes/src/lynxes/.libs"

mkdir -p "$out_dir"
mojo build \
  --emit shared-lib \
  -o "$out_dir/liblynxes_mojo_kernels.so" \
  "$repo_root/kernels/mojo/lynxes_mojo_kernels.mojo"
