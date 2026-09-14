#!/usr/bin/env bash
# Build the PyO3 extension and run the Python smoke test against it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

manifest="crates/embedded-dsp-py/Cargo.toml"
export CARGO_TARGET_DIR="${EDS_PY_TARGET_DIR:-$repo_root/target/py}"

cargo build --manifest-path "$manifest"

# Import the cdylib under the name the module's init symbol expects.
stage="$(mktemp -d)"
cp "$CARGO_TARGET_DIR/debug/libembedded_dsp.so" "$stage/embedded_dsp.so"

PYTHONPATH="$stage" python3 crates/embedded-dsp-py/tests/smoke.py

rm -rf "$stage"
