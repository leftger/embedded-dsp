#!/usr/bin/env bash
# Build the C ABI and compile/run a C smoke test against the shipped header.
#
# `embedded-dsp-ffi` is a standalone workspace (see its Cargo.toml), so it is
# addressed by manifest path rather than as a root-workspace member, and its
# artifacts are redirected under the main `target/` directory.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

manifest="crates/embedded-dsp-ffi/Cargo.toml"
export CARGO_TARGET_DIR="${EDS_FFI_TARGET_DIR:-$repo_root/target/ffi}"
out_dir="$CARGO_TARGET_DIR/debug"

cargo clippy --manifest-path "$manifest" --all-targets -- -D warnings
cargo build --manifest-path "$manifest"

# The C compiler is the check: the test uses every exported symbol through the
# public header, so a Rust/header signature drift fails to compile or link.
cc -std=c11 -Wall -Wextra -Werror -pedantic \
    crates/embedded-dsp-ffi/tests/c_smoke.c \
    -I crates/embedded-dsp-ffi/include \
    -L "$out_dir" -lembedded_dsp_ffi \
    -Wl,-rpath,"$out_dir" \
    -lm -o "$out_dir/eds_c_smoke"

"$out_dir/eds_c_smoke"
