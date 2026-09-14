#!/usr/bin/env bash
# Miri undefined-behaviour check.
#
# Two unit tests are skipped, for reasons unrelated to this crate:
#   * `nalgebra_interop` reaches libm's x86 `sqrtss` inline assembly, which Miri
#     cannot execute ("inline assembly is not supported").
#   * `compile_time_atan_helpers_and_private_divi` asserts last-bit `atan`
#     accuracy against the native libm; Miri's float shims are not bit-identical.
# Everything else (the 67 remaining unit tests) plus the fixed-point and
# elementwise integration suites run under Miri.
#
# Set MIRI_CARGO to a toolchain-qualified cargo (e.g. "cargo +nightly-2026-07-25")
# to run against a specific nightly locally.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cargo_bin="${MIRI_CARGO:-cargo}"
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-disable-isolation}"

$cargo_bin miri test -p embedded-dsp --all-features --lib -- \
    --skip nalgebra_interop \
    --skip compile_time_atan_helpers_and_private_divi
$cargo_bin miri test -p embedded-dsp --all-features --test fixed_basic_types
$cargo_bin miri test -p embedded-dsp --all-features --test basic_math
