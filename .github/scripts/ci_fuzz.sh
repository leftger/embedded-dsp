#!/usr/bin/env bash
# Bounded differential fuzzing of the FIR and biquad kernels.
#
# The targets compare the `f32` kernels against in-process `f64` references, so a
# found input is a real numerical divergence, not just a crash. Bounded so the
# job stays well under a minute; run longer locally with FUZZ_RUNS.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root/fuzz"

cargo fuzz build

for target in fir_differential biquad_differential; do
    echo "fuzzing: $target"
    cargo fuzz run "$target" -- -runs="${FUZZ_RUNS:-20000}" -max_len=4096
done
