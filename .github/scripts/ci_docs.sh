#!/usr/bin/env bash
# Build docs with warnings denied (broken links, missing docs, etc.).
set -euo pipefail

export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

cargo doc --no-deps --all-features

echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=embedded_dsp/index.html"></head></html>' > target/doc/index.html
