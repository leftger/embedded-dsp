#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml for the embedded-dsp repository.
#
# Usage:
#   bash .github/scripts/ci_local.sh
#
# Environment:
#   CI_LOCAL_SKIP       Comma/space separated job names to skip, e.g.
#                       CI_LOCAL_SKIP="coverage,bench,publish"
#   CI_LOCAL_TOOLCHAIN  Toolchain channel CI pins for the stable jobs
#                       (default: stable). MSRV always comes from ci.yml.
#
# This script deliberately mirrors the workflow's command lines rather than
# approximating them, and it reports every job as PASS/FAIL/SKIP in a final
# summary. A job that cannot run is an explicit SKIP, never a silent pass --
# an earlier version of this script exited 0 while skipping the no_std and
# coverage jobs entirely.
#
# It cannot reproduce artifact uploads (codecov, JUnit) or the GitHub runner
# image, so a green run here is strong evidence but not a guarantee.

set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

workflow=".github/workflows/ci.yml"
if [[ ! -f "$workflow" ]]; then
  echo "error: missing $workflow (run from the repository root)" >&2
  exit 1
fi

# ── Configuration mirrored from the workflow ────────────────────────────────
# MSRV and the coverage floor are read from the workflow's `env:` block so the
# two cannot drift apart.
MSRV="$(sed -n 's/^[[:space:]]*MSRV:[[:space:]]*"\?\([0-9.]*\)"\?[[:space:]]*$/\1/p' "$workflow" | head -1)"
if [[ -z "$MSRV" ]]; then
  echo "error: could not read MSRV from $workflow" >&2
  exit 1
fi

COVERAGE_MIN="$(sed -n 's/^[[:space:]]*COVERAGE_MIN:[[:space:]]*"\?\([0-9.]*\)"\?[[:space:]]*$/\1/p' "$workflow" | head -1)"
if [[ -z "$COVERAGE_MIN" ]]; then
  echo "error: could not read COVERAGE_MIN from $workflow" >&2
  exit 1
fi

TOOLCHAIN="${CI_LOCAL_TOOLCHAIN:-stable}"
CARGO=(cargo "+${TOOLCHAIN}")

NO_STD_TARGETS=(
  thumbv6m-none-eabi
  thumbv7em-none-eabi
  thumbv7em-none-eabihf
  thumbv8m.main-none-eabihf
  riscv32imc-unknown-none-elf
  wasm32-unknown-unknown
)

STANDALONE_FEATURES=(
  basic-math complex-math filtering controller cordic distance dither dsm
  fast-math fec filter-analysis filter-design fixed-point interpolation kalman lut
  matrix modem pipeline pll psd quaternion resampling sequence spatial statistics snapshot
  synthesis validation support transform window companding const-generics
  audio beamforming miniconf
)

FEATURE_SUBSETS=(
  "libm,basic-math"
  "libm,filtering"
  "libm,filtering,pipeline"
  "libm,complex-math,distance,interpolation,statistics"
  "libm,matrix,support,window,fast-math"
  "libm,controller,fast-math,filter-analysis,filtering,pipeline,pll,resampling,synthesis"
  "libm,fec"
  "libm,sequence"
  "libm,modem"
  "full,libm"
)

# ── Reporting ──────────────────────────────────────────────────────────────
pass=0
failed=0
skipped=0
SKIPPED_JOBS=()
OPTIONAL_FAILED=()

# Prints the reason to skip and returns 0 when the job should not run.
should_skip() {
  local job="$1"
  if [[ " ${CI_LOCAL_SKIP:-} " == *" $job "* ]]; then
    echo "CI_LOCAL_SKIP"
    return 0
  fi
  case "$job" in
    audit)
      command -v cargo-deny >/dev/null 2>&1 || { echo "cargo-deny not installed"; return 0; } ;;
    coverage)
      command -v cargo-llvm-cov >/dev/null 2>&1 || { echo "cargo-llvm-cov not installed"; return 0; } ;;
    license)
      command -v licensee >/dev/null 2>&1 || { echo "licensee (ruby gem) not installed"; return 0; } ;;
  esac
  return 1
}

run_job() {
  local name="$1"
  shift
  local reason
  if reason="$(should_skip "$name")"; then
    echo "SKIP: $name ($reason)"
    skipped=$((skipped + 1))
    SKIPPED_JOBS+=("$name [$reason]")
    return 0
  fi

  echo ""
  echo "========== $name =========="
  if "$@"; then
    echo "PASS: $name"
    pass=$((pass + 1))
  else
    echo "FAIL: $name" >&2
    failed=$((failed + 1))
  fi
}

# Mirrors a job with `continue-on-error: true` in the workflow: reported, but
# never counted as a blocking failure.
run_optional_job() {
  local name="$1"
  shift
  local reason
  if reason="$(should_skip "$name")"; then
    echo "SKIP: $name ($reason)"
    skipped=$((skipped + 1))
    SKIPPED_JOBS+=("$name [$reason]")
    return 0
  fi

  echo ""
  echo "========== $name (continue-on-error in CI) =========="
  if "$@"; then
    echo "PASS: $name"
    pass=$((pass + 1))
  else
    echo "NON-BLOCKING FAIL: $name (matches continue-on-error in CI)" >&2
    OPTIONAL_FAILED+=("$name")
  fi
}

# ── Jobs (same commands as the workflow) ───────────────────────────────────

job_fmt() {
  "${CARGO[@]}" fmt --all --check
}

job_check() {
  "${CARGO[@]}" check --all-targets --all-features
}

job_check_no_std() {
  local status=0 target
  for target in "${NO_STD_TARGETS[@]}"; do
    echo "-- $target (full,libm)"
    rustup target add "$target" --toolchain "$TOOLCHAIN" >/dev/null 2>&1 || true
    "${CARGO[@]}" check -p embedded-dsp --target "$target" \
      --no-default-features --features full,libm || status=1
    case "$target" in
      thumbv7em* | thumbv8m*)
        echo "-- $target (full,libm,cortex-m-dsp)"
        "${CARGO[@]}" check -p embedded-dsp --target "$target" \
          --no-default-features --features full,libm,cortex-m-dsp || status=1
        ;;
    esac
  done
  return "$status"
}

job_msrv() {
  if ! rustup toolchain list | grep -q "${MSRV}"; then
    echo "installing toolchain ${MSRV}"
    rustup toolchain install "$MSRV" --profile minimal || true
  fi
  cargo "+${MSRV}" check -p embedded-dsp --no-default-features --features full,libm
}

job_clippy() {
  # NOTE: `-D warnings`, matching CI. The active toolchain matters here: clippy
  # lints are added in minor releases, so this must run on the same channel the
  # workflow pins.
  local status=0
  "${CARGO[@]}" clippy --all-targets --all-features -- -D warnings || status=1
  rustup target add thumbv7em-none-eabihf --toolchain "$TOOLCHAIN" >/dev/null 2>&1 || true
  "${CARGO[@]}" clippy -p embedded-dsp --no-default-features --features full,libm \
    --target thumbv7em-none-eabihf -- -D warnings || status=1
  return "$status"
}

job_feature_matrix() {
  local status=0 feature features
  rustup target add thumbv7em-none-eabihf --toolchain "$TOOLCHAIN" >/dev/null 2>&1 || true

  echo "-- each module feature builds standalone"
  for feature in "${STANDALONE_FEATURES[@]}"; do
    "${CARGO[@]}" check -p embedded-dsp --no-default-features \
      --features "libm,$feature" --target thumbv7em-none-eabihf || status=1
  done

  echo "-- representative feature subsets (tests)"
  for features in "${FEATURE_SUBSETS[@]}"; do
    "${CARGO[@]}" test -p embedded-dsp --no-default-features --features "$features" || status=1
  done
  return "$status"
}

job_test() {
  local status=0
  "${CARGO[@]}" test --workspace --all-targets --all-features || status=1
  "${CARGO[@]}" test -p embedded-dsp --no-default-features --features full,libm || status=1
  return "$status"
}

job_docs() {
  bash .github/scripts/ci_docs.sh
}

job_publish() {
  "${CARGO[@]}" publish -p embedded-dsp --dry-run
}

job_audit() {
  # NOTE: `--all-features` is a *global* cargo-deny flag and must precede the
  # subcommand. cargo-deny-action passes it in this order.
  cargo deny --all-features check
}

job_coverage() {
  # Mirrors the workflow: the two feature configurations are merged so the
  # printed total matches the lcov that is uploaded to codecov.
  "${CARGO[@]}" llvm-cov clean --workspace || return 1
  "${CARGO[@]}" llvm-cov --no-report -p embedded-dsp --all-features || return 1
  "${CARGO[@]}" llvm-cov --no-report -p embedded-dsp \
    --no-default-features --features full,libm || return 1

  local summary pct
  summary="$("${CARGO[@]}" llvm-cov report --summary-only)" || return 1
  echo "$summary" | tail -n 12

  # Same floor as the workflow's "Enforce coverage floor" step.
  pct="$(echo "$summary" | awk '$1 == "TOTAL" { gsub(/%/, "", $10); print $10 }')"
  if [[ -z "$pct" ]]; then
    echo "could not read the coverage total" >&2
    return 1
  fi
  echo "library line coverage: ${pct}% (floor: ${COVERAGE_MIN}%)"
  awk -v pct="$pct" -v min="$COVERAGE_MIN" 'BEGIN {
    if (pct + 0 < min + 0) {
      printf "coverage %.2f%% is below the %.2f%% floor\n", pct, min
      exit 1
    }
  }'
}

job_bench() {
  "${CARGO[@]}" bench -p embedded-dsp --bench dsp_benchmarks -- \
    --check-baseline "$repo_root/crates/embedded-dsp/benches/dsp_baseline.txt" \
    --max-regression 3.0
}

job_embedded_bench() {
  rustup target add thumbv7em-none-eabihf --toolchain "$TOOLCHAIN" >/dev/null 2>&1 || true
  ( cd tests/embedded && "${CARGO[@]}" build --release --target thumbv7em-none-eabihf )
}

job_license() {
  bash .github/scripts/verify_license.sh
}

# ── Driver ─────────────────────────────────────────────────────────────────
echo "Local CI mirror for $(basename "$repo_root") (from $workflow)"
echo "toolchain=${TOOLCHAIN} (CI pins the stable channel)  MSRV=${MSRV}"

# The workflow uses floating `stable`. If the local default toolchain differs,
# a bare `cargo` would silently test something else -- this is how a new clippy
# lint can pass locally and fail in CI.
default_toolchain="$(rustup show active-toolchain 2>/dev/null | awk '{print $1}')"
echo "rustup default=${default_toolchain:-<none>}"
if [[ -n "${default_toolchain:-}" && "$default_toolchain" != "${TOOLCHAIN}"* ]]; then
  echo "WARNING: your default toolchain (${default_toolchain}) is not '${TOOLCHAIN}'."
  echo "         Use 'cargo +${TOOLCHAIN} ...' locally, or the results may not match CI."
fi
echo "${TOOLCHAIN} version: $("${CARGO[@]}" --version 2>/dev/null || echo '<unavailable>')"

run_job fmt job_fmt
run_job check job_check
run_job check-no-std job_check_no_std
run_job msrv job_msrv
run_job clippy job_clippy
run_job feature-matrix job_feature_matrix
run_job test job_test
run_job docs job_docs
run_optional_job publish job_publish
run_job audit job_audit
run_job coverage job_coverage
run_job bench job_bench
run_job embedded-bench job_embedded_bench
run_job license job_license

echo ""
echo "─────────── summary ───────────"
echo "passed:  $pass"
echo "failed:  $failed"
echo "skipped: $skipped"
if (( skipped > 0 )); then
  echo "skipped jobs: ${SKIPPED_JOBS[*]}"
fi
if (( ${#OPTIONAL_FAILED[@]} > 0 )); then
  echo "non-blocking failures: ${OPTIONAL_FAILED[*]}"
fi

if (( failed > 0 )); then
  echo "" >&2
  echo "local CI mirror FAILED ($failed job(s))" >&2
  exit 1
fi

if (( skipped > 0 )); then
  echo ""
  echo "local CI mirror passed, but $skipped job(s) were skipped (see above)."
  echo "Install the missing tooling to cover them; do not treat this as fully green."
  exit 0
fi

echo ""
echo "local CI mirror passed."
