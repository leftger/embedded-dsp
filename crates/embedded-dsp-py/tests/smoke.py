"""Smoke test for the embedded-dsp Python bindings (abi3).

Run against a built extension module, e.g.:

    CARGO_TARGET_DIR=target/py cargo build --manifest-path crates/embedded-dsp-py/Cargo.toml
    mkdir -p /tmp/eds_py && cp target/py/debug/libembedded_dsp.so /tmp/eds_py/embedded_dsp.so
    PYTHONPATH=/tmp/eds_py python3 crates/embedded-dsp-py/tests/smoke.py
"""

import math

import embedded_dsp

EXPECTED_TYPES = {
    "lowpass",
    "highpass",
    "bandpass",
    "allpass",
    "notch",
    "peaking",
    "lowshelf",
    "highshelf",
    "iho",
}


def expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    assert embedded_dsp.version()
    assert set(embedded_dsp.BIQUAD_TYPES) == EXPECTED_TYPES

    peaking = embedded_dsp.eq_coeffs("peaking", 1000.0, 48000.0, 0.707, gain_db=6.0)
    assert len(peaking) == 5 and all(math.isfinite(c) for c in peaking)

    iho = embedded_dsp.eq_coeffs("iho", 2000.0, 48000.0, 0.707, gain_db=-6.0)
    assert len(iho) == 5 and all(math.isfinite(c) for c in iho)

    expect_value_error(lambda: embedded_dsp.eq_coeffs("bogus", 1000.0, 48000.0, 0.707))
    expect_value_error(lambda: embedded_dsp.eq_coeffs("lowpass", 0.0, 48000.0, 0.707))

    out = embedded_dsp.fir_f32([0.25, 0.5, 0.25], [1.0, 0.0, 0.0, 0.0])
    assert abs(out[0] - 0.25) < 1e-6
    assert abs(out[1] - 0.5) < 1e-6
    assert abs(out[2] - 0.25) < 1e-6

    passthrough = embedded_dsp.biquad_cascade_f32(
        [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]
    )
    assert all(abs(a - b) < 1e-6 for a, b in zip(passthrough, [1.0, 2.0, 3.0]))

    print(f"python smoke: ok, version {embedded_dsp.version()}")


if __name__ == "__main__":
    main()
