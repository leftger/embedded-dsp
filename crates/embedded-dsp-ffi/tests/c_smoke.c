/*
 * C smoke test for the embedded-dsp C ABI.
 *
 * Compiled against include/embedded_dsp.h and linked against the cdylib by
 * .github/scripts/ci_ffi.sh, so a signature drift between the Rust exports and
 * the header fails CI. Kept dependency-free beyond libc/libm.
 */

#include "embedded_dsp.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
    printf("embedded-dsp C ABI %s\n", eds_version());

    /* EQ designer: a +6 dB peaking filter must be finite. */
    float coeffs[5] = {0};
    assert(eds_eq_coeffs(EDS_BIQUAD_PEAKING, 1000.0f, 48000.0f, 0.707f, 6.0f,
                         coeffs) == EDS_OK);
    for (int i = 0; i < 5; ++i) {
        assert(isfinite(coeffs[i]));
    }

    /* IHo is a first-class type through the C API too. */
    assert(eds_eq_coeffs(EDS_BIQUAD_IHO, 2000.0f, 48000.0f, 0.707f, -6.0f,
                         coeffs) == EDS_OK);

    /* Out-of-range parameters are reported, not silently accepted. */
    assert(eds_eq_coeffs(EDS_BIQUAD_LOWPASS, 0.0f, 48000.0f, 0.707f, 0.0f,
                         coeffs) == EDS_ERROR_ARG);
    assert(eds_eq_coeffs(99, 1000.0f, 48000.0f, 0.707f, 0.0f, coeffs) ==
           EDS_ERROR_TYPE);

    /* FIR: a 3-tap moving average over an impulse reproduces the taps. */
    float fir_coeffs[3] = {0.25f, 0.5f, 0.25f};
    float state[3] = {0};
    float src[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float dst[4] = {0};
    assert(eds_fir_f32(3, fir_coeffs, state, src, dst, 4) == EDS_OK);
    assert(fabsf(dst[0] - 0.25f) < 1e-6f);
    assert(fabsf(dst[1] - 0.5f) < 1e-6f);
    assert(fabsf(dst[2] - 0.25f) < 1e-6f);
    assert(fabsf(dst[3]) < 1e-6f);

    /* Biquad cascade: a unity passthrough stage is transparent. */
    float bq_coeffs[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float bq_state[4] = {0};
    float bq_dst[4] = {0};
    assert(eds_biquad_cascade_df1_f32(1, 0, bq_coeffs, bq_state, src, bq_dst, 4) ==
           EDS_OK);
    for (int i = 0; i < 4; ++i) {
        assert(fabsf(bq_dst[i] - src[i]) < 1e-6f);
    }

    /* CFFT: four DC samples put all energy in bin 0. */
    float fft[8] = {1, 0, 1, 0, 1, 0, 1, 0};
    assert(eds_cfft_f32(fft, 4, 0) == EDS_OK);
    assert(fabsf(fft[0] - 4.0f) < 1e-4f);
    assert(fabsf(fft[1]) < 1e-4f);

    /* Null pointers are rejected without crashing. */
    assert(eds_fir_f32(3, NULL, state, src, dst, 4) == EDS_ERROR_NULL);
    assert(eds_cfft_f32(NULL, 4, 0) == EDS_ERROR_NULL);

    puts("c_smoke: ok");
    return 0;
}
