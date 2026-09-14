/*
 * embedded-dsp C ABI.
 *
 * Hand-maintained header for the `embedded-dsp-ffi` crate. The C smoke test in
 * `crates/embedded-dsp-ffi/tests/c_smoke.c` (run by `.github/scripts/ci_ffi.sh`)
 * compiles against this header and links the produced library, so the two cannot
 * drift without CI failing.
 *
 * Every function returns 0 (`EDS_OK`) on success and a negative `EDS_ERROR_*`
 * code on failure. The processing kernels are streaming: `state` is read and
 * written in place and must be zeroed by the caller once, before the first call.
 * They never reset the state themselves.
 */

#ifndef EMBEDDED_DSP_H
#define EMBEDDED_DSP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define EDS_OK 0
#define EDS_ERROR_NULL (-1)   /* a required pointer was null */
#define EDS_ERROR_LENGTH (-2) /* a length or index was out of range */
#define EDS_ERROR_TYPE (-3)   /* an enum discriminant was not recognised */
#define EDS_ERROR_PANIC (-4)  /* the call panicked; caught at the FFI boundary */
#define EDS_ERROR_ARG (-5)    /* a parameter failed validation */

/* Biquad response types accepted by `eds_eq_coeffs`. */
#define EDS_BIQUAD_LOWPASS 0
#define EDS_BIQUAD_HIGHPASS 1
#define EDS_BIQUAD_BANDPASS 2
#define EDS_BIQUAD_ALLPASS 3
#define EDS_BIQUAD_NOTCH 4
#define EDS_BIQUAD_PEAKING 5
#define EDS_BIQUAD_LOWSHELF 6
#define EDS_BIQUAD_HIGHSHELF 7
#define EDS_BIQUAD_IHO 8

/* Version string (e.g. "0.6.0"). Static storage; do not free. */
const char *eds_version(void);

/*
 * FIR filter, `f32`, over a block.
 *
 *   coeffs : num_taps coefficients
 *   state  : num_taps values, zeroed by the caller before the first call
 *   src    : len input samples
 *   dst    : len output samples
 */
int32_t eds_fir_f32(uint32_t num_taps, const float *coeffs, float *state,
                    const float *src, float *dst, uint32_t len);

/*
 * Direct Form I biquad cascade, `f32`, over a block.
 *
 *   coeffs     : 5 * num_stages values, [b0, b1, b2, a1, a2] per stage
 *   state      : 4 * num_stages values, zeroed before the first call
 *   post_shift : extra fixed-point headroom; pass 0 for `f32`
 *   src, dst   : len samples each
 */
int32_t eds_biquad_cascade_df1_f32(uint32_t num_stages, uint32_t post_shift,
                                   const float *coeffs, float *state,
                                   const float *src, float *dst, uint32_t len);

/*
 * In-place complex FFT of `n_complex` interleaved (re, im) pairs.
 *
 *   data      : 2 * n_complex floats
 *   ifft_flag : non-zero selects the inverse transform (unnormalised)
 */
int32_t eds_cfft_f32(float *data, uint32_t n_complex, uint32_t ifft_flag);

/*
 * Designs audio-EQ biquad coefficients [b0, b1, b2, a1, a2] (Direct Form I).
 *
 *   typ    : an EDS_BIQUAD_* constant
 *   q      : quality factor
 *   gain_db: used by peaking, the shelves, and IHo
 *   out_coeffs: 5 floats; written as the passthrough biquad on EDS_ERROR_ARG
 */
int32_t eds_eq_coeffs(uint32_t typ, float frequency_hz, float sample_rate_hz,
                      float q, float gain_db, float *out_coeffs);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDED_DSP_H */
