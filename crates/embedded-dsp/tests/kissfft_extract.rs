//! Packed RFFT, mixed-radix CFFT, and streaming overlap-scrap FIR.

use embedded_dsp::filtering::{FastFirF32, conv_f32};
use embedded_dsp::transform::{
    cfft_f32, cfft_f32_len_ok, fft_is_235_smooth, irfft_f32, next_fast_fft_size, rfft_f32,
};

fn naive_dft(interleaved: &[f32], n: usize, inverse: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; 2 * n];
    let sign = if inverse { 1.0 } else { -1.0 };
    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let a = sign * 2.0 * core::f32::consts::PI * (k as f32) * (t as f32) / (n as f32);
            let (wr, wi) = (a.cos(), a.sin());
            let xr = interleaved[2 * t];
            let xi = interleaved[2 * t + 1];
            re += xr * wr - xi * wi;
            im += xr * wi + xi * wr;
        }
        if inverse {
            re /= n as f32;
            im /= n as f32;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
}

fn naive_rfft(src: &[f32]) -> Vec<f32> {
    let n = src.len();
    let mut c = vec![0.0f32; 2 * n];
    for i in 0..n {
        c[2 * i] = src[i];
    }
    naive_dft(&c, n, false)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn next_fast_fft_size_is_235_smooth() {
    assert!(fft_is_235_smooth(12));
    assert!(fft_is_235_smooth(240));
    assert!(!fft_is_235_smooth(7));
    assert!(!fft_is_235_smooth(14));
    assert_eq!(next_fast_fft_size(7), 8);
    assert_eq!(next_fast_fft_size(11), 12);
    assert_eq!(next_fast_fft_size(13), 15);
    assert!(cfft_f32_len_ok(12));
    assert!(cfft_f32_len_ok(64));
    assert!(!cfft_f32_len_ok(7));
}

#[test]
fn packed_rfft_f32_matches_naive_and_roundtrips() {
    let src: Vec<f32> = (0..32)
        .map(|i| (2.0 * core::f32::consts::PI * 3.0 * i as f32 / 32.0).sin())
        .collect();
    let naive = naive_rfft(&src);
    let mut packed = [0.0f32; 64];
    rfft_f32(&src, &mut packed, 32, 0);
    assert!(
        max_abs_diff(&packed, &naive) < 2e-4,
        "packed vs naive max {}",
        max_abs_diff(&packed, &naive)
    );

    let mut time = [0.0f32; 32];
    irfft_f32(&packed, &mut time, 32);
    for i in 0..32 {
        assert!(
            (time[i] - src[i]).abs() < 2e-4,
            "irfft[{i}] {} vs {}",
            time[i],
            src[i]
        );
    }
}

#[test]
fn packed_rfft_f32_even_mixed_radix_length() {
    let n = 12;
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let naive = naive_rfft(&src);
    let mut packed = [0.0f32; 24];
    rfft_f32(&src, &mut packed, n, 0);
    assert!(
        max_abs_diff(&packed, &naive) < 2e-4,
        "n=12 packed vs naive {}",
        max_abs_diff(&packed, &naive)
    );
    let mut time = [0.0f32; 12];
    irfft_f32(&packed, &mut time, n);
    for i in 0..n {
        assert!((time[i] - src[i]).abs() < 2e-4);
    }
}

#[test]
fn mixed_radix_cfft_matches_naive_dft() {
    for n in [6, 12, 15, 20, 24, 48, 60] {
        let mut data = vec![0.0f32; 2 * n];
        for i in 0..n {
            data[2 * i] = (i as f32).sin();
            data[2 * i + 1] = (i as f32 * 0.3).cos() * 0.25;
        }
        let expect = naive_dft(&data, n, false);
        cfft_f32(&mut data, n, 0, 1);
        assert!(
            max_abs_diff(&data, &expect) < 5e-4,
            "n={n} forward max {}",
            max_abs_diff(&data, &expect)
        );
        cfft_f32(&mut data, n, 1, 1);
        let mut orig = vec![0.0f32; 2 * n];
        for i in 0..n {
            orig[2 * i] = (i as f32).sin();
            orig[2 * i + 1] = (i as f32 * 0.3).cos() * 0.25;
        }
        assert!(
            max_abs_diff(&data, &orig) < 5e-4,
            "n={n} roundtrip max {}",
            max_abs_diff(&data, &orig)
        );
    }
}

#[test]
fn mixed_radix_impulse_dc() {
    let n = 12;
    let mut data = [0.0f32; 24];
    data[0] = 1.0;
    cfft_f32(&mut data, n, 0, 1);
    for i in 0..n {
        assert!((data[2 * i] - 1.0).abs() < 1e-5);
        assert!(data[2 * i + 1].abs() < 1e-5);
    }
}

#[test]
fn fast_fir_matches_linear_convolution() {
    let signal = [0.5f32, 1.0, -0.25, 0.75, 0.0, 2.0, -1.0, 0.5, 0.25, 1.5];
    let kernel = [1.0f32, 0.5, -0.25, 0.125];
    let mut direct = [0.0f32; 13];
    conv_f32(&signal, &kernel, &mut direct);

    let mut fir = FastFirF32::<16>::new(&kernel).expect("fft size 16");
    let mut streamed = [0.0f32; 16];
    let mut n = fir.process(&signal, &mut streamed);
    n += fir.flush(&mut streamed[n..]);
    assert_eq!(n, 13);
    for i in 0..13 {
        assert!(
            (streamed[i] - direct[i]).abs() < 1e-4,
            "fastfir[{i}] {} vs conv {}",
            streamed[i],
            direct[i]
        );
    }
}

#[test]
fn fast_fir_chunked_input_matches_one_shot() {
    let signal: [f32; 20] = core::array::from_fn(|i| (i as f32) * 0.07 - 0.5);
    let kernel = [0.25f32, 0.5, 0.25];
    let mut a = FastFirF32::<32>::new(&kernel).unwrap();
    let mut b = FastFirF32::<32>::new(&kernel).unwrap();
    let mut one = [0.0f32; 32];
    let mut n1 = a.process(&signal, &mut one);
    n1 += a.flush(&mut one[n1..]);

    let mut chunked = [0.0f32; 32];
    let mut n2 = 0;
    for chunk in signal.chunks(3) {
        n2 += b.process(chunk, &mut chunked[n2..]);
    }
    n2 += b.flush(&mut chunked[n2..]);
    assert_eq!(n1, n2);
    assert!(max_abs_diff(&one[..n1], &chunked[..n2]) < 1e-4);
}

#[test]
fn helpers_and_cfft_guard_clauses() {
    assert!(!fft_is_235_smooth(0));
    assert!(!fft_is_235_smooth(1));
    assert_eq!(next_fast_fft_size(0), 2);
    assert_eq!(next_fast_fft_size(1), 2);
    assert!(!cfft_f32_len_ok(0));
    assert!(!cfft_f32_len_ok(1));
    assert!(!cfft_f32_len_ok(540));
    assert!(cfft_f32_len_ok(2));

    let mut tiny = [0.0f32; 2];
    cfft_f32(&mut tiny, 1, 0, 1);
    cfft_f32(&mut tiny, 4, 0, 1);
    assert_eq!(tiny, [0.0, 0.0]);

    let mut seven = [1.0f32; 14];
    cfft_f32(&mut seven, 7, 0, 1);
    assert_eq!(seven[0], 1.0);

    let mut big = [0.0f32; 1080];
    big[0] = 1.0;
    cfft_f32(&mut big, 540, 1, 1);
    assert_eq!(big[0], 1.0);
}

#[test]
fn rfft_irfft_fallbacks_and_error_paths() {
    let src15: Vec<f32> = (0..15).map(|i| (i as f32) * 0.05).collect();
    let naive = naive_rfft(&src15);
    let mut spec = [0.0f32; 30];
    rfft_f32(&src15, &mut spec, 15, 0);
    assert!(max_abs_diff(&spec, &naive) < 5e-4);
    let mut time = [0.0f32; 15];
    irfft_f32(&spec, &mut time, 15);
    for i in 0..15 {
        assert!((time[i] - src15[i]).abs() < 5e-4);
    }

    let src8 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut ifft_out = [0.0f32; 16];
    rfft_f32(&src8, &mut ifft_out, 8, 1);

    let mut too_small = [0.0f32; 3];
    rfft_f32(&src8, &mut too_small, 8, 0);
    assert_eq!(too_small, [0.0, 0.0, 0.0]);

    let src7 = [1.0f32; 7];
    let mut spec7 = [0.0f32; 14];
    rfft_f32(&src7, &mut spec7, 7, 0);
    assert_eq!(spec7[0], 0.0);
    irfft_f32(&spec7, &mut [0.0f32; 7], 7);
    irfft_f32(&[0.0f32; 2], &mut [0.0f32; 8], 8);
}

#[test]
fn radix5_mixed_sizes_roundtrip() {
    for n in [5, 10, 25, 45] {
        let mut data = vec![0.0f32; 2 * n];
        for i in 0..n {
            data[2 * i] = (i as f32 * 0.2).sin();
        }
        let orig = data.clone();
        cfft_f32(&mut data, n, 0, 1);
        cfft_f32(&mut data, n, 1, 1);
        assert!(
            max_abs_diff(&data, &orig) < 1e-3,
            "n={n} roundtrip {}",
            max_abs_diff(&data, &orig)
        );
    }
}

#[test]
fn fast_fir_constructors_flush_and_reset() {
    assert!(FastFirF32::<16>::new(&[]).is_none());
    assert!(FastFirF32::<16>::new(&[0.0; 17]).is_none());
    assert!(FastFirF32::<7>::new(&[1.0]).is_none());
    assert!(FastFirF32::<1024>::new(&[1.0]).is_none());

    let mut fir = FastFirF32::<16>::new(&[1.0f32, 0.5]).unwrap();
    assert_eq!(fir.n_taps(), 2);
    assert_eq!(fir.ngood(), 15);
    assert_eq!(fir.process(&[1.0; 4], &mut []), 0);

    let mut out = [0.0f32; 32];
    let n = fir.process(&[1.0; 20], &mut out);
    assert!(n > 0);
    fir.reset();
    let mut flushed = [0.0f32; 8];
    let _ = fir.flush(&mut flushed);

    let mut identity = FastFirF32::<8>::new(&[1.0]).unwrap();
    assert_eq!(identity.flush(&mut [0.0f32; 4]), 0);

    let mut tight = FastFirF32::<16>::new(&[1.0, 0.0, 0.0]).unwrap();
    let mut tiny_out = [0.0f32; 1];
    let _ = tight.process(&[0.5; 20], &mut tiny_out);
    let _ = tight.flush(&mut tiny_out);
}
