//! `miniconf::Tree` control-plane integration tests.

use embedded_dsp::config::BiquadSettings;
use miniconf::json_core;

#[test]
fn biquad_settings_are_leaf_addressable() {
    let mut settings = BiquadSettings::default();

    json_core::set(&mut settings, "/b0", b"0.5").unwrap();
    json_core::set(&mut settings, "/b1", b"0.25").unwrap();
    json_core::set(&mut settings, "/a1", b"-0.25").unwrap();
    json_core::set(&mut settings, "/offset", b"1").unwrap();

    assert_eq!(settings.b0, 0.5);
    assert_eq!(settings.b1, 0.25);
    assert_eq!(settings.a1, -0.25);
    assert_eq!(settings.offset, 1.0);

    let filter = settings.build();
    assert_eq!(filter.coeff.ba, [0.5, 0.25, 0.0, -0.25, 0.0]);
    assert_eq!(filter.u, 1.0);
    assert_eq!(filter.min, f32::NEG_INFINITY);
    assert_eq!(filter.max, f32::INFINITY);
}

#[test]
fn biquad_settings_default_is_transparent() {
    // Default coefficients are zero with unbounded clamps, so building the
    // default yields a pass-through filter.
    let mut settings = BiquadSettings::default();
    json_core::set(&mut settings, "/b0", b"1.0").unwrap();
    let filter = settings.build();
    let mut state = embedded_dsp::filtering::DirectForm1::<f32>::new();
    assert_eq!(filter.process_df1(&mut state, 3.5), 3.5);
}
