//! Guards the hand-maintained `full` feature against the module list in
//! `src/lib.rs`.
//!
//! `full` is the "every algorithm module" aggregator, and the per-module
//! features are listed by hand in `Cargo.toml`. Nothing at compile time notices
//! when a new `pub mod` lands without a matching `full` entry, so this test
//! reads both files and fails with the offenders. Modules wired through
//! `gated_mod!` are the only ones that can be gated: `math`, `types`, and
//! `intrinsics` are always available and have no feature to check.

use std::collections::BTreeSet;

/// Features that gate a module but are deliberately *not* part of `full`,
/// because they pull in an optional third-party dependency instead of an
/// in-crate algorithm module. Keep this list tiny and justified.
const OPT_IN: &[&str] = &[
    // `nalgebra_interop`: bridges `quaternion` to the `nalgebra` crate.
    "nalgebra",
    // `config`: `miniconf` control-plane settings for runtime-tunable filters.
    "miniconf",
];

fn read(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `gated_mod!("feature", module)` / `gated_mod!(math "feature", module)`
/// invocation in `lib.rs`, as `(feature, module)` pairs.
fn gated_modules(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(start) = rest.find("gated_mod!(") {
        rest = &rest[start + "gated_mod!(".len()..];
        let end = rest.find(");").expect("unterminated gated_mod! invocation");
        let call = &rest[..end];

        let open = call.find('"').expect("gated_mod! feature literal");
        let close = call[open + 1..]
            .find('"')
            .expect("gated_mod! closing quote")
            + open
            + 1;
        let feature = call[open + 1..close].to_string();

        let module = call
            .rsplit(',')
            .next()
            .expect("gated_mod! module argument")
            .trim()
            .to_string();
        assert!(
            !module.is_empty() && !module.contains('"'),
            "could not parse the module name out of `gated_mod!({call})`"
        );

        out.push((feature, module));
        rest = &rest[end..];
    }
    out
}

/// The entries of the `full = [...]` array in `Cargo.toml`.
fn full_features(manifest: &str) -> BTreeSet<String> {
    let start = manifest
        .find("full = [")
        .expect("a `full = [...]` feature in Cargo.toml")
        + "full = [".len();
    let end = manifest[start..].find(']').expect("closing `]` for `full`") + start;
    manifest[start..end]
        .split(',')
        .map(|entry| entry.trim().trim_matches('"'))
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn every_gated_module_is_in_full_or_an_opt_in_feature() {
    let modules = gated_modules(&read("src/lib.rs"));
    assert!(!modules.is_empty(), "found no `gated_mod!` invocations");

    let full = full_features(&read("Cargo.toml"));
    let mut missing = Vec::new();
    for (feature, module) in &modules {
        if !full.contains(feature) && !OPT_IN.contains(&feature.as_str()) {
            missing.push(format!("  {module}  (feature `{feature}`)"));
        }
    }

    assert!(
        missing.is_empty(),
        "these feature-gated modules are neither in the `full` feature nor the \
         OPT_IN allowlist in tests/feature_manifest.rs:\n{}\n\n\
         Add the feature to `full = [...]` in crates/embedded-dsp/Cargo.toml, or \
         add it to OPT_IN with a comment explaining why it is opt-in.",
        missing.join("\n")
    );
}
