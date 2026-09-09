//! Dual Processor Harness & Null Cancellation Testing View.
//!
//! Inspired by DSP-Testbench: host and control two independent DSP processing pipelines,
//! route signals, and perform A/B null cancellation tests to verify bit-exactness,
//! assess SIMD/fixed-point optimizations, and quantify distortion.

use crate::state::{ProcessorConfig, ProcessorMode, StudioState};
use crate::widgets::scope_plot::ScopePlotWidget;
use eframe::egui;

#[derive(Default)]
pub struct ProcessorsView {}

impl ProcessorsView {
    pub fn new() -> Self {
        Self {}
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("🎛️ 2. Processor Workbench & A/B Null Test Harness");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("🎯 Configure Exact Null Cancellation Test")
                    .clicked()
                {
                    state.load_preset(crate::state::StudioPreset::NullCancellationTest);
                }
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Host and compare two DSP algorithms concurrently. Use A/B Null-Testing (Slot A - Slot B) \
            to measure difference cancellation down to -120 dB, verifying that fixed-point or algorithmic \
            optimizations produce identical results with zero alteration.",
        );
        ui.add_space(8.0);

        let mut changed = false;

        // Side-by-side columns for Processor Slot A and Processor Slot B
        ui.columns(2, |cols| {
            // Processor Slot A
            cols[0].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("💠 Processor Slot A")
                            .strong()
                            .color(egui::Color32::from_rgb(80, 210, 150)),
                    );
                });
                ui.separator();

                changed |=
                    render_processor_controls(ui, &mut state.proc_a, "proc_a", state.sample_rate);
            });

            // Processor Slot B
            cols[1].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🔶 Processor Slot B")
                            .strong()
                            .color(egui::Color32::from_rgb(240, 180, 50)),
                    );
                });
                ui.separator();

                changed |=
                    render_processor_controls(ui, &mut state.proc_b, "proc_b", state.sample_rate);
            });
        });

        ui.add_space(8.0);

        // Null Test and Comparison Panel
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("🎯 A/B Null-Test Cancellation Monitor").strong());
                ui.separator();

                if ui
                    .checkbox(
                        &mut state.null_test_mode,
                        "Enable Master Null Output (A - B)",
                    )
                    .changed()
                {
                    changed = true;
                }

                ui.separator();

                // Compute peak difference and cancellation in dB
                let mut max_diff = 0.0f32;
                let mut sum_diff_sq = 0.0f32;
                for &d in &state.time_null {
                    let a = d.abs();
                    if a > max_diff {
                        max_diff = a;
                    }
                    sum_diff_sq += d * d;
                }
                let rms_diff = (sum_diff_sq / (state.time_null.len() as f32).max(1.0)).sqrt();
                let cancel_db = if rms_diff > 1e-6 {
                    20.0 * rms_diff.log10()
                } else {
                    -120.0
                };

                ui.label(format!("Peak Diff: {:.6}", max_diff));
                ui.label("•");
                ui.label(format!("RMS Diff: {:.6}", rms_diff));
                ui.label("•");

                let (badge_color, badge_text) = if cancel_db <= -100.0 {
                    (
                        egui::Color32::from_rgb(60, 220, 120),
                        format!("Cancellation: {:.1} dB (PERFECT NULL)", cancel_db),
                    )
                } else if cancel_db <= -40.0 {
                    (
                        egui::Color32::from_rgb(100, 180, 255),
                        format!("Cancellation: {:.1} dB", cancel_db),
                    )
                } else {
                    (
                        egui::Color32::from_rgb(255, 180, 60),
                        format!("Residual: {:.1} dB", cancel_db),
                    )
                };

                ui.colored_label(badge_color, badge_text);
            });
        });

        ui.add_space(8.0);

        // Difference Waveform Scope
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("🔬 Time-Domain Waveforms (Slot A, Slot B, Difference)")
                        .strong(),
                );
            });

            let mut scope = ScopePlotWidget::new(
                state.sample_rate,
                &state.time_a,
                &state.time_b,
                &state.time_null,
            );
            scope.show_null_diff = true;
            scope.ui(ui, 140.0);
        });

        if changed {
            state.recompute_all();
        }
    }
}

fn render_processor_controls(
    ui: &mut egui::Ui,
    proc: &mut ProcessorConfig,
    id_salt: &str,
    sample_rate: f32,
) -> bool {
    let mut changed = false;

    // Algorithm Mode
    ui.horizontal(|ui| {
        ui.label("Algorithm:");
        let prev_mode = proc.mode;
        egui::ComboBox::from_id_salt(format!("{}_mode", id_salt))
            .selected_text(proc.mode.display_name())
            .show_ui(ui, |ui| {
                for &m in ProcessorMode::ALL {
                    ui.selectable_value(&mut proc.mode, m, m.display_name());
                }
            });
        if proc.mode != prev_mode {
            changed = true;
        }
    });

    ui.add_space(4.0);

    let has_cutoff = !matches!(
        proc.mode,
        ProcessorMode::Bypass | ProcessorMode::FixedPointQ15
    );
    if has_cutoff {
        ui.horizontal(|ui| {
            ui.label("Cutoff / Center:");
            let max_c = (sample_rate * 0.45).min(20000.0);
            if ui
                .add(
                    egui::Slider::new(&mut proc.cutoff_hz, 20.0..=max_c)
                        .logarithmic(true)
                        .suffix(" Hz"),
                )
                .changed()
            {
                changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Q / Resonance:");
            if ui
                .add(
                    egui::Slider::new(&mut proc.q, 0.1..=20.0)
                        .logarithmic(true)
                        .step_by(0.01),
                )
                .changed()
            {
                changed = true;
            }
        });
    }

    if matches!(proc.mode, ProcessorMode::BiquadPeaking) {
        ui.horizontal(|ui| {
            ui.label("Boost / Cut Gain:");
            if ui
                .add(egui::Slider::new(&mut proc.gain_db, -24.0..=24.0).suffix(" dB"))
                .changed()
            {
                changed = true;
            }
        });
    }

    if matches!(
        proc.mode,
        ProcessorMode::SvfLowpass
            | ProcessorMode::SvfHighpass
            | ProcessorMode::SvfBandpass
            | ProcessorMode::SvfNotch
            | ProcessorMode::SvfPeak
    ) {
        ui.horizontal(|ui| {
            ui.label("Nonlinear Drive:");
            if ui
                .add(egui::Slider::new(&mut proc.drive, 0.0..=2.0).step_by(0.05))
                .changed()
            {
                changed = true;
            }
        });
    }

    if matches!(proc.mode, ProcessorMode::FixedPointQ15) {
        ui.label(
            "Emulating 16-bit Q15 saturation and truncation. Compare against Slot A to inspect quantization floor.",
        );
    }

    ui.horizontal(|ui| {
        if ui.checkbox(&mut proc.mute, "Mute").changed() {
            changed = true;
        }
        if ui.checkbox(&mut proc.invert, "Invert Output (Ø)").changed() {
            changed = true;
        }
    });

    changed
}
