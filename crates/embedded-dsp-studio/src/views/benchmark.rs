//! Micro-Benchmarking & Numerical Comparison View.
//!
//! Inspired by DSP-Testbench: pump blocks of audio through various routines
//! and gather real microsecond latency, cycle times, and throughput statistics.

use crate::state::StudioState;
use eframe::egui;

#[derive(Default)]
pub struct BenchmarkView {}

impl BenchmarkView {
    pub fn new() -> Self {
        Self {}
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("⏱️ 5. Performance Benchmarks & Numerical Accuracy");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🚀 Run DSP Benchmarks").clicked() {
                    state.run_benchmarks();
                }
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Execute direct high-resolution micro-benchmarks on host CPU measuring execution duration, \
            M samples/sec throughput, and speedup ratios of embedded-dsp's fast_math approximations.",
        );
        ui.add_space(8.0);

        ui.group(|ui| {
            ui.label(egui::RichText::new("📋 Benchmark Diagnostics Report").strong());

            egui::ScrollArea::vertical()
                .max_height(350.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut state.last_bench_report.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(12)
                            .interactive(false),
                    );
                });
        });
    }
}
