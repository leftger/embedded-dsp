//! Live Signal Analyzer View: FFT Scope, Oscilloscope, Goniometer & Metering.
//!
//! Inspired by DSP-Testbench: simultaneous log-frequency FFT spectrum analyzer,
//! time-domain oscilloscope, 45-degree Goniometer Lissajous phase correlation scope,
//! and peak/RMS level meters.

use crate::state::StudioState;
use crate::widgets::fft_plot::FftPlotWidget;
use crate::widgets::goniometer::GoniometerWidget;
use crate::widgets::meter::LevelMeterWidget;
use crate::widgets::scope_plot::ScopePlotWidget;
use eframe::egui;

pub struct AnalyzerView {
    pub show_ch_a: bool,
    pub show_ch_b: bool,
    pub show_null: bool,
    pub scope_zoom: usize,
}

impl Default for AnalyzerView {
    fn default() -> Self {
        Self {
            show_ch_a: true,
            show_ch_b: true,
            show_null: false,
            scope_zoom: 512,
        }
    }
}

impl AnalyzerView {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("📊 3. Real-Time Spectral & Multi-Scope Analyzer");

            ui.separator();
            ui.checkbox(&mut self.show_ch_a, "🟢 Ch A");
            ui.checkbox(&mut self.show_ch_b, "🟡 Ch B");
            ui.checkbox(&mut self.show_null, "🔴 Null Diff");

            ui.separator();
            ui.label("Scope Window:");
            ui.add(
                egui::Slider::new(&mut self.scope_zoom, 64..=2048)
                    .logarithmic(true)
                    .suffix(" samples"),
            );
        });

        ui.add_space(6.0);

        // 1. Log-Frequency FFT Spectrum Analyzer (Top Half)
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("📈 Logarithmic Frequency FFT Spectrum (20 Hz - 20 kHz)")
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label("Cursor: hover over graph for frequency & dB");
                });
            });

            let mut fft_widget = FftPlotWidget::new(
                state.sample_rate,
                &state.fft_mag_a,
                &state.fft_mag_b,
                &state.fft_mag_null,
            );
            fft_widget.show_channel_a = self.show_ch_a;
            fft_widget.show_channel_b = self.show_ch_b;
            fft_widget.show_null_diff = self.show_null;
            fft_widget.ui(ui, 240.0);
        });

        ui.add_space(8.0);

        // 2. Lower Split Section: Oscilloscope + Goniometer + Metering
        ui.columns(3, |cols| {
            // Column 1: Time-Domain Oscilloscope (width 50%)
            cols[0].group(|ui| {
                ui.label(egui::RichText::new("🌊 Time-Domain Oscilloscope").strong());
                let mut scope = ScopePlotWidget::new(
                    state.sample_rate,
                    &state.time_a,
                    &state.time_b,
                    &state.time_null,
                );
                scope.show_channel_a = self.show_ch_a;
                scope.show_channel_b = self.show_ch_b;
                scope.show_null_diff = self.show_null;
                scope.zoom_factor = self.scope_zoom;
                scope.ui(ui, 190.0);
            });

            // Column 2: 45° Goniometer / Phase Scope
            cols[1].group(|ui| {
                ui.label(egui::RichText::new("🌐 Phase Scope (Goniometer)").strong());
                ui.label("Ch A vs Ch B Lissajous correlation");

                let gon = GoniometerWidget::new(&state.time_a, &state.time_b);
                gon.ui(ui, 160.0);
            });

            // Column 3: Peak / RMS Level Meters & Monitoring
            cols[2].group(|ui| {
                ui.label(egui::RichText::new("🔊 Metering & Monitor").strong());

                ui.horizontal(|ui| {
                    // Meter A
                    ui.vertical(|ui| {
                        ui.label("Ch A");
                        LevelMeterWidget::new("A", state.vu_peak_a, state.vu_rms_a)
                            .ui(ui, 22.0, 100.0);
                        let db_a = if state.vu_rms_a > 1e-4 {
                            20.0 * state.vu_rms_a.log10()
                        } else {
                            -60.0
                        };
                        ui.label(format!("{:.0}dB", db_a));
                    });

                    ui.add_space(4.0);

                    // Meter B
                    ui.vertical(|ui| {
                        ui.label("Ch B");
                        LevelMeterWidget::new("B", state.vu_peak_b, state.vu_rms_b)
                            .ui(ui, 22.0, 100.0);
                        let db_b = if state.vu_rms_b > 1e-4 {
                            20.0 * state.vu_rms_b.log10()
                        } else {
                            -60.0
                        };
                        ui.label(format!("{:.0}dB", db_b));
                    });

                    ui.add_space(4.0);

                    // Master Meter
                    ui.vertical(|ui| {
                        ui.label("Master");
                        LevelMeterWidget::new("M", state.vu_peak_master, state.vu_rms_master)
                            .ui(ui, 22.0, 100.0);
                        let db_m = if state.vu_rms_master > 1e-4 {
                            20.0 * state.vu_rms_master.log10()
                        } else {
                            -60.0
                        };
                        ui.label(format!("{:.0}dB", db_m));
                    });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Gain:");
                    if ui
                        .add(egui::Slider::new(&mut state.master_gain, 0.0..=2.0))
                        .changed()
                    {
                        state.recompute_all();
                    }
                });

                ui.horizontal(|ui| {
                    if ui.checkbox(&mut state.master_mute, "Mute").changed() {
                        state.recompute_all();
                    }
                    if ui.checkbox(&mut state.limiter, "Limiter").changed() {
                        state.recompute_all();
                    }
                });
            });
        });
    }
}
