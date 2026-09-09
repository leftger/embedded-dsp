//! Signal Lab: Dual Signal Generator (Source A & B) Configuration View.

use crate::state::{SignalSourceConfig, SourceRouting, StudioState, WaveformType};
use crate::widgets::scope_plot::ScopePlotWidget;
use eframe::egui;

#[derive(Default)]
pub struct SignalLabView {}

impl SignalLabView {
    pub fn new() -> Self {
        Self {}
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("📡 1. Signal Lab - Dual Test Signal Generator");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("Sample Rate: {:.0} Hz", state.sample_rate));
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Synthesize anti-aliased PolyBLEP periodic oscillators, linear/exponential chirp sweeps, \
            Paul Kellett pink noise (1/f), white noise, and impulse/step signals for comprehensive testing.",
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            if ui.button("📥 Export Source A WAV").clicked() {
                let wav = crate::export::export_wav_16bit(&state.time_a, state.sample_rate as u32);
                crate::export::save_or_download_file("source_a.wav", "audio/wav", &wav);
            }
            if ui.button("📊 Export Source A CSV").clicked() {
                let csv = crate::export::export_csv(&state.time_a, state.sample_rate);
                crate::export::save_or_download_file("source_a.csv", "text/csv", csv.as_bytes());
            }
            ui.separator();
            if ui.button("📥 Export Source B WAV").clicked() {
                let wav = crate::export::export_wav_16bit(&state.time_b, state.sample_rate as u32);
                crate::export::save_or_download_file("source_b.wav", "audio/wav", &wav);
            }
            if ui.button("📊 Export Source B CSV").clicked() {
                let csv = crate::export::export_csv(&state.time_b, state.sample_rate);
                crate::export::save_or_download_file("source_b.csv", "text/csv", csv.as_bytes());
            }
        });
        ui.add_space(6.0);

        let mut changed = false;

        // Side-by-side columns for Source A and Source B
        ui.columns(2, |cols| {
            // Source A Column
            cols[0].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🟡 Signal Source A")
                            .strong()
                            .color(egui::Color32::from_rgb(80, 210, 150)),
                    );
                    ui.checkbox(&mut state.source_a.enabled, "Enabled");
                });
                ui.separator();

                changed |=
                    render_source_controls(ui, &mut state.source_a, "source_a", state.sample_rate);
            });

            // Source B Column
            cols[1].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🟠 Signal Source B")
                            .strong()
                            .color(egui::Color32::from_rgb(240, 180, 50)),
                    );
                    ui.checkbox(&mut state.source_b.enabled, "Enabled");
                });
                ui.separator();

                changed |=
                    render_source_controls(ui, &mut state.source_b, "source_b", state.sample_rate);
            });
        });

        ui.add_space(10.0);

        // Routing Section
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("🔀 Processor Signal Routing Matrix").strong());
                ui.separator();

                ui.label("Feed Processor A from:");
                let prev_ra = state.routing_a;
                egui::ComboBox::from_id_salt("routing_a_combo")
                    .selected_text(state.routing_a.display_name())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.routing_a,
                            SourceRouting::SourceA,
                            "Source A",
                        );
                        ui.selectable_value(
                            &mut state.routing_a,
                            SourceRouting::SourceB,
                            "Source B",
                        );
                        ui.selectable_value(
                            &mut state.routing_a,
                            SourceRouting::SumBoth,
                            "Source A + B (Sum)",
                        );
                    });
                if state.routing_a != prev_ra {
                    changed = true;
                }

                ui.separator();

                ui.label("Feed Processor B from:");
                let prev_rb = state.routing_b;
                egui::ComboBox::from_id_salt("routing_b_combo")
                    .selected_text(state.routing_b.display_name())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.routing_b,
                            SourceRouting::SourceA,
                            "Source A",
                        );
                        ui.selectable_value(
                            &mut state.routing_b,
                            SourceRouting::SourceB,
                            "Source B",
                        );
                        ui.selectable_value(
                            &mut state.routing_b,
                            SourceRouting::SumBoth,
                            "Source A + B (Sum)",
                        );
                    });
                if state.routing_b != prev_rb {
                    changed = true;
                }
            });
        });

        ui.add_space(8.0);

        // Mini oscilloscope preview of generated signals
        ui.group(|ui| {
            ui.label(egui::RichText::new("📈 Real-Time Generated Waveform Preview").strong());
            ScopePlotWidget::new(
                state.sample_rate,
                &state.time_a,
                &state.time_b,
                &state.time_null,
            )
            .ui(ui, 140.0);
        });

        if changed {
            state.recompute_all();
        }
    }
}

fn render_source_controls(
    ui: &mut egui::Ui,
    src: &mut SignalSourceConfig,
    id_salt: &str,
    sample_rate: f32,
) -> bool {
    let mut changed = false;

    // Waveform selector
    ui.horizontal(|ui| {
        ui.label("Waveform:");
        let prev_wave = src.waveform;
        egui::ComboBox::from_id_salt(format!("{}_wave", id_salt))
            .selected_text(src.waveform.display_name())
            .show_ui(ui, |ui| {
                for &wf in WaveformType::ALL {
                    ui.selectable_value(&mut src.waveform, wf, wf.display_name());
                }
            });
        if src.waveform != prev_wave {
            changed = true;
        }
    });

    ui.add_space(4.0);

    // Frequency slider (for periodic waveforms)
    let is_periodic = matches!(
        src.waveform,
        WaveformType::Sine
            | WaveformType::Sawtooth
            | WaveformType::Square
            | WaveformType::Triangle
            | WaveformType::ChirpLinear
            | WaveformType::ChirpExponential
    );

    if is_periodic {
        ui.horizontal(|ui| {
            ui.label("Frequency:");
            let max_f = (sample_rate * 0.49).min(20000.0);
            if ui
                .add(
                    egui::Slider::new(&mut src.frequency_hz, 20.0..=max_f)
                        .logarithmic(true)
                        .suffix(" Hz"),
                )
                .changed()
            {
                changed = true;
            }
        });
    }

    // Amplitude slider
    ui.horizontal(|ui| {
        ui.label("Amplitude:");
        if ui
            .add(egui::Slider::new(&mut src.amplitude, 0.0..=1.0).step_by(0.01))
            .changed()
        {
            changed = true;
        }
    });

    // Phase deg
    if is_periodic {
        ui.horizontal(|ui| {
            ui.label("Phase Offset:");
            if ui
                .add(egui::Slider::new(&mut src.phase_deg, 0.0..=360.0).suffix("°"))
                .changed()
            {
                changed = true;
            }
        });
    }

    // Mute and Invert
    ui.horizontal(|ui| {
        if ui.checkbox(&mut src.mute, "Mute").changed() {
            changed = true;
        }
        if ui.checkbox(&mut src.invert, "Invert Phase (Ø)").changed() {
            changed = true;
        }
    });

    changed
}
