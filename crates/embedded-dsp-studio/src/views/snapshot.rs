//! Forensic Snapshot & Impulse Response View.
//!
//! Inspired by DSP-Testbench: freeze and capture a deterministic 4096-sample
//! impulse or step response buffer to forensically measure peak gain, ringing,
//! settling time, energy, and filter stability.

use crate::state::StudioState;
use eframe::egui;

#[derive(Default)]
pub struct SnapshotView {
    pub settling_threshold: f32,
}

impl SnapshotView {
    pub fn new() -> Self {
        Self {
            settling_threshold: 0.01,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("🔍 4. Forensic Snapshot & Impulse Response Analyzer");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("📸 Capture 4096-Sample Impulse Snapshot")
                    .clicked()
                {
                    state.trigger_snapshot(self.settling_threshold);
                }
                ui.separator();
                ui.label("Threshold:");
                ui.add(
                    egui::DragValue::new(&mut self.settling_threshold)
                        .range(0.0001..=0.1)
                        .speed(0.001),
                );
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Injects a deterministic Dirac delta impulse [δ(n)] into Processor Slot A, freezing a 4096-sample \
            window. Accurately measures filter settling time, damping, resonance ringing, and numerical stability.",
        );
        ui.add_space(6.0);

        if state.snapshot_captured {
            ui.horizontal(|ui| {
                if ui.button("📥 Export Snapshot WAV (16-bit PCM)").clicked() {
                    let wav = crate::export::export_wav_16bit(
                        state.snapshot_buffer.samples(),
                        state.sample_rate as u32,
                    );
                    crate::export::save_or_download_file("impulse_snapshot.wav", "audio/wav", &wav);
                }
                if ui.button("📊 Export Snapshot CSV").clicked() {
                    let csv = crate::export::export_csv(
                        state.snapshot_buffer.samples(),
                        state.sample_rate,
                    );
                    crate::export::save_or_download_file(
                        "impulse_snapshot.csv",
                        "text/csv",
                        csv.as_bytes(),
                    );
                }
            });
            ui.add_space(6.0);
        }

        // Analysis Metric Cards
        if let Some(info) = state.snapshot_response_info {
            ui.columns(4, |cols| {
                cols[0].group(|ui| {
                    ui.label("Peak Impulse Gain");
                    ui.heading(format!(
                        "{:.4} ({:+.2} dB)",
                        info.peak_gain,
                        20.0 * info.peak_gain.max(1e-5).log10()
                    ));
                });

                cols[1].group(|ui| {
                    ui.label("Settling Time");
                    let ms = (info.settling_time_samples as f32) / state.sample_rate * 1000.0;
                    ui.heading(format!("{} spl ({:.2} ms)", info.settling_time_samples, ms));
                });

                cols[2].group(|ui| {
                    ui.label("Total Signal Energy (∑h²)");
                    ui.heading(format!("{:.4}", info.total_energy));
                });

                cols[3].group(|ui| {
                    ui.label("Numerical Stability");
                    if info.is_stable {
                        ui.heading(
                            egui::RichText::new("STABLE ✔")
                                .color(egui::Color32::from_rgb(80, 220, 120)),
                        );
                    } else {
                        ui.heading(
                            egui::RichText::new("EXPLOSION / UNSTABLE ⚠")
                                .color(egui::Color32::from_rgb(255, 60, 60)),
                        );
                    }
                });
            });

            ui.add_space(8.0);
        }

        // Captured Waveform Plot with Settling Threshold Line
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("🔬 Frozen 4096-Sample Response Waveform").strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "Samples Captured: {}",
                        state.snapshot_buffer.samples().len()
                    ));
                });
            });

            let samples = state.snapshot_buffer.samples();
            let desired_size = egui::vec2(ui.available_width(), 260.0);
            let (_response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
            let rect = _response.rect;

            // Background
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(10, 13, 18));
            painter.rect_stroke(
                rect,
                4.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(32, 40, 56)),
                egui::StrokeKind::Outside,
            );

            let mid_y = rect.center().y;
            let amp_scale = (rect.height() * 0.42).max(10.0);

            // Zero line
            painter.line_segment(
                [
                    egui::pos2(rect.left(), mid_y),
                    egui::pos2(rect.right(), mid_y),
                ],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 75)),
            );

            let n = samples.len();
            if n > 1 {
                let dx = rect.width() / ((n - 1) as f32);
                let mut points = Vec::with_capacity(n);

                for i in 0..n {
                    let x = rect.left() + (i as f32) * dx;
                    let y = mid_y - samples[i] * amp_scale;
                    points.push(egui::pos2(x, y.clamp(rect.top(), rect.bottom())));
                }

                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(100, 200, 255)),
                ));

                // Mark Settling Point
                if let Some(info) = state.snapshot_response_info {
                    if info.settling_time_samples > 0 && info.settling_time_samples < n {
                        let settle_x = rect.left() + (info.settling_time_samples as f32) * dx;
                        painter.line_segment(
                            [
                                egui::pos2(settle_x, rect.top()),
                                egui::pos2(settle_x, rect.bottom()),
                            ],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 180, 50)),
                        );
                        painter.text(
                            egui::pos2(settle_x + 4.0, rect.top() + 6.0),
                            egui::Align2::LEFT_TOP,
                            format!("Settled @ n={}", info.settling_time_samples),
                            egui::FontId::monospace(10.0),
                            egui::Color32::from_rgb(255, 180, 50),
                        );
                    }
                }
            } else {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "No snapshot captured yet. Click 'Capture 4096-Sample Impulse Snapshot'.",
                    egui::FontId::proportional(14.0),
                    egui::Color32::from_rgb(100, 115, 135),
                );
            }
        });
    }
}
