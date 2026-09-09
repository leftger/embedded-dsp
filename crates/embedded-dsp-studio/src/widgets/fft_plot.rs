//! Log-Frequency Multi-Channel FFT Spectrum Analyzer Widget.
//!
//! Inspired by DSP-Testbench's FftScope: logarithmic frequency scale (20 Hz - 20 kHz),
//! calibrated dB amplitude range, standard octave grid markings, multi-channel color overlays,
//! and interactive cursor tooltip with exact frequency and magnitude readout.

use eframe::egui;

pub struct FftPlotWidget<'a> {
    pub sample_rate: f32,
    pub mag_a: &'a [f32],
    pub mag_b: &'a [f32],
    pub mag_null: &'a [f32],
    pub show_channel_a: bool,
    pub show_channel_b: bool,
    pub show_null_diff: bool,
}

impl<'a> FftPlotWidget<'a> {
    pub fn new(sample_rate: f32, mag_a: &'a [f32], mag_b: &'a [f32], mag_null: &'a [f32]) -> Self {
        Self {
            sample_rate,
            mag_a,
            mag_b,
            mag_null,
            show_channel_a: true,
            show_channel_b: true,
            show_null_diff: false,
        }
    }

    pub fn ui(self, ui: &mut egui::Ui, height: f32) {
        let desired_size = egui::vec2(ui.available_width(), height);
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
        let rect = response.rect;

        // Background
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(10, 13, 18));
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(32, 40, 56)),
            egui::StrokeKind::Outside,
        );

        let min_freq = 20.0f32;
        let max_freq = (self.sample_rate * 0.5).min(22050.0);
        let min_log = min_freq.log10();
        let max_log = max_freq.log10();
        let log_span = (max_log - min_log).max(0.1);

        let min_db = -100.0f32;
        let max_db = 10.0f32;
        let db_span = max_db - min_db;

        let to_screen_x = |freq: f32| -> f32 {
            let clamped = freq.clamp(min_freq, max_freq);
            let norm = (clamped.log10() - min_log) / log_span;
            rect.left() + norm * rect.width()
        };

        let to_screen_y = |db: f32| -> f32 {
            let norm = (db.clamp(min_db, max_db) - min_db) / db_span;
            rect.bottom() - norm * rect.height()
        };

        let to_freq = |screen_x: f32| -> f32 {
            let norm = ((screen_x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            10.0f32.powf(min_log + norm * log_span)
        };

        let to_db = |screen_y: f32| -> f32 {
            let norm = ((rect.bottom() - screen_y) / rect.height()).clamp(0.0, 1.0);
            min_db + norm * db_span
        };

        // 1. Horizontal dB Grid Lines
        let db_ticks = [0.0, -20.0, -40.0, -60.0, -80.0];
        for &db in &db_ticks {
            let y = to_screen_y(db);
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(0.5, egui::Color32::from_rgb(25, 34, 48)),
            );
            painter.text(
                egui::pos2(rect.left() + 4.0, y - 2.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.0} dB", db),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(100, 115, 135),
            );
        }

        // 2. Vertical Log Frequency Grid Lines
        let grid_freqs = [
            (50.0, "50"),
            (100.0, "100"),
            (250.0, "250"),
            (500.0, "500"),
            (1000.0, "1k"),
            (2000.0, "2k"),
            (5000.0, "5k"),
            (10000.0, "10k"),
            (20000.0, "20k"),
        ];

        for &(f, label) in &grid_freqs {
            if f >= min_freq && f <= max_freq {
                let x = to_screen_x(f);
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(0.5, egui::Color32::from_rgb(25, 34, 48)),
                );
                painter.text(
                    egui::pos2(x + 2.0, rect.bottom() - 4.0),
                    egui::Align2::LEFT_BOTTOM,
                    label,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(100, 115, 135),
                );
            }
        }

        // 3. Draw Channel Curves
        let draw_curve = |mags: &[f32], color: egui::Color32, stroke_width: f32| {
            let num_bins = mags.len();
            if num_bins < 2 {
                return;
            }

            let nyquist = self.sample_rate * 0.5;
            let bin_hz = nyquist / (num_bins as f32);

            let mut points: Vec<egui::Pos2> = Vec::with_capacity((rect.width() as usize) + 1);
            let mut prev_px = -1i32;
            let mut max_db_in_px = -120.0f32;

            for i in 1..num_bins {
                let freq = (i as f32) * bin_hz;
                if freq < min_freq {
                    continue;
                }
                if freq > max_freq {
                    break;
                }

                let px = to_screen_x(freq) as i32;
                let db = mags[i];

                if px == prev_px {
                    max_db_in_px = max_db_in_px.max(db);
                } else {
                    if prev_px >= 0 {
                        points.push(egui::pos2(prev_px as f32, to_screen_y(max_db_in_px)));
                    }
                    prev_px = px;
                    max_db_in_px = db;
                }
            }

            if prev_px >= 0 {
                points.push(egui::pos2(prev_px as f32, to_screen_y(max_db_in_px)));
            }

            if points.len() >= 2 {
                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(stroke_width, color),
                ));
            }
        };

        if self.show_null_diff {
            draw_curve(self.mag_null, egui::Color32::from_rgb(255, 70, 70), 1.5);
        }
        if self.show_channel_b {
            draw_curve(self.mag_b, egui::Color32::from_rgb(240, 180, 50), 1.5);
        }
        if self.show_channel_a {
            draw_curve(self.mag_a, egui::Color32::from_rgb(80, 210, 150), 1.5);
        }

        // 4. Interactive Cursor Tooltip
        if let Some(mouse_pos) = response.hover_pos() {
            if rect.contains(mouse_pos) {
                // Crosshairs
                painter.line_segment(
                    [
                        egui::pos2(mouse_pos.x, rect.top()),
                        egui::pos2(mouse_pos.x, rect.bottom()),
                    ],
                    egui::Stroke::new(0.8, egui::Color32::from_rgb(120, 150, 190)),
                );
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), mouse_pos.y),
                        egui::pos2(rect.right(), mouse_pos.y),
                    ],
                    egui::Stroke::new(0.8, egui::Color32::from_rgb(120, 150, 190)),
                );

                let cursor_freq = to_freq(mouse_pos.x);
                let cursor_db = to_db(mouse_pos.y);

                let freq_label = if cursor_freq >= 1000.0 {
                    format!("{:.2} kHz", cursor_freq / 1000.0)
                } else {
                    format!("{:.0} Hz", cursor_freq)
                };
                let tooltip_text = format!("{} | {:.1} dB", freq_label, cursor_db);

                let text_pos = egui::pos2(
                    (mouse_pos.x + 8.0).min(rect.right() - 110.0),
                    (mouse_pos.y - 18.0).max(rect.top() + 4.0),
                );

                painter.rect_filled(
                    egui::Rect::from_min_size(text_pos, egui::vec2(105.0, 18.0)),
                    3.0,
                    egui::Color32::from_rgba_unmultiplied(16, 20, 30, 220),
                );
                painter.text(
                    text_pos + egui::vec2(4.0, 2.0),
                    egui::Align2::LEFT_TOP,
                    tooltip_text,
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(220, 235, 255),
                );
            }
        }
    }
}
