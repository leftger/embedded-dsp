//! Interactive Time-Domain Multi-Channel Oscilloscope Widget.
//!
//! Inspired by DSP-Testbench's Oscilloscope: multi-channel overlay, amplitude scaling,
//! sample index / time axis, zero crossing guide, and peak display.

use eframe::egui;

pub struct ScopePlotWidget<'a> {
    pub sample_rate: f32,
    pub time_a: &'a [f32],
    pub time_b: &'a [f32],
    pub time_null: &'a [f32],
    pub show_channel_a: bool,
    pub show_channel_b: bool,
    pub show_null_diff: bool,
    pub zoom_factor: usize,
}

impl<'a> ScopePlotWidget<'a> {
    pub fn new(
        sample_rate: f32,
        time_a: &'a [f32],
        time_b: &'a [f32],
        time_null: &'a [f32],
    ) -> Self {
        Self {
            sample_rate,
            time_a,
            time_b,
            time_null,
            show_channel_a: true,
            show_channel_b: true,
            show_null_diff: false,
            zoom_factor: 512,
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

        let mid_y = rect.center().y;
        let amp_scale = (rect.height() * 0.45).max(10.0);

        // Center zero line
        painter.line_segment(
            [
                egui::pos2(rect.left(), mid_y),
                egui::pos2(rect.right(), mid_y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 75)),
        );

        // +1.0 and -1.0 guidelines
        let y_plus1 = mid_y - amp_scale;
        let y_minus1 = mid_y + amp_scale;
        painter.line_segment(
            [
                egui::pos2(rect.left(), y_plus1),
                egui::pos2(rect.right(), y_plus1),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(25, 34, 48)),
        );
        painter.line_segment(
            [
                egui::pos2(rect.left(), y_minus1),
                egui::pos2(rect.right(), y_minus1),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(25, 34, 48)),
        );

        painter.text(
            egui::pos2(rect.left() + 4.0, y_plus1 - 2.0),
            egui::Align2::LEFT_BOTTOM,
            "+1.0",
            egui::FontId::monospace(9.0),
            egui::Color32::from_rgb(80, 95, 115),
        );
        painter.text(
            egui::pos2(rect.left() + 4.0, y_minus1 + 2.0),
            egui::Align2::LEFT_TOP,
            "-1.0",
            egui::FontId::monospace(9.0),
            egui::Color32::from_rgb(80, 95, 115),
        );

        let num_samples = self.time_a.len().min(self.zoom_factor).max(2);
        let dx = rect.width() / ((num_samples - 1) as f32);

        let draw_channel = |samples: &[f32], color: egui::Color32, stroke_width: f32| {
            let n = samples.len().min(num_samples);
            if n < 2 {
                return;
            }

            let mut points = Vec::with_capacity(n);
            for i in 0..n {
                let x = rect.left() + (i as f32) * dx;
                let y = mid_y - samples[i] * amp_scale;
                points.push(egui::pos2(x, y.clamp(rect.top(), rect.bottom())));
            }

            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(stroke_width, color),
            ));
        };

        if self.show_null_diff {
            draw_channel(self.time_null, egui::Color32::from_rgb(255, 70, 70), 1.5);
        }
        if self.show_channel_b {
            draw_channel(self.time_b, egui::Color32::from_rgb(240, 180, 50), 1.5);
        }
        if self.show_channel_a {
            draw_channel(self.time_a, egui::Color32::from_rgb(80, 210, 150), 1.5);
        }

        // Time / Sample Duration Readout
        let duration_ms = (num_samples as f32) / self.sample_rate * 1000.0;
        let badge_text = format!("Window: {} samples ({:.2} ms)", num_samples, duration_ms);
        painter.text(
            egui::pos2(rect.right() - 6.0, rect.top() + 6.0),
            egui::Align2::RIGHT_TOP,
            badge_text,
            egui::FontId::monospace(10.0),
            egui::Color32::from_rgb(140, 160, 185),
        );
    }
}
