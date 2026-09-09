//! Goniometer / Lissajous Phase Scope Widget.
//!
//! Inspired by DSP-Testbench's Goniometer: 45-degree Mid/Side rotated phase correlation plot.
//! When Channel A == Channel B (in phase), it forms a vertical line.
//! When out of phase, it widens into an ellipse or horizontal line.

use eframe::egui;

pub struct GoniometerWidget<'a> {
    pub channel_a: &'a [f32],
    pub channel_b: &'a [f32],
}

impl<'a> GoniometerWidget<'a> {
    pub fn new(channel_a: &'a [f32], channel_b: &'a [f32]) -> Self {
        Self {
            channel_a,
            channel_b,
        }
    }

    pub fn ui(self, ui: &mut egui::Ui, size: f32) {
        let desired_size = egui::vec2(size, size);
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
        let rect = response.rect;
        let center = rect.center();
        let radius = (rect.width().min(rect.height()) * 0.45).max(10.0);

        // Circular scope background
        painter.circle_filled(center, radius, egui::Color32::from_rgb(10, 13, 18));
        painter.circle_stroke(
            center,
            radius,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(32, 40, 56)),
        );

        // Crosshairs: Vertical (M: A+B) and Horizontal (S: A-B)
        painter.line_segment(
            [
                egui::pos2(center.x, center.y - radius),
                egui::pos2(center.x, center.y + radius),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(30, 42, 60)),
        );
        painter.line_segment(
            [
                egui::pos2(center.x - radius, center.y),
                egui::pos2(center.x + radius, center.y),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(30, 42, 60)),
        );

        // Diagonal Axis Markers (L and R)
        let diag_offset = radius * 0.7071;
        painter.line_segment(
            [
                egui::pos2(center.x - diag_offset, center.y + diag_offset),
                egui::pos2(center.x + diag_offset, center.y - diag_offset),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(22, 30, 44)),
        );
        painter.line_segment(
            [
                egui::pos2(center.x - diag_offset, center.y - diag_offset),
                egui::pos2(center.x + diag_offset, center.y + diag_offset),
            ],
            egui::Stroke::new(0.5, egui::Color32::from_rgb(22, 30, 44)),
        );

        painter.text(
            egui::pos2(center.x, center.y - radius + 4.0),
            egui::Align2::CENTER_TOP,
            "+M (In-Phase)",
            egui::FontId::monospace(8.0),
            egui::Color32::from_rgb(90, 110, 135),
        );
        painter.text(
            egui::pos2(center.x + radius - 4.0, center.y),
            egui::Align2::RIGHT_CENTER,
            "+S",
            egui::FontId::monospace(8.0),
            egui::Color32::from_rgb(90, 110, 135),
        );

        // Plot M/S Lissajous points
        let n = self.channel_a.len().min(self.channel_b.len()).min(1024);
        if n >= 2 {
            let inv_sqrt2 = 0.70710678f32;
            let mut points = Vec::with_capacity(n);

            for i in 0..n {
                let a = self.channel_a[i];
                let b = self.channel_b[i];

                // Rotate 45 degrees:
                // M = (A + B) / sqrt(2) -> vertical axis
                // S = (A - B) / sqrt(2) -> horizontal axis
                let s = (a - b) * inv_sqrt2;
                let m = (a + b) * inv_sqrt2;

                let px = center.x + s * radius;
                let py = center.y - m * radius; // Invert y because screen Y is top-down

                points.push(egui::pos2(
                    px.clamp(center.x - radius, center.x + radius),
                    py.clamp(center.y - radius, center.y + radius),
                ));
            }

            // Draw connecting glow trace
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(80, 220, 180, 160),
                ),
            ));
        }
    }
}
