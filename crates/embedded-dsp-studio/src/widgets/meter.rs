//! Peak and RMS Audio Level Meter Widget.
//!
//! Inspired by DSP-Testbench's Metering: RMS (wide bar) and Peak (indicator)
//! with dB markings and clip indicator.

use eframe::egui;

pub struct LevelMeterWidget<'a> {
    pub label: &'a str,
    pub peak: f32,
    pub rms: f32,
}

impl<'a> LevelMeterWidget<'a> {
    pub fn new(label: &'a str, peak: f32, rms: f32) -> Self {
        Self { label, peak, rms }
    }

    pub fn ui(self, ui: &mut egui::Ui, width: f32, height: f32) {
        let desired_size = egui::vec2(width, height);
        let (_response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
        let rect = _response.rect;

        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(12, 16, 22));
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(32, 40, 56)),
            egui::StrokeKind::Outside,
        );

        let min_db = -60.0f32;
        let max_db = 0.0f32;
        let to_norm = |val: f32| -> f32 {
            if val <= 1e-4 {
                0.0
            } else {
                let db = 20.0 * val.log10();
                ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
            }
        };

        let rms_norm = to_norm(self.rms);
        let peak_norm = to_norm(self.peak);

        // RMS filled bar
        let bar_h = rect.height() * rms_norm;
        if bar_h > 0.0 {
            let bar_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 2.0, rect.bottom() - bar_h - 2.0),
                egui::vec2(rect.width() - 4.0, bar_h),
            );

            let color = if self.rms >= 0.95 {
                egui::Color32::from_rgb(255, 100, 60)
            } else if self.rms >= 0.7 {
                egui::Color32::from_rgb(240, 200, 50)
            } else {
                egui::Color32::from_rgb(60, 200, 120)
            };

            painter.rect_filled(bar_rect, 1.0, color);
        }

        // Peak line indicator
        let peak_y = rect.bottom() - rect.height() * peak_norm;
        let is_clipped = self.peak >= 1.0;
        let peak_color = if is_clipped {
            egui::Color32::from_rgb(255, 40, 40)
        } else {
            egui::Color32::from_rgb(255, 255, 255)
        };
        painter.line_segment(
            [
                egui::pos2(rect.left(), peak_y),
                egui::pos2(rect.right(), peak_y),
            ],
            egui::Stroke::new(2.0, peak_color),
        );

        // Clip badge at top
        let clip_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left() + 2.0, rect.top() + 2.0),
            egui::vec2(rect.width() - 4.0, 6.0),
        );
        painter.rect_filled(
            clip_rect,
            1.0,
            if is_clipped {
                egui::Color32::from_rgb(255, 40, 40)
            } else {
                egui::Color32::from_rgb(40, 20, 20)
            },
        );

        if !self.label.is_empty() {
            painter.text(
                egui::pos2(rect.center().x, rect.bottom() - 2.0),
                egui::Align2::CENTER_BOTTOM,
                self.label,
                egui::FontId::monospace(8.0),
                egui::Color32::from_rgb(100, 120, 140),
            );
        }
    }
}
