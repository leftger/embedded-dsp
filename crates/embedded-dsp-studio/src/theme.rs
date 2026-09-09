//! Embedded DSP Studio modern dark theme styling.

use eframe::egui;

pub fn configure_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();

    style.visuals.override_text_color = Some(egui::Color32::from_rgb(225, 235, 245));

    style.visuals.window_fill = egui::Color32::from_rgb(16, 19, 26);
    style.visuals.panel_fill = egui::Color32::from_rgb(12, 14, 20);

    style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(20, 24, 34);
    style.visuals.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(38, 46, 64));

    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(25, 30, 44);
    style.visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(45, 55, 78));

    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(38, 48, 70);
    style.visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(80, 160, 240));

    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(50, 80, 130);
    style.visuals.widgets.active.bg_stroke =
        egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(120, 200, 255));

    style.visuals.selection.bg_fill = egui::Color32::from_rgb(60, 110, 190);

    ctx.set_style_of(egui::Theme::Dark, style);
}
