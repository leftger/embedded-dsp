//! Main Application Shell and Tabbed Layout for embedded-dsp studio.

use crate::state::{StudioPreset, StudioState};
use crate::theme::configure_theme;
use crate::views::analyzer::AnalyzerView;
use crate::views::benchmark::BenchmarkView;
use crate::views::codegen::CodegenView;
use crate::views::processors::ProcessorsView;
use crate::views::signal_lab::SignalLabView;
use crate::views::snapshot::SnapshotView;
use eframe::egui;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StudioTab {
    #[default]
    SignalLab,
    Processors,
    Analyzer,
    Snapshot,
    Benchmark,
    Codegen,
}

pub struct EmbeddedDspStudioApp {
    pub current_tab: StudioTab,
    pub state: StudioState,
    pub signal_lab_view: SignalLabView,
    pub processors_view: ProcessorsView,
    pub analyzer_view: AnalyzerView,
    pub snapshot_view: SnapshotView,
    pub benchmark_view: BenchmarkView,
    pub codegen_view: CodegenView,
}

impl Default for EmbeddedDspStudioApp {
    fn default() -> Self {
        Self {
            current_tab: StudioTab::Analyzer, // Start with the visual analyzer for immediate feedback
            state: StudioState::default(),
            signal_lab_view: SignalLabView::new(),
            processors_view: ProcessorsView::new(),
            analyzer_view: AnalyzerView::new(),
            snapshot_view: SnapshotView::new(),
            benchmark_view: BenchmarkView::new(),
            codegen_view: CodegenView::new(),
        }
    }
}

impl eframe::App for EmbeddedDspStudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        egui::Panel::top("dsp_studio_top_header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🎛️ embedded-dsp studio");
                ui.label("• Embedded DSP Workbench & Testbench");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::widgets::global_theme_preference_buttons(ui);
                });
            });

            ui.add_space(4.0);

            // Presets and Action Bar
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("top_presets_combo")
                    .selected_text("🏛️ Testbench Workflow Presets")
                    .show_ui(ui, |ui| {
                        for &preset in StudioPreset::ALL {
                            if ui
                                .button(preset.title())
                                .on_hover_text(preset.description())
                                .clicked()
                            {
                                self.state.load_preset(preset);
                            }
                        }
                    });

                ui.separator();

                if ui
                    .button("🎯 Quick A/B Null Test")
                    .on_hover_text("Configure identical filters with opposite polarity to verify zero cancellation.")
                    .clicked()
                {
                    self.state.load_preset(StudioPreset::NullCancellationTest);
                    self.current_tab = StudioTab::Processors;
                }

                if ui
                    .button("🔬 Q15 Fixed-Point Noise")
                    .on_hover_text("Compare Float32 vs Q15 fixed-point truncation noise.")
                    .clicked()
                {
                    self.state.load_preset(StudioPreset::FloatVsFixedQ15);
                    self.current_tab = StudioTab::Analyzer;
                }

                if ui
                    .button("📸 Impulse Snapshot")
                    .on_hover_text("Evaluate Dirac impulse response and settling time.")
                    .clicked()
                {
                    self.state.load_preset(StudioPreset::ImpulseResponseForensic);
                    self.current_tab = StudioTab::Snapshot;
                }
            });

            ui.add_space(4.0);

            // Workflow Step Navigation Bar
            ui.horizontal(|ui| {
                let tab_1 = format!("1. Signal Lab ({:?})", self.state.source_a.waveform);
                let tab_2 = format!("2. Processors ({:?})", self.state.proc_a.mode);
                let tab_3 = "3. 📊 Live Analyzer".to_string();
                let tab_4 = "4. 🔍 Forensic Snapshot".to_string();
                let tab_5 = "5. ⏱️ Benchmarks".to_string();
                let tab_6 = "6. ⚡ Codegen".to_string();

                ui.selectable_value(&mut self.current_tab, StudioTab::SignalLab, tab_1);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Processors, tab_2);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Analyzer, tab_3);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Snapshot, tab_4);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Benchmark, tab_5);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Codegen, tab_6);
            });
        });

        egui::CentralPanel::default().show(ui, |ui| match self.current_tab {
            StudioTab::SignalLab => self.signal_lab_view.show(ui, &mut self.state),
            StudioTab::Processors => self.processors_view.show(ui, &mut self.state),
            StudioTab::Analyzer => self.analyzer_view.show(ui, &mut self.state),
            StudioTab::Snapshot => self.snapshot_view.show(ui, &mut self.state),
            StudioTab::Benchmark => self.benchmark_view.show(ui, &mut self.state),
            StudioTab::Codegen => self.codegen_view.show(ui, &self.state),
        });

        // Request 60 FPS continuous redraw for interactive scopes
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

pub fn run_studio() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 740.0])
            .with_min_inner_size([900.0, 560.0])
            .with_title("embedded-dsp Studio - DSP Workbench & Testbench"),
        ..Default::default()
    };

    eframe::run_native(
        "embedded-dsp Studio",
        native_options,
        Box::new(|cc| {
            configure_theme(&cc.egui_ctx);
            Ok(Box::new(EmbeddedDspStudioApp::default()))
        }),
    )
}
