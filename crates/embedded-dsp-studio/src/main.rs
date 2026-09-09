#![allow(
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::manual_div_ceil,
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::collapsible_if
)]

mod app;
pub mod export;
mod state;
mod theme;
mod views;
mod widgets;

#[cfg(test)]
mod tests;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    app::run_studio()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Redirect log to browser console
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();
    wasm_bindgen_futures::spawn_local(async {
        app::run_studio_web("the_canvas_id")
            .await
            .expect("Failed to start eframe on canvas");
    });
}
