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
mod state;
mod theme;
mod views;
mod widgets;

fn main() -> eframe::Result<()> {
    app::run_studio()
}
