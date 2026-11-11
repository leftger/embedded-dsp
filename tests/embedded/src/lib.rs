#![no_std]
#![no_main]

use core::{
    hint::black_box,
    sync::atomic::{self, Ordering},
};
use cortex_m::peripheral::DWT;
use defmt::*;
#[cfg(feature = "rtt")]
use {defmt_rtt as _, panic_probe as _};

use embedded_dsp::pipeline::{Inplace, Process};

fn time<F: FnMut()>(mut func: F) -> u32 {
    cortex_m::interrupt::free(|_cs| {
        func();
        func();
        let c = DWT::cycle_count();
        atomic::compiler_fence(Ordering::SeqCst);
        func();
        atomic::compiler_fence(Ordering::SeqCst);
        DWT::cycle_count().wrapping_sub(c)
    })
}

#[inline(never)]
fn noinline<F: FnOnce()>(func: F) {
    func()
}

pub fn timeit<F: FnMut()>(mut func: F) -> u32 {
    time(|| {
        noinline(&mut func);
        noinline(&mut func);
    }) - time(|| {
        noinline(&mut func);
    })
}

#[derive(Default, Debug, Clone, defmt::Format)]
pub struct CyclesResults {
    pub single: u32,
    pub chunk: u32,
    pub slice: u32,
    pub inplace_chunk: u32,
    pub inplace_slice: u32,
}

pub const SLICE: usize = 1024;
pub const CHUNK: usize = 4;

impl CyclesResults {
    pub fn header() {
        info!(
            "Units: cycles per sample, chunk size: {=usize}, slice size: {=usize}",
            CHUNK, SLICE
        );
        info!("Name                  [single, chunk, slice, chunk inplace, slice inplace]");
    }

    pub fn show(&self, name: &str) {
        info!(
            "{=str}: single={=u32}, chunk={=u32}, slice={=u32}",
            name, self.single, self.chunk, self.slice
        );
    }
}

pub fn bench_process<P: Process<X, Y>, X: Copy + Default, Y: Copy + Default>(
    proc: &mut P,
) -> CyclesResults {
    let proc = black_box(proc);
    let x = black_box(X::default());
    let single = timeit(|| {
        black_box(proc.process(x));
    });
    let x = black_box([X::default(); SLICE]);
    let mut y = black_box([Y::default(); SLICE]);
    let slice = timeit(|| proc.block(&x, &mut y));
    let x = black_box([X::default(); CHUNK]);
    let mut y = black_box([Y::default(); CHUNK]);
    let chunk = timeit(|| proc.block(&x, &mut y));
    CyclesResults {
        single,
        chunk,
        slice,
        ..Default::default()
    }
}

pub fn bench_inplace<P: Inplace<X>, X: Copy + Default>(proc: &mut P) -> CyclesResults {
    let mut ret = bench_process::<_, X, X>(proc);
    let proc = black_box(proc);
    let mut y = black_box([X::default(); SLICE]);
    ret.inplace_slice = timeit(|| proc.inplace(&mut y));
    let mut y = black_box([X::default(); CHUNK]);
    ret.inplace_chunk = timeit(|| proc.inplace(&mut y));
    ret
}

// ─────────────────────────────────────────────────────────────────────────────
// Stack Painting and High-Water Mark (HWM) Memory Measurement
// ─────────────────────────────────────────────────────────────────────────────

/// Sentinel byte written across unused stack memory to detect peak usage.
pub const STACK_PAINT_BYTE: u8 = 0xAA;

/// Returns an address close to the caller's stack pointer.
#[inline(never)]
pub fn current_sp() -> usize {
    let local = 0u8;
    core::ptr::addr_of!(local) as usize
}

/// Paints stack memory between `bottom` and `top` with `STACK_PAINT_BYTE`.
#[inline(never)]
pub fn paint_stack(bottom: usize, top: usize) {
    let mut addr = bottom;
    while addr < top {
        unsafe {
            core::ptr::write_volatile(addr as *mut u8, STACK_PAINT_BYTE);
        }
        addr += 1;
    }
}

/// Computes the peak stack usage (in bytes) within painted interval `[bottom, top)`.
#[inline(never)]
pub fn measure_stack_hwm(bottom: usize, top: usize) -> usize {
    let mut addr = bottom;
    while addr < top {
        let byte = unsafe { core::ptr::read_volatile(addr as *const u8) };
        if byte != STACK_PAINT_BYTE {
            break;
        }
        addr += 1;
    }
    top.saturating_sub(addr)
}

/// Executes closure `f`, measuring the maximum stack high-water mark (HWM) used.
///
/// Leaves `guard` bytes below the caller's stack frame untouched, paints a `window`
/// of bytes below that, runs `f`, and calculates how deep the stack pointer penetrated.
pub fn run_with_stack_hwm<R, F: FnOnce() -> R>(guard: usize, window: usize, f: F) -> (R, usize) {
    let top = current_sp().saturating_sub(guard);
    let bottom = top.saturating_sub(window);
    paint_stack(bottom, top);

    let res = f();

    let used = measure_stack_hwm(bottom, top);
    (res, used)
}
