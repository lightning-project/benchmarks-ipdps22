use crate::util::{benchmark, BenchmarkResult};
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::transform::PermutationDist;
use lightning::api::distribution::{Stencil2DDist, TileDist};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, DTYPE_FLOAT, DTYPE_INT};
use std::mem::swap;

const MAX_PD: f32 = 3.0e6;
const PRECISION: f32 = 0.001;
const SPEC_HEAT_SI: f32 = 1.75e6;
const K_SI: f32 = 100.0;
const FACTOR_CHIP: f32 = 0.5;
const CHIP_HEIGHT: f32 = 0.016;
const CHIP_WITDH: f32 = 0.016;
const T_CHIP: f32 = 0.0005;
const CTA_SIZE: u64 = 16;

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    CudaKernelBuilder::from_file("resources/hotspot.cu", "calculate_temp")?
        .define("BLOCK_SIZE", CTA_SIZE)
        .param_value("grid_cols", DTYPE_INT)
        .param_value("grid_rows", DTYPE_INT)
        .param_value("border_cols", DTYPE_INT)
        .param_value("border_rows", DTYPE_INT)
        .param_value("Cap", DTYPE_FLOAT)
        .param_value("Rx", DTYPE_FLOAT)
        .param_value("Ry", DTYPE_FLOAT)
        .param_value("Rz", DTYPE_FLOAT)
        .param_value("step", DTYPE_FLOAT)
        .param_value("time_elapsed", DTYPE_FLOAT)
        .param_array("power", DTYPE_FLOAT)
        .param_array("temp_src", DTYPE_FLOAT)
        .param_array("temp_dst", DTYPE_FLOAT)
        .annotate(
            "block [bx, by] => \
                        read temp_src[(BLOCK_SIZE - 2) * by - border_rows : (BLOCK_SIZE - 2) * by - border_rows + BLOCK_SIZE,\
                                      (BLOCK_SIZE - 2) * bx - border_cols : (BLOCK_SIZE - 2) * bx - border_cols + BLOCK_SIZE],\
                        read power[(BLOCK_SIZE - 2) * by - border_rows : (BLOCK_SIZE - 2) * by - border_rows + BLOCK_SIZE,\
                                   (BLOCK_SIZE - 2) * bx - border_cols : (BLOCK_SIZE - 2) * bx - border_cols + BLOCK_SIZE],\
                        write temp_dst[(BLOCK_SIZE - 2) * by - border_rows + 1 : (BLOCK_SIZE - 2) * by - border_rows + BLOCK_SIZE - 1,\
                                       (BLOCK_SIZE - 2) * bx - border_cols + 1 : (BLOCK_SIZE - 2) * bx - border_cols + BLOCK_SIZE - 1] ",
        )?
        .compile(&ctx)
        .map_err(|e| e.into())
}

fn compute_tile_size(ctx: &Context, rows: u64, cols: u64) -> (u64, u64) {
    let ngpus = ctx.system().devices().len() as u64;
    let nnodes = ctx.system().workers().len() as u64;
    let ngpus_per_node = ngpus / nnodes;
    let max_tile_size = 50_000_000;

    let mut row_factor = nnodes * ngpus_per_node;
    let col_factor = 1;

    loop {
        let tile_rows = div_ceil(rows, row_factor);
        let tile_cols = div_ceil(cols, col_factor);
        let tile_size = tile_rows * tile_cols;

        if tile_size < max_tile_size {
            return (tile_rows, tile_cols);
        }

        row_factor += nnodes * ngpus_per_node;
    }
}

fn execute(
    ctx: &Context,
    rows: u64,
    cols: u64,
    tile_rows: u64,
    tile_cols: u64,
    calculate_temp: &CudaKernel,
) -> Result<BenchmarkResult> {
    if cols > i32::MAX as u64 || rows > i32::MAX as u64 {
        anyhow::bail!("grid dimensions cannot exceed i32::MAX");
    }

    let cta_size = CTA_SIZE;
    let grid_height: f32 = CHIP_HEIGHT / rows as f32;
    let grid_width: f32 = CHIP_WITDH / cols as f32;
    let cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height;
    let rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height);
    let ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width);
    let rz = T_CHIP / (K_SI * grid_height * grid_width);
    let max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
    let step = PRECISION / max_slope;
    let time_elapsed = 0.001_f32;

    let tile_rows = round_up(tile_rows, CTA_SIZE - 2); // Round up to multiple of CTA_SIZE-2
    let tile_cols = round_up(tile_cols, CTA_SIZE - 2);

    let dist = Stencil2DDist::new([tile_rows, tile_cols], 1);
    let power: Array<float> = ctx.zeros((rows, cols), dist)?;
    let mut temp_src: Array<float> = ctx.zeros((rows, cols), dist)?;
    let mut temp_dst: Array<float> = ctx.zeros((rows, cols), dist)?;

    let time = benchmark(&ctx, || {
        for _ in 0..10 {
            calculate_temp.launch(
                (
                    div_ceil(cols, cta_size - 2) * cta_size,
                    div_ceil(rows, cta_size - 2) * cta_size,
                ),
                (cta_size, cta_size),
                PermutationDist::swap_xy(TileDist::new([
                    div_ceil(tile_rows, cta_size - 2) * cta_size,
                    div_ceil(tile_cols, cta_size - 2) * cta_size,
                ])),
                (
                    cols,
                    rows,
                    1,
                    1,
                    cap,
                    rx,
                    ry,
                    rz,
                    step,
                    time_elapsed,
                    &power,
                    &temp_src,
                    &temp_dst,
                ),
            )?;

            swap(&mut temp_src, &mut temp_dst);
        }

        Ok(())
    })?;

    Ok(time)
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let max_n = args.end.unwrap_or(131_072);
    let mut n = 64 * 64;
    let mut results = std::collections::HashMap::new();

    while n <= max_n {
        if n >= args.begin {
            let dim = (n as f64).sqrt() as u64;

            let (tile_rows, tile_cols) = compute_tile_size(&ctx, dim, dim);
            let key = (n, tile_rows, tile_cols);

            if !results.contains_key(&key) {
                let time = execute(&ctx, dim, dim, tile_rows, tile_cols, &kernel)?;
                results.insert(key, time);
            }

            println!(
                "n={} rows={} cols={} {} tile_cols={} tile_rows={} tiles={}",
                dim * dim,
                dim,
                dim,
                &results[&key],
                tile_cols,
                tile_rows,
                div_ceil(dim, tile_cols) * div_ceil(dim, tile_rows),
            );
        }

        n *= 2;
    }

    Ok(())
}
