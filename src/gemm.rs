use crate::util::{benchmark, random_array};
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::{RowBlockCyclic, TileDist};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, HasDataType, DTYPE_INT};

const BLOCK_SIZE_X: u64 = 32;
const BLOCK_SIZE_Y: u64 = 8;
const TILE_SIZE_X: u64 = 4;
const TILE_SIZE_Y: u64 = 4;

type FloatType = float;

pub(crate) fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    let gemm = CudaKernelBuilder::from_file("resources/matmul.cu", "matrix_multiply")?
        .debugging(false)
        .define("block_size_x", BLOCK_SIZE_X)
        .define("block_size_y", BLOCK_SIZE_Y)
        .define("tile_size_x", TILE_SIZE_X)
        .define("tile_size_y", TILE_SIZE_Y)
        .define("float_type", FloatType::data_type().ctype())
        .param_value("n", DTYPE_INT)
        .param_value("m", DTYPE_INT)
        .param_value("p", DTYPE_INT)
        .param_array("C", FloatType::data_type())
        .param_array("A", FloatType::data_type())
        .param_array("B", FloatType::data_type())
        .annotate(
            "block [bx, by] => read A[by *block_size_y*tile_size_y : (by+1)*block_size_y*tile_size_y , :], \
                               read B[:,bx*block_size_x*tile_size_x : (bx+1)*block_size_x*tile_size_x], \
                               readwrite C[by*block_size_y*tile_size_y : (by+1)*block_size_y*tile_size_y, \
                                           bx*block_size_x*tile_size_x : (bx+1)*block_size_x*tile_size_x]",
        )?
        .compile(&ctx)?;

    Ok(gemm)
}

fn run_gemm(
    ctx: &Context,
    gemm: &CudaKernel,
    size: u64,
    points_per_chunk: u64,
    min_chunks_per_gpu: u64,
    tiled: bool,
) -> Result<()> {
    assert!(min_chunks_per_gpu > 0);

    let num_gpus = ctx.system().devices().len() as u64;
    let mut num_chunks = num_gpus * min_chunks_per_gpu;
    let mut chunk_size;

    loop {
        chunk_size = round_up(div_ceil(size, num_chunks), BLOCK_SIZE_Y * TILE_SIZE_Y);

        if tiled && (chunk_size * chunk_size <= points_per_chunk || num_chunks >= size * size) {
            break;
        } else if !tiled && (chunk_size * size <= points_per_chunk || num_chunks >= size) {
            break;
        } else {
            num_chunks += num_gpus;
        }
    }

    let dist = RowBlockCyclic::new(chunk_size);
    let am: Array<float> = ctx.empty((size, size), dist)?;
    let bm: Array<float> = ctx.empty((size, size), dist)?;
    let cm: Array<float> = ctx.empty((size, size), dist)?;

    random_array(&am, 0.0, 1.0)?;
    random_array(&bm, 0.0, 1.0)?;
    ctx.synchronize()?;

    let result = benchmark(&ctx, || {
        let mut prev_event = None;

        for start in (0..size).step_by(chunk_size as usize) {
            let end = (start + chunk_size).clamp(0, size);
            let len = end - start;

            let event = gemm.launch(
                (div_ceil(len, TILE_SIZE_X), div_ceil(size, TILE_SIZE_Y)),
                (BLOCK_SIZE_X, BLOCK_SIZE_Y),
                TileDist::new((
                    div_ceil(chunk_size, TILE_SIZE_X),
                    div_ceil(chunk_size, TILE_SIZE_Y),
                )),
                (
                    size,                        // N
                    size,                        // M
                    len,                         // P
                    &cm.slice((.., start..end)), // N x P
                    &am,                         // N x M
                    &bm.slice((.., start..end)), // M x P
                ),
            )?;

            if let Some(p) = prev_event.replace(event) {
                p.wait()?;
            }
        }

        Ok(())
    })?;

    println!(
        "n={} size={} chunk_size={} min_chunks_per_gpu={} {}",
        size * size * size,
        size,
        chunk_size,
        min_chunks_per_gpu,
        result
    );

    Ok(())
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let mut n = 16 * 16 * 16;
    let max_n = args.end.unwrap_or(1024 * 1024 * 1024);

    while n <= max_n {
        if n >= args.begin {
            let size = (n as f64).powf(1.0 / 3.0).ceil() as u64;
            run_gemm(&ctx, &kernel, size, 150_000_000, 1, false)?;
            run_gemm(&ctx, &kernel, size, 150_000_000, 2, false)?;
        }

        n *= 2;
    }

    Ok(())
}
