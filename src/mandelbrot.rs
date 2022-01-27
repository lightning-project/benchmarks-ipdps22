use crate::util::benchmark;
use anyhow::Result;
use lightning::api::distribution::transform::PermutationDist;
use lightning::api::distribution::{RandomDist, TileDist};
use lightning::api::*;
use lightning::types::util::div_ceil;
use lightning::types::{DataType, DTYPE_FLOAT, DTYPE_INT, DTYPE_U8};

const CRUNCH: i32 = 8192;

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    let kernel = CudaKernelBuilder::from_file("resources/mandelbrot.cu", "Mandelbrot0<float>")?
        .param_array("dst", DataType::of::<[u8; 4]>())
        .param_value("imageW", DTYPE_INT)
        .param_value("imageH", DTYPE_INT)
        .param_value("crunch", DTYPE_INT)
        .param_value("xOff", DTYPE_FLOAT)
        .param_value("yOff", DTYPE_FLOAT)
        .param_value("xJP", DTYPE_FLOAT)
        .param_value("yJP", DTYPE_FLOAT)
        .param_value("scale", DTYPE_FLOAT)
        .param_value("colors", DataType::of::<[u8; 4]>())
        .param_value("frame", DTYPE_INT)
        .param_value("animationFrame", DTYPE_INT)
        .param_value("isJ", DTYPE_U8)
        .annotate("global [x,y] => write dst[y,x]")?
        .compile(&ctx)?;

    Ok(kernel)
}

fn run_kernel(ctx: &Context, kernel: &CudaKernel, n: u64, max_points_per_chunk: u64) -> Result<()> {
    let cta_size = 256;
    let ngpus = ctx.system().devices().len() as u64;
    let mut num_chunks = ngpus;
    let mut rows_per_chunk;

    loop {
        rows_per_chunk = div_ceil(n, num_chunks);

        if n * rows_per_chunk <= max_points_per_chunk || rows_per_chunk == 1 {
            break;
        }

        num_chunks += ngpus;
    }

    let dist = TileDist::new((rows_per_chunk, n));
    let result: Array<[u8; 4]> = ctx.zeros((n, n), dist)?;
    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        kernel.launch(
            (n, n),
            (cta_size, 1),
            RandomDist::new(PermutationDist::swap_xy(dist)),
            (
                &result,
                result.ncols(),
                result.nrows(),
                CRUNCH,
                -1.0_f32,             // xOff
                -2.5_f32,             // yOff
                0.0_f32,              // xJP
                0.0_f32,              // yJP
                2.0_f32 / (n as f32), // scale
                [255u8, 0, 255, 0],
                0,
                0,
                false as u8, // isJulia
            ),
        )?;

        Ok(())
    })?;

    println!(
        "n={} throughput={}, size={} tile_size={} ntiles={} {}",
        n * n,
        (n * n) as f64 / time.average().as_secs_f64() / 1e9,
        n,
        rows_per_chunk,
        result.regions()?.count(),
        time
    );

    Ok(())
}

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;

    let max_points_per_tile = 50_000_000;
    let mut n = 256;
    let max_n = max_n.unwrap_or(262_144);

    while n <= max_n {
        let dim = (n as f64).sqrt() as u64; // dim x dim grid
        run_kernel(&ctx, &kernel, dim, max_points_per_tile)?;

        n *= 2;
    }

    Ok(())
}

pub(crate) fn run_tilesize_tuning(ctx: Context) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;

    let dim = 1 << 16;
    let mut points_per_tile = dim * dim;

    while points_per_tile >= dim {
        run_kernel(&ctx, &kernel, dim, points_per_tile)?;

        points_per_tile /= 2;
    }

    Ok(())
}
