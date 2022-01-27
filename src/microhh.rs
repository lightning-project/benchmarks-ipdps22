use crate::util::{benchmark, random_array};
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::stencil3d::Stencil3DDist;
use lightning::api::distribution::{ReplicateDist, TileDist};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{Dim3, DTYPE_DOUBLE, DTYPE_INT};

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    let diff = CudaKernelBuilder::from_file("resources/diff_4.cu", "diff_c_g")?
        .param_array("at", DTYPE_DOUBLE)
        .param_array("a", DTYPE_DOUBLE)
        .param_array("dzi4", DTYPE_DOUBLE)
        .param_array("dzhi4", DTYPE_DOUBLE)
        .param_value("dxi", DTYPE_DOUBLE)
        .param_value("dyi", DTYPE_DOUBLE)
        .param_value("visc", DTYPE_DOUBLE)
        .param_value("istart", DTYPE_INT)
        .param_value("jstart", DTYPE_INT)
        .param_value("kstart", DTYPE_INT)
        .param_value("iend", DTYPE_INT)
        .param_value("jend", DTYPE_INT)
        .param_value("kend", DTYPE_INT)
        .param_value("icells", DTYPE_INT)
        .param_value("jcells", DTYPE_INT)
        .param_value("ngc", DTYPE_INT)
        .annotate(
            "global [i, j, k] =>
            write at[i + istart, j + jstart, k + kstart],
            read a[i + istart - 3:i + istart + 4, j + jstart, k + kstart],
            read a[i + istart, j + jstart - 3:j + jstart + 4, k + kstart],
            read a[i + istart, j + jstart, k +kstart-3 : k+kstart+4],
            read dzi4[k + kstart],
            read dzhi4[k+kstart-1 : k+kstart+3]
        ",
        )?
        .compile(ctx)?;

    Ok(diff)
}

fn execute(ctx: &Context, cells: u64, kernel: &CudaKernel) -> Result<()> {
    let cta_size = [32, 32, 1];
    let niter = 5;

    let num_gpus = ctx.system().devices().len() as u64;
    let tile_x = round_up(div_ceil(cells, num_gpus), cta_size[0]);
    let tile_z = cells;

    let num_tiles_y = div_ceil(tile_x * tile_z * cells, 1_000_000);
    let tile_y = round_up(div_ceil(cells, num_tiles_y), cta_size[1]);
    let tile_size = Dim3::new(tile_x, tile_y, tile_z);

    let visc = 1e-5;
    let ijgc = 3;
    let igc = ijgc;
    let jgc = ijgc;
    let kgc = 3;
    let icells = cells;
    let jcells = cells;
    let _kcells = cells;
    let itot = cells - 2 * igc;
    let jtot = cells - 2 * jgc;
    let ktot = cells - 2 * kgc;
    let xsize = 3200;
    let ysize = 3200;
    let _zsize = 3200;
    let dx = xsize as f64 / itot as f64;
    let dy = ysize as f64 / jtot as f64;
    let dxi = 1.0 / dx;
    let dyi = 1.0 / dy;
    let istart = igc;
    let jstart = jgc;
    let kstart = kgc;
    let iend = itot + igc;
    let jend = jtot + jgc;
    let kend = ktot + kgc;
    let ngc = 0; // unused

    let dist = Stencil3DDist::new(tile_size, 0).halo([0, 0, 0], [2 * igc, 2 * jgc, 2 * kgc]);

    let inputs = ctx.empty((cells, cells, cells), &dist)?;
    let outputs = ctx.empty((cells, cells, cells), &dist)?;
    let dzi4 = ctx.empty(cells, ReplicateDist::new())?;
    let dzhi4 = ctx.empty(cells, ReplicateDist::new())?;

    random_array(&inputs, 0.0, 1.0)?;
    random_array(&outputs, 0.0, 1.0)?;
    random_array(&dzi4, 0.0, 1.0)?;
    random_array(&dzhi4, 0.0, 1.0)?;
    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        for _ in 0..niter {
            kernel.launch(
                (cells, cells, cells),
                cta_size,
                TileDist::new(tile_size),
                (
                    &inputs, &outputs, &dzi4, &dzhi4, dxi, dyi, visc, istart, jstart, kstart, iend,
                    jend, kend, icells, jcells, ngc,
                ),
            )?;
        }

        Ok(())
    })?;

    println!(
        "n={} size={} tile_x={} tile_y={} tile_z={} {}",
        cells * cells * cells,
        cells,
        tile_x,
        tile_y,
        tile_z,
        time
    );

    Ok(())
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let max_n = args.end.unwrap_or(1024 * 1024 * 1024);
    let mut n = 1;

    while n <= max_n {
        if n >= args.begin {
            let size = (n as f64).powf(1.0 / 3.0).ceil() as u64;
            execute(&ctx, size, &kernel)?;
        }

        n *= 2;
    }

    Ok(())
}
