use crate::util::{benchmark, random_array, BenchmarkResult};
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::transform::PermutationDist;
use lightning::api::distribution::{
    BlockDist, CentralizeDist, ReplicateDist, RowBlockCyclic, TileDist,
};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{Dim3, MemoryId, DTYPE_DOUBLE, DTYPE_SIZE_T, DTYPE_U32};

struct Kernels {
    calculate_gavg: CudaKernel,
    calculate_bincount: CudaKernel,
    calculate_cocavg: CudaKernel,
    process_cocavg: CudaKernel,
    calculate_dist: CudaKernel,
    sum_y: CudaKernel,
    select_best_dist: CudaKernel,
}

fn compile_kernels(ctx: &Context) -> Result<Kernels> {
    fn kernel_tiled(
        fun: &str,
        block_size: impl Into<Dim3>,
        tile_factor: impl Into<Dim3>,
    ) -> Result<CudaKernelBuilder> {
        let block_size = block_size.into();
        let tile_factor = tile_factor.into();
        let tile_size = block_size * tile_factor;

        let mut kernel = CudaKernelBuilder::from_file("resources/cgc.cu", fun)?;
        kernel
            .debugging(false)
            .block_size(block_size)?
            .define("BLOCK_SIZE_X", block_size[0])
            .define("BLOCK_SIZE_Y", block_size[1])
            .define("BLOCK_SIZE_Z", block_size[2])
            .define("TILE_FACTOR_X", tile_factor[0])
            .define("TILE_FACTOR_Y", tile_factor[1])
            .define("TILE_FACTOR_Z", tile_factor[2])
            .add_constant("TILE_SIZE_X", tile_size[0] as i64)
            .add_constant("TILE_SIZE_Y", tile_size[1] as i64)
            .add_constant("TILE_SIZE_Z", tile_size[2] as i64);

        Ok(kernel)
    }

    fn kernel(fun: &str, block_size: impl Into<Dim3>) -> Result<CudaKernelBuilder> {
        kernel_tiled(fun, block_size.into(), Dim3::one())
    }

    let calculate_gavg = kernel("calculate_gavg", (16, 16))?
        .param_value("nrows", DTYPE_SIZE_T)
        .param_value("ncols", DTYPE_SIZE_T)
        .param_array("Z", DTYPE_DOUBLE)
        .param_array("sum", DTYPE_DOUBLE)
        .annotate("global [row, col] => read Z[row][col], reduce(+) sum")?
        .compile(ctx)?;

    let calculate_bincount = kernel("calculate_bincount", 256)?
        .param_value("nitems", DTYPE_SIZE_T)
        .param_value("nclusters", DTYPE_SIZE_T)
        .param_array("item_clusters", DTYPE_U32)
        .param_array("counts", DTYPE_U32)
        .annotate("global [gx] => read item_clusters[gx], reduce(+) counts[:]")?
        .compile(ctx)?;

    //let calculate_cocavg = kernel_tiled("calculate_cocavg", (8, 8), (4, 4))?
    let calculate_cocavg = kernel_tiled("calculate_cocavg", (2, 32), (2, 8))?
        //.option("--maxrregcount=107")
        .param_value("nrows", DTYPE_SIZE_T)
        .param_value("ncols", DTYPE_SIZE_T)
        .param_value("nclusters_row", DTYPE_SIZE_T)
        .param_value("nclusters_col", DTYPE_SIZE_T)
        .param_array("Z", DTYPE_DOUBLE)
        .param_array("row_clusters", DTYPE_U32)
        .param_array("col_clusters", DTYPE_U32)
        .param_array("cocavg", DTYPE_DOUBLE)
        .annotate("block [x,y] => read row_clusters[x * TILE_SIZE_X : (x + 1) * TILE_SIZE_X],\
                                  read col_clusters[y * TILE_SIZE_Y : (y + 1) * TILE_SIZE_Y],\
                                  read Z[x * TILE_SIZE_X : (x + 1) * TILE_SIZE_X][y * TILE_SIZE_Y : (y + 1) * TILE_SIZE_Y],\
                                  reduce(+) cocavg[:,:]")?
        .compile(ctx)?;

    let process_cocavg = kernel("process_cocavg", (16, 16))?
        .param_value("nclusters_row", DTYPE_SIZE_T)
        .param_value("nclusters_col", DTYPE_SIZE_T)
        .param_array("gavg", DTYPE_DOUBLE)
        .param_array("nel_row_clusters", DTYPE_U32)
        .param_array("nel_col_clusters", DTYPE_U32)
        .param_array("cocavg", DTYPE_DOUBLE)
        .param_array("log_cocavg", DTYPE_DOUBLE)
        .param_value("epsilon", DTYPE_DOUBLE)
        .annotate(
            "global [col, row] => read gavg, \
                                  readwrite cocavg[row][col], \
                                  write log_cocavg[row][col], \
                                  read nel_row_clusters[row], \
                                  read nel_col_clusters[col]",
        )?
        .compile(ctx)?;

    let calculate_dist = kernel_tiled("calculate_dist", (16, 4), (8, 8))?
        .param_value("nrows", DTYPE_SIZE_T)
        .param_value("ncols", DTYPE_SIZE_T)
        .param_value("nclusters", DTYPE_SIZE_T)
        .param_array("Z", DTYPE_DOUBLE)
        .param_array("log_cocavg", DTYPE_DOUBLE)
        .param_array("col_clusters", DTYPE_U32)
        .param_array("dists", DTYPE_DOUBLE)
        .annotate("block [x,y] => read Z[y * TILE_SIZE_Y : (y + 1) * TILE_SIZE_Y][x * TILE_SIZE_X : (x + 1) * TILE_SIZE_X],\
                                  read log_cocavg[:,:],\
                                  read col_clusters[x * TILE_SIZE_X : (x + 1) * TILE_SIZE_X],\
                                  reduce(+) dists[y * TILE_SIZE_Y : (y + 1) * TILE_SIZE_Y][:]")?
        .compile(ctx)?;

    let sum_y = kernel("sum_y", 256)?
        .param_value("nclusters_row", DTYPE_SIZE_T)
        .param_value("nclusters_col", DTYPE_SIZE_T)
        .param_array("cocavg", DTYPE_DOUBLE)
        .param_array("nel_clusters", DTYPE_U32)
        .param_array("Y", DTYPE_DOUBLE)
        .annotate("global i => read nel_clusters[:], read cocavg[i,:], write Y[i]")?
        .compile(ctx)?;

    let select_best_dist = kernel("select_best_dist", 256)?
        .param_value("nitems", DTYPE_SIZE_T)
        .param_value("nclusters", DTYPE_SIZE_T)
        .param_array("all_dists", DTYPE_DOUBLE)
        .param_array("Y", DTYPE_DOUBLE)
        .param_array("item_clusters", DTYPE_U32)
        .param_array("min_dist", DTYPE_DOUBLE)
        .annotate(
            "global gid => read Y[:], \
                           read all_dists[gid,:], \
                           write item_clusters[gid], \
                           write min_dist[gid]",
        )?
        .compile(ctx)?;

    Ok(Kernels {
        calculate_gavg,
        calculate_bincount,
        calculate_cocavg,
        process_cocavg,
        calculate_dist,
        sum_y,
        select_best_dist,
    })
}

fn block_size(kernel: &CudaKernel) -> Dim3 {
    Dim3::new(
        kernel.constant("BLOCK_SIZE_X") as u64,
        kernel.constant("BLOCK_SIZE_Y") as u64,
        kernel.constant("BLOCK_SIZE_Z") as u64,
    )
}

fn extract_tile_factor(kernel: &CudaKernel) -> Dim3 {
    Dim3::new(
        kernel.constant("TILE_FACTOR_X") as u64,
        kernel.constant("TILE_FACTOR_Y") as u64,
        kernel.constant("TILE_FACTOR_Z") as u64,
    )
}

struct State {
    dist_1d: RowBlockCyclic<Vec<MemoryId>>,
    dist_2d: TileDist<Vec<MemoryId>>,
    nrows: u64,
    ncols: u64,
    nclusters_row: u64,
    nclusters_col: u64,
    Z: Array<f64>,
    initial_row_clusters: Array<u32>,
    initial_col_clusters: Array<u32>,
    row_clusters: Array<u32>,
    col_clusters: Array<u32>,
    all_dists: Array<f64>,
    gavg: Array<f64>,
    nel_col_clusters: Array<u32>,
    nel_row_clusters: Array<u32>,
    cocavg: Array<f64>,
    log_cocavg: Array<f64>,
    Y_row: Array<f64>,
    Y_col: Array<f64>,
}

fn init_state(
    ctx: &Context,
    n: u64,
    nclusters: u64,
    nchunks: u64,
    centralize_vecs: bool,
) -> Result<State> {
    const ALIGN: u64 = 32 * 8;

    let (nrows, ncols) = (n, n);
    let (nclusters_row, nclusters_col) = (nclusters, nclusters);

    let memories: Vec<_> = ctx.system().devices().iter().map(|e| e.memory_id).collect();

    let chunk_size = round_up(div_ceil(n, nchunks), ALIGN);
    let dist_2d = TileDist::with_memories([chunk_size, chunk_size], memories.clone());
    let Z: Array<f64> = ctx.empty((nrows, ncols), &dist_2d)?;
    random_array(&Z, 0.0, 100.0)?;

    let chunk_size = match centralize_vecs {
        true => div_ceil(n, memories.len() as u64),
        false => n,
    };
    let dist_1d = RowBlockCyclic::with_memories(round_up(chunk_size, ALIGN), memories);
    let all_dists: Array<f64> = ctx.empty((n, nclusters), &dist_1d)?;

    // This can all be centralized
    let dist = CentralizeDist::root();
    let gavg: Array<f64> = ctx.empty(1, dist)?;
    let cocavg: Array<f64> = ctx.empty((nclusters_row, nclusters_col), dist)?;
    let Y_row: Array<f64> = ctx.empty(nclusters_row, dist)?;
    let Y_col: Array<f64> = ctx.empty(nclusters_col, dist)?;
    let log_cocavg: Array<f64> = ctx.empty((nclusters_row, nclusters_col), dist)?;
    let nel_col_clusters: Array<u32> = ctx.empty(nclusters_col, dist)?;
    let nel_row_clusters: Array<u32> = ctx.empty(nclusters_row, dist)?;

    let initial_row_clusters: Array<u32> = ctx.empty(nrows, dist)?;
    let initial_col_clusters: Array<u32> = ctx.empty(ncols, dist)?;
    random_array(&initial_row_clusters, 0, nclusters_row as u32)?;
    random_array(&initial_col_clusters, 0, nclusters_col as u32)?;

    // These can all be replicated
    let dist = ReplicateDist::new();
    let row_clusters: Array<u32> = ctx.empty(nrows, dist)?;
    let col_clusters: Array<u32> = ctx.empty(ncols, dist)?;

    let state = State {
        dist_1d,
        dist_2d,
        nrows,
        ncols,
        nclusters_row,
        nclusters_col,
        Z,
        initial_row_clusters,
        initial_col_clusters,
        row_clusters,
        col_clusters,
        all_dists,
        gavg,
        nel_col_clusters,
        nel_row_clusters,
        cocavg,
        log_cocavg,
        Y_row,
        Y_col,
    };

    reset_state(ctx, &state)?;
    Ok(state)
}

fn reset_state(_ctx: &Context, state: &State) -> Result<()> {
    state
        .row_clusters
        .assign_from(&state.initial_row_clusters)?;
    state
        .col_clusters
        .assign_from(&state.initial_col_clusters)?;
    Ok(())
}

fn run_iteration(ctx: &Context, kernels: &Kernels, state: &State) -> Result<Event> {
    let epsilon = 0.0;
    let nrows = state.nrows;
    let ncols = state.ncols;
    let nclusters_row = state.nclusters_row;
    let nclusters_col = state.nclusters_col;

    state.gavg.fill_zeros()?;
    state.nel_col_clusters.fill_zeros()?;
    state.nel_row_clusters.fill_zeros()?;
    state.cocavg.fill_zeros()?;

    let dist = &state.dist_2d;
    kernels.calculate_gavg.launch(
        (nrows, ncols),
        block_size(&kernels.calculate_gavg),
        dist,
        (nrows, ncols, &state.Z, &state.gavg),
    )?;

    kernels.calculate_bincount.launch(
        nrows,
        block_size(&kernels.calculate_bincount),
        &state.dist_1d,
        (
            nrows,
            nclusters_row,
            &state.row_clusters,
            &state.nel_row_clusters,
        ),
    )?;

    kernels.calculate_bincount.launch(
        ncols,
        block_size(&kernels.calculate_bincount),
        &state.dist_1d,
        (
            ncols,
            nclusters_col,
            &state.col_clusters,
            &state.nel_col_clusters,
        ),
    )?;

    let tile_factor = extract_tile_factor(&kernels.calculate_cocavg);
    let dist = state.dist_2d.stride_by(tile_factor)?;
    kernels.calculate_cocavg.launch(
        (
            div_ceil(nrows, tile_factor[0]),
            div_ceil(ncols, tile_factor[1]),
        ),
        block_size(&kernels.calculate_cocavg),
        dist,
        (
            nrows,
            ncols,
            nclusters_row,
            nclusters_col,
            &state.Z,
            &state.row_clusters,
            &state.col_clusters,
            &state.cocavg,
        ),
    )?;

    kernels.sum_y.launch(
        nclusters_row,
        block_size(&kernels.sum_y),
        CentralizeDist::root(),
        (
            nclusters_row,
            nclusters_col,
            &state.cocavg,
            &state.nel_col_clusters,
            &state.Y_row,
        ),
    )?;

    kernels.sum_y.launch(
        nclusters_row,
        block_size(&kernels.sum_y),
        CentralizeDist::root(),
        (
            nclusters_col,
            nclusters_row,
            &state.cocavg.swap_axes(0, 1),
            &state.nel_row_clusters,
            &state.Y_col,
        ),
    )?;

    kernels.process_cocavg.launch(
        (nclusters_col, nclusters_row),
        block_size(&kernels.process_cocavg),
        CentralizeDist::root(),
        (
            nclusters_row,
            nclusters_col,
            &state.gavg,
            &state.nel_row_clusters,
            &state.nel_col_clusters,
            &state.cocavg,
            &state.log_cocavg,
            epsilon,
        ),
    )?;

    let temp_dists: Array<f64> =
        ctx.zeros((nrows, nclusters_row), BlockDist::with_alignment(1024))?;

    let tile_factor = extract_tile_factor(&kernels.calculate_dist);
    let dist = state.dist_2d.stride_by(tile_factor)?;

    kernels.calculate_dist.launch(
        (
            div_ceil(ncols, tile_factor[0]),
            div_ceil(nrows, tile_factor[1]),
        ),
        block_size(&kernels.calculate_dist),
        PermutationDist::swap_xy(dist),
        (
            nrows,
            ncols,
            nclusters_row,
            &state.Z,
            &state.log_cocavg,
            &state.col_clusters,
            &temp_dists,
        ),
    )?;

    let dist = &state.dist_1d;
    let min_dist: Array<f64> = ctx.empty(nrows, dist)?;
    kernels.select_best_dist.launch(
        nrows,
        block_size(&kernels.select_best_dist),
        dist,
        (
            nrows,
            nclusters_row,
            &temp_dists,
            &state.Y_row,
            &state.row_clusters,
            &min_dist,
        ),
    )?;
    drop(min_dist);
    drop(temp_dists);

    let temp_dists: Array<f64> = ctx.zeros((ncols, nclusters_col), &state.dist_1d)?;

    let tile_factor = extract_tile_factor(&kernels.calculate_dist);
    let dist = state.dist_2d.stride_by(tile_factor)?;

    kernels.calculate_dist.launch(
        (
            div_ceil(nrows, tile_factor[0]),
            div_ceil(ncols, tile_factor[1]),
        ),
        block_size(&kernels.calculate_dist),
        dist,
        (
            ncols,
            nrows,
            nclusters_col,
            &state.Z.swap_axes(0, 1),
            &state.cocavg.swap_axes(0, 1),
            &state.row_clusters,
            &temp_dists,
        ),
    )?;

    let min_dist: Array<f64> = ctx.empty(ncols, &state.dist_1d)?;
    let event = kernels.select_best_dist.launch(
        ncols,
        block_size(&kernels.select_best_dist),
        &state.dist_1d,
        (
            ncols,
            nclusters_col,
            &temp_dists,
            &state.Y_col,
            &state.col_clusters,
            &min_dist,
        ),
    )?;
    drop(min_dist);
    drop(temp_dists);

    Ok(event)
}

fn run_cgc(ctx: &Context, kernels: &Kernels, state: &State) -> Result<BenchmarkResult> {
    let niter = 1;

    benchmark(ctx, || {
        reset_state(&ctx, &state)?;
        //let mut prev_event = None;

        for _ in 0..niter {
            let _event = run_iteration(ctx, &kernels, &state)?;

            //if let Some(event) = prev_event.replace(event) {
            //    event.wait()?;
            //}
        }

        Ok(())
    })
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernels = compile_kernels(&ctx)?;
    let nclusters = args.k.unwrap_or(25);
    let mut n = 5000;

    loop {
        let mut successfull = 0;

        for &nchunks in &[8, 4, 2, 1] {
            for &flag in &[true, false] {
                if n / nchunks <= 30000 {
                    let state = init_state(&ctx, n, nclusters, nchunks, flag)?;
                    let result = run_cgc(&ctx, &kernels, &state)?;
                    successfull += 1;

                    println!(
                        "n={} k={} nchunks={} centralize_vec={} {}",
                        n,
                        nclusters,
                        nchunks * nchunks,
                        flag,
                        result
                    );
                }
            }
        }

        if successfull == 0 {
            break;
        }

        n += 5000;
    }
    Ok(())
}
