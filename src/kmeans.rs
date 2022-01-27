use crate::util::benchmark;
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::{BlockCyclic, CentralizeDist, ReplicateDist};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, int, DTYPE_FLOAT, DTYPE_INT, DTYPE_U64};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand_distr::Normal;
use std::cmp::min;

const CTA_SIZE: usize = 1024;

struct Kernels {
    update_membership: CudaKernel,
    compute_centers: CudaKernel,
}

fn compile_kernels(ctx: &Context) -> Result<Kernels> {
    fn create(name: &str) -> Result<CudaKernelBuilder> {
        let file = "resources/kmeans.cu";
        let options = [
            "-std=c++14",
            // "-Xptxas=\"-v\"",
            //"-maxrregcount", "40"
        ];

        let mut kernel = CudaKernelBuilder::from_file(file, name)?;
        kernel.options(&options);
        kernel.define("block_size", CTA_SIZE);
        kernel.debugging(false);
        Ok(kernel)
    }

    let update_membership = create("update_membership")?
        .param_value("npoints", DTYPE_U64)
        .param_value("nclusters", DTYPE_U64)
        .param_value("nfeatures", DTYPE_U64)
        .param_array("points", DTYPE_FLOAT)
        .param_array("centers", DTYPE_FLOAT)
        .param_array("membership", DTYPE_INT)
        .param_array("new_centers_sums", DTYPE_FLOAT)
        .param_array("new_sizes", DTYPE_U64)
        .param_array("num_deltas", DTYPE_U64)
        .annotate(
            "global i => \
               read points[i,:], \
               read centers[:,:], \
               readwrite membership[i], \
               reduce(+) new_centers_sums[:,:],\
               reduce(+) new_sizes[:],\
               reduce(+) num_deltas",
        )?
        .compile(&ctx)?;

    let compute_centers = create("compute_centers")?
        .param_array("center_sums", DTYPE_FLOAT)
        .param_array("sizes", DTYPE_U64)
        .param_array("centers", DTYPE_FLOAT)
        .annotate(
            "block [c, f] => read center_sums[c, f],
                             read sizes[c],
                             write centers[c, f]",
        )?
        .compile(&ctx)?;

    Ok(Kernels {
        update_membership,
        compute_centers,
    })
}

fn init_points(points: &Array<float>, ncenters: u64) -> Result<Vec<Vec<f32>>> {
    let rng = &mut SmallRng::seed_from_u64(0);
    let (npoints, nfeatures) = points.shape();
    let mut centers = vec![];

    for _ in 0..ncenters {
        centers.push(
            rng.sample_iter(Uniform::new(0.0, 100.0))
                .take(nfeatures as usize)
                .collect::<Vec<_>>(),
        );
    }

    let mut offset = min(4096, npoints);
    let labels = (0..offset)
        .map(|_| rng.gen_range(0..ncenters) as usize)
        .collect::<Vec<_>>();

    for feature in 0..nfeatures {
        let data = labels
            .iter()
            .map(|&i| {
                let mean = centers[i][feature as usize];
                let dist = Normal::new(mean, 0.01).unwrap();
                rng.sample(dist)
            })
            .collect::<Vec<_>>();

        points.column(feature).slice(0..offset).copy_from(&data)?;
    }

    while offset < npoints {
        let len = (npoints - offset).min(offset);
        let src = points.slice((0..len, ..));
        let dst = points.slice((offset..(offset + len), ..));
        src.assign_to(&dst)?.wait()?;
        offset += len;
    }

    Ok(centers)
}

fn run_kmeans(ctx: &Context, kernels: &Kernels, n: u64, k: u64, chunk_size: u64) -> Result<()> {
    let npoints = n;
    let nclusters = k;
    let nfeatures = 4;
    let niter = 5;
    let cta_size = CTA_SIZE as u64;

    let block_size = cta_size;
    let num_blocks = div_ceil(npoints, block_size);
    let num_gpus = ctx.system().devices().len() as u64;
    let num_superblocks = round_up(div_ceil(npoints * nfeatures, chunk_size), num_gpus);
    let blocks_per_superblock = div_ceil(num_blocks, num_superblocks);
    let superblock_size = block_size * blocks_per_superblock;

    let dist = BlockCyclic::new(superblock_size);
    let points: Array<float> = ctx.empty((n, nfeatures), dist)?;
    let membership: Array<int> = ctx.zeros(n, dist)?;
    let centers: Array<float> = ctx.empty((nclusters, nfeatures), ReplicateDist::new())?;
    let sizes: Array<u64> = ctx.empty(nclusters, ReplicateDist::new())?;

    let _real_centers = init_points(&points, nclusters)?;
    centers.assign_from(&points.slice((..nclusters, ..)))?;

    let dist = CentralizeDist::root();
    let new_centers_sums: Array<float> = ctx.empty(centers.extents(), dist)?;
    let new_sizes: Array<u64> = ctx.empty(sizes.extents(), dist)?;
    let num_deltas: Array<u64> = ctx.empty((), dist)?;
    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        for _ in 0..niter {
            new_centers_sums.fill_zeros()?;
            new_sizes.fill_zeros()?;
            num_deltas.fill_zeros()?;

            kernels.update_membership.launch(
                npoints,
                cta_size,
                BlockCyclic::new(superblock_size),
                (
                    npoints,
                    nclusters,
                    nfeatures,
                    &points,
                    &centers,
                    &membership,
                    &new_centers_sums,
                    &new_sizes,
                    &num_deltas,
                ),
            )?;

            new_sizes.assign_to(&sizes)?;

            kernels.compute_centers.launch(
                (nclusters, nfeatures),
                1,
                CentralizeDist::root(),
                (&new_centers_sums, &sizes, &centers),
            )?;
        }

        Ok(())
    })?;

    println!("n={} k={} chunk_size={} {}", n, k, superblock_size, time);

    Ok(())
}

/// Execute for different problem sizes
pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernels = compile_kernels(&ctx)?;
    let mut n = 1024;
    let max_n = args.end.unwrap_or(1_000_000_000);

    while n <= max_n {
        if n >= args.begin {
            run_kmeans(&ctx, &kernels, n, 40, 100_000_000)?;
        }

        n *= 2;
    }

    Ok(())
}

/// Execute for different values of k
pub(crate) fn run_k_tuning(ctx: Context, n: Option<u64>) -> Result<()> {
    let kernels = compile_kernels(&ctx)?;
    let n = n.unwrap_or(1_000_000_000);
    let mut k = 5;
    let max_k = 80;

    while k <= max_k {
        run_kmeans(&ctx, &kernels, n, k, 100_000_000)?;
        k += 5;
    }

    Ok(())
}

/// Execute for different chunk sizes
pub(crate) fn run_chunksize_tuning(ctx: Context, n: Option<u64>) -> Result<()> {
    let kernels = compile_kernels(&ctx)?;
    let n = n.unwrap_or(1_000_000_000);
    let mut chunk_size = 250_000 * 4;

    while chunk_size < 4 * n {
        run_kmeans(&ctx, &kernels, n, 40, chunk_size)?;
        chunk_size *= 2;
    }

    Ok(())
}
