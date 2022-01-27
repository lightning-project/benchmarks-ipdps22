use crate::util::{benchmark, random_array};
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::{AllGPUs, BlockCyclic, ReplicateDist};
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, HasDataType, DTYPE_U64};
use std::mem::swap;

type FloatType = f32;

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    let fun = format!("spmv_ellpackr_kernel<{}>", float::data_type().ctype());
    let spmv = CudaKernelBuilder::from_file("resources/Spmv.cu", fun)?
        .param_array("val", FloatType::data_type())
        .param_array("cols", DTYPE_U64)
        .param_array("rowLengths", DTYPE_U64)
        .param_value("dim", DTYPE_U64)
        .param_array("out", FloatType::data_type())
        .param_array("in", FloatType::data_type())
        .annotate("global i => read val[i,:], read cols[i,:], read rowLengths[i], read in[:], write out[i]")?
        .compile(ctx)?;

    Ok(spmv)
}

fn execute(ctx: &Context, dim: u64, max_deg: u64, spmv: &CudaKernel) -> Result<()> {
    let cta_size = 128;
    let niter = 10;
    let num_gpus = ctx.system().devices().len() as u64;
    let num_chunks = round_up(div_ceil(dim * max_deg, 100_000_000), num_gpus);
    let chunk_size = round_up(div_ceil(dim, num_chunks), cta_size);

    let dist = BlockCyclic::new(chunk_size);
    let row_lengths = ctx.empty(dim, dist)?;
    let cols = ctx.empty((dim, max_deg), dist)?;
    let vals: Array<FloatType> = ctx.empty((dim, max_deg), dist)?;
    let mut inputs: Array<FloatType> = ctx.empty(dim, ReplicateDist::with_memories(AllGPUs))?;
    let mut outputs: Array<FloatType> = ctx.empty(dim, ReplicateDist::with_memories(AllGPUs))?;
    let intermediate: Array<FloatType> = ctx.empty(dim, dist)?;

    random_array(&vals, 0.0 as FloatType, 1.0 as FloatType)?;
    random_array(&cols, 0, dim)?;
    random_array(&row_lengths, 0, max_deg)?;
    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        inputs.fill_ones()?;

        for _ in 0..niter {
            spmv.launch(
                dim,
                cta_size,
                dist,
                (&vals, &cols, &row_lengths, dim, &intermediate, &inputs),
            )?;

            outputs.assign_from(&intermediate)?;
            swap(&mut inputs, &mut outputs);
        }

        Ok(())
    })?;

    println!(
        "n={} dim={} max_deg={} num_chunks={} {}",
        dim * dim,
        dim,
        max_deg,
        num_chunks,
        time
    );
    Ok(())
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let k = args.k.unwrap_or(1000);
    let max_n = args.end.unwrap_or(1024 * 1024 * 1024);
    let mut n = 1;

    while n <= max_n {
        if n >= args.begin {
            let dim = (n as f64).sqrt().ceil() as u64;
            let deg = ((dim as f64) / (k as f64)).ceil() as u64;
            execute(&ctx, dim, deg, &kernel)?;
        }

        n *= 2;
    }

    Ok(())
}
