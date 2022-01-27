use crate::util::benchmark;
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::BlockCyclic;
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{DTYPE_INT, DTYPE_U64};
use std::convert::TryInto;

const CTA_SIZE: u64 = 1024;

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    let kernel = CudaKernelBuilder::from_file("resources/MD5Hash.cu", "FindKeyWithDigest_Kernel")?
        .define("block_size_x", CTA_SIZE)
        .param_value("searchDigest0", DTYPE_INT)
        .param_value("searchDigest1", DTYPE_INT)
        .param_value("searchDigest2", DTYPE_INT)
        .param_value("searchDigest3", DTYPE_INT)
        .param_value("keyspace", DTYPE_U64)
        .param_value("byteLength", DTYPE_U64)
        .param_value("valsPerByte", DTYPE_U64)
        .param_array("foundIndex", DTYPE_U64)
        .annotate("global _ => reduce(min) foundIndex")?
        .compile(ctx)?;

    Ok(kernel)
}

fn execute(ctx: &Context, n: u64, kernel: &CudaKernel) -> Result<()> {
    let keyspace = n;
    let vals_per_byte = 256;
    let byte_length: u64 = 7;
    let threadspace = div_ceil(keyspace, vals_per_byte);
    let result: Array<u64> = ctx.scalar(u64::MAX)?;
    ctx.synchronize()?;

    let mut needle = vec![0; byte_length as usize];
    needle[0] = 1;
    let search_digest = md5::compute(&needle);

    let num_gpus = ctx.system().devices().len() as u64;
    let num_chunks = round_up(div_ceil(keyspace, 5_000_000_000), num_gpus);
    let chunk_size = round_up(div_ceil(threadspace, num_chunks), CTA_SIZE);
    let dist = BlockCyclic::new(chunk_size);

    let time = benchmark(&ctx, || {
        result.fill(u64::MAX)?;

        kernel.launch(
            threadspace,
            CTA_SIZE,
            dist,
            (
                i32::from_ne_bytes(search_digest[0..4].try_into().unwrap()),
                i32::from_ne_bytes(search_digest[4..8].try_into().unwrap()),
                i32::from_ne_bytes(search_digest[8..12].try_into().unwrap()),
                i32::from_ne_bytes(search_digest[12..16].try_into().unwrap()),
                keyspace,
                byte_length,
                vals_per_byte,
                &result,
            ),
        )?;

        Ok(())
    })?;

    assert_eq!(result.to_vec()?, &[1]); // check result
    println!("n={} num_chunks={} {}", n, num_chunks, time);

    Ok(())
}

pub(crate) fn run(ctx: Context, args: Args) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let max_n = args.end.unwrap_or(1024 * 1024 * 1024);
    let mut n = 1;

    while n <= max_n {
        if n >= args.begin {
            execute(&ctx, n, &kernel)?;
        }

        n *= 2;
    }

    Ok(())
}
