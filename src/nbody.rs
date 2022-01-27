use crate::util::benchmark;
use anyhow::Result;
use lightning::api::distribution::BlockDist;
use lightning::api::*;
use lightning::types::{DTYPE_FLOAT, DTYPE_FLOAT4, DTYPE_INT};
use rand::prelude::*;
use std::mem::swap;

#[allow(non_camel_case_types)]
type float4 = [f32; 4];

fn init_array(position: &Array<float4>, rng: &mut dyn RngCore) -> Result<()> {
    let mut gen = || rng.gen_range(0.0f32..1.0f32);
    let values = (0..position.len() as usize)
        .map(|_| [gen(), gen(), gen(), gen()])
        .collect::<Vec<_>>();

    position.copy_from(&values)?;
    Ok(())
}

fn compile_kernel(ctx: &Context, threads_per_body: u64) -> Result<CudaKernel> {
    let threads_per_block = 512;
    assert_eq!(threads_per_body % threads_per_body, 0);
    assert!(threads_per_body.is_power_of_two());
    assert!(threads_per_body <= 32);

    let kernel = CudaKernelBuilder::from_file("resources/nbody.cu", "integrateBodies")?
        .options(&["-std=c++14"])
        .define("BODIES_PER_BLOCK", threads_per_block / threads_per_body)
        .define("THREADS_PER_BODY", threads_per_body)
        .param_array("newPos", DTYPE_FLOAT4)
        .param_array("oldPos", DTYPE_FLOAT4)
        .param_array("vel", DTYPE_FLOAT4)
        .param_value("numBodies", DTYPE_INT)
        .param_value("deltaTime", DTYPE_FLOAT)
        .param_value("damping", DTYPE_FLOAT)
        .annotate(
            "global i => read oldPos[:], \
                                   write newPos[i / THREADS_PER_BODY], \
                                   write vel[i / THREADS_PER_BODY]",
        )?
        .compile(&ctx)?;

    Ok(kernel)
}

fn run_kernel(ctx: &Context, kernel: &CudaKernel, npoints: u64) -> Result<()> {
    let threads_per_body = kernel.constants()["THREADS_PER_BODY"] as u64;
    let bodies_per_block = kernel.constants()["BODIES_PER_BLOCK"] as u64;
    let threads_per_block = threads_per_body * bodies_per_block;

    let dist = BlockDist::with_alignment(bodies_per_block);

    let velocity: Array<float4> = ctx.zeros(npoints, dist)?;
    let mut new_position: Array<float4> = ctx.empty(npoints, dist)?;
    let mut old_position: Array<float4> = ctx.empty(npoints, dist)?;

    let mut rng = SmallRng::seed_from_u64(0);
    init_array(&old_position, &mut rng)?;

    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        for _ in 0..10 {
            kernel.launch(
                npoints * threads_per_body,
                threads_per_block,
                BlockDist::with_alignment(threads_per_block),
                (
                    &new_position,
                    &old_position,
                    &velocity,
                    npoints,
                    0.01f32,
                    0.999f32,
                ),
            )?;

            swap(&mut new_position, &mut old_position);
        }

        Ok(())
    })?;

    println!(
        "n={} dim={} {} threads_per_body={} block_size={}",
        npoints * npoints,
        npoints,
        time,
        threads_per_body,
        threads_per_block
    );

    Ok(())
}

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    for &threads_per_body in &[1, 2, 4, 8, 16, 32] {
        if threads_per_body != 32 {
            continue;
        }
        let kernel = compile_kernel(&ctx, threads_per_body)?;

        let mut n = 256;
        let max_n = max_n.unwrap_or(10_000_000);

        while n <= max_n {
            let dim = (n as f64).sqrt() as u64;
            run_kernel(&ctx, &kernel, dim)?;

            n *= 2;
        }
    }

    Ok(())
}
