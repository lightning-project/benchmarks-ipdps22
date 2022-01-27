use crate::util::benchmark;
use anyhow::Result;
use lightning::api::distribution::{CentralizeDist, StencilDist};
use lightning::api::*;
use lightning::types::{float, DTYPE_FLOAT, DTYPE_LONG};
use std::mem::swap;

static KERNEL: &str = r#"
#include <cub.cuh>

using namespace lightning;

__device__ void stencil(
        dim3 blockIdx,
        int64_t n,
        Scalar<float> max_diff,
        Vector<float> output,
        const Vector<float> input
) {
    int64_t i = (int64_t)blockDim.x  * (int64_t)blockIdx.x + (int64_t)threadIdx.x;

    float left = i-1 >= 0 ? (float)input[i - 1] : 0;
    float right = i+1 < n ? (float)input[i + 1] : 0;
    float old_val = i < n ? (float)input[i] : 0;
    float new_val = (left + old_val + right) / 3.0;
    float diff = 0.0;

    if (i < n) {
        output[i] = new_val;
        diff = fabs(new_val - old_val);
    }

    diff = cub::BlockReduce<float, 256>().Reduce(diff, cub::Max());
    if (threadIdx.x == 0) *max_diff = diff;
}
"#;

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    let stencil = CudaKernelBuilder::new(KERNEL, "stencil")
        .option("-std=c++14")
        .param_value("n", DTYPE_LONG)
        .param_array("max_diff", DTYPE_FLOAT)
        .param_array("output", DTYPE_FLOAT)
        .param_array("input", DTYPE_FLOAT)
        .annotate(
            "global i => write output[i], \
                         read input[i-1 : i+2], \
                         reduce(max) max_diff",
        )?
        .compile(&ctx)?;

    let cta_size = 256;
    let n = max_n.unwrap_or(100000000);
    let num_iterators = 10;
    let mut tile_size = cta_size * 100;

    while tile_size < n {
        let dist = StencilDist::new(tile_size, 1);

        let mut output: Array<float> = ctx.zeros(n, dist)?;
        let mut input: Array<float> = ctx.zeros(n, dist)?;
        let max_diff: Array<float> = ctx.zeros(num_iterators, CentralizeDist::root())?;
        ctx.synchronize()?;

        let time = benchmark(&ctx, || {
            for i in 0..num_iterators {
                stencil.launch(n, cta_size, dist, (n, &max_diff.slice(i), &output, &input))?;
                swap(&mut output, &mut input);
            }

            Ok(())
        })?;

        println!("n={} tile_size={} {}", n, tile_size, time);
        tile_size *= 2;
    }

    Ok(())
}
