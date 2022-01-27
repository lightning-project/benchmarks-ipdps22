use crate::util::benchmark;
use anyhow::Result;
use lightning::api::distribution::BlockCyclic;
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, DTYPE_FLOAT, DTYPE_LONG};

static KERNEL: &str = r#"
using namespace lightning;

__device__ void vector_add(
        dim3 blockIdx,
        uint64_t n,
        Vector<float> c,
        const Vector<float> a,
        const Vector<float> b
) {
    uint64_t i = (uint64_t)blockDim.x  * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    if (i >= n) return;

    c[i] = a[i] + b[i];
}
"#;

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    let vector_add = CudaKernelBuilder::new(KERNEL, "vector_add")
        .debugging(false)
        .param_value("n", DTYPE_LONG)
        .param_array("c", DTYPE_FLOAT)
        .param_array("a", DTYPE_FLOAT)
        .param_array("b", DTYPE_FLOAT)
        .annotate("global i => write c[i], read a[i], read b[i]")?
        .compile(&ctx)?;

    let ngpus = ctx.system().devices().len() as u64;
    let cta_size = 256;
    let mut n = 256;
    let max_n = max_n.unwrap_or(1 << 35);

    while n <= max_n {
        // Conditions
        // - maximum blocks size is 1e8
        // - no. of blocks is multiple of ngpus
        // - block size is multiple of cta_size
        let max_block_size = 100_000_000;
        let num_blocks = div_ceil(n, max_block_size * ngpus) * ngpus;
        let block_size = round_up(div_ceil(n, num_blocks), cta_size);
        let dist = BlockCyclic::new(block_size);

        let a: Array<float> = ctx.zeros(n, dist)?;
        let b: Array<float> = ctx.zeros(n, dist)?;
        let c: Array<float> = ctx.zeros(n, dist)?;
        ctx.synchronize()?;

        let time = benchmark(&ctx, || {
            vector_add.launch_like(&a, cta_size, (n, &c, &a, &b))?;
            ctx.synchronize()?;
            Ok(())
        })?;

        println!("workload={} n={} {}", n, n, time);
        n *= 2;
    }

    Ok(())
}
