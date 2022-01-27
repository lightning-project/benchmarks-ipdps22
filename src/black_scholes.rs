use crate::util::{benchmark, random_array};
use anyhow::Result;
use lightning::api::distribution::RowBlockCyclic;
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{float, DTYPE_FLOAT, DTYPE_U64};

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    const REPEATS: usize = 3;
    let max_n = max_n.unwrap_or(1_000_000_000_000);

    let kernel = CudaKernelBuilder::from_file("resources/black_scholes.cu", "BlackScholesGPU")?
        .param_array("d_CallResult", DTYPE_FLOAT)
        .param_array("d_PutResult", DTYPE_FLOAT)
        .param_array("d_StockPrice", DTYPE_FLOAT)
        .param_array("d_OptionStrike", DTYPE_FLOAT)
        .param_array("d_OptionYears", DTYPE_FLOAT)
        .param_value("Riskfree", DTYPE_FLOAT)
        .param_value("Volatility", DTYPE_FLOAT)
        .param_value("optN", DTYPE_U64)
        .annotate(
            "global i => read d_StockPrice[i], read d_OptionStrike[i], read d_OptionYears[i]",
        )?
        .annotate("global i => write d_CallResult[i], write d_PutResult[i]")?
        .compile(&ctx)?;

    let ngpus = ctx.system().devices().len() as u64;
    let mut n = 4096;

    while n <= max_n {
        let cta_size = 256;
        let risk_free = 0.02_f32;
        let volatility = 0.30_f32;

        // Conditions
        // - maximum blocks size is 1e8
        // - no. of blocks is multiple of ngpus
        // - block size is multiple of cta_size
        let max_block_size = 100_000_000;
        let num_blocks = round_up(div_ceil(n, max_block_size), ngpus);
        let block_size = round_up(div_ceil(n, num_blocks), cta_size);
        let dist = RowBlockCyclic::new(block_size);

        let call_result: Array<float> = ctx.empty(n, dist)?;
        let put_result: Array<float> = ctx.empty(n, dist)?;
        let stock_price: Array<float> = ctx.empty(n, dist)?;
        let option_strike: Array<float> = ctx.empty(n, dist)?;
        let option_years: Array<float> = ctx.empty(n, dist)?;

        random_array(&stock_price, 5.0, 30.0)?;
        random_array(&option_strike, 10.0, 100.0)?;
        random_array(&option_years, 0.25, 10.0)?;

        let time = benchmark(&ctx, || {
            kernel.launch_like(
                &call_result,
                cta_size,
                (
                    &call_result,
                    &put_result,
                    &stock_price,
                    &option_strike,
                    &option_years,
                    risk_free,
                    volatility,
                    n,
                ),
            )?;

            Ok(())
        })?;

        println!("n={} {}", n, time);
        n *= 2;
    }

    Ok(())
}
