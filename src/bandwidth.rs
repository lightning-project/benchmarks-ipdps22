use crate::util::benchmark;
use crate::Args;
use anyhow::Result;
use lightning::api::distribution::CentralizeDist;
use lightning::api::*;
use std::collections::VecDeque;
use std::mem::swap;

pub(crate) fn run(ctx: Context, _args: Args) -> Result<()> {
    let mut memories = VecDeque::new();
    for worker in ctx.system().workers() {
        memories.push_back(worker.memory_id);

        for device in &worker.devices {
            memories.push_back(device.memory_id);
        }
    }

    let size = 1_000_000_000; // 1 GB

    while let Some(a) = memories.pop_front() {
        for &b in &memories {
            let mut x: Array<u8> = ctx.zeros(size, CentralizeDist::new(a))?;
            let mut y: Array<u8> = ctx.zeros(size, CentralizeDist::new(b))?;
            ctx.synchronize()?;

            let result = benchmark(&ctx, || {
                x.assign_to(&y)?;
                swap(&mut x, &mut y);
                Ok(())
            })?;

            let size_in_gb = (size as f64) * 1e-9;
            let time_in_sec = result.average().as_secs_f64();

            println!(
                "bandwidth {:?} <-> {:?}: {} GB/sec",
                a,
                b,
                size_in_gb / time_in_sec
            );
        }
    }

    Ok(())
}
