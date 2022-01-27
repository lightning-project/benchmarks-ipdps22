#![allow(dead_code, non_snake_case)]
#![deny(unused_must_use)]

use anyhow::Result;
use clap::{App, Arg};
use lightning::types::Config;

mod allocator;
mod bandwidth;
mod black_scholes;
mod cgc;
mod correlator;
mod gemm;
mod hotspot;
mod kmeans;
mod mandelbrot;
mod md5hash;
mod microhh;
mod nbody;
mod saxpy;
mod spmv;
mod stencil;
mod util;

#[global_allocator]
static GLOBAL: allocator::Allocator = allocator::Allocator;

pub(crate) struct Args {
    program: String,
    begin: u64,
    end: Option<u64>,
    k: Option<u64>,
}

fn parse_args() -> Result<(Args, Config)> {
    let args = App::new("lightning benchmarks")
        .arg(Arg::with_name("name").takes_value(true).required(true))
        .arg(Arg::with_name("begin").takes_value(true))
        .arg(Arg::with_name("end").takes_value(true))
        .arg(Arg::with_name("k").long("k").takes_value(true))
        .arg(
            Arg::with_name("chunk-size")
                .long("chunk-size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("host-mem-size")
                .long("host-mem-size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("device-mem-size")
                .long("device-mem-size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("storage-dir")
                .long("storage-dir")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("storage-size")
                .long("storage-size")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("scheduling-lookahead")
                .long("scheduling-lookahead")
                .takes_value(true),
        )
        .get_matches();

    let mut config = Config::from_env();

    if let Some(dir) = args.value_of("storage-dir") {
        config.worker.storage_dir = Some(dir.into());
    }

    config.worker.storage_capacity = if let Some(size) = args.value_of("storage-size") {
        size.parse()?
    } else {
        100_000_000_000
    };

    config.worker.host_mem_block = 5_000_000_000;
    config.worker.host_mem_max = if let Some(size) = args.value_of("host-mem-size") {
        size.parse()?
    } else {
        50_000_000_000
    };

    if let Some(size) = args.value_of("device-mem-size") {
        config.worker.device_mem_max = Some(size.parse()?);
    }

    if let Some(size) = args.value_of("scheduling-lookahead") {
        config.worker.scheduling_lookahead_size = size.parse()?;
    }

    let (begin, end) = match (args.value_of("begin"), args.value_of("end")) {
        (Some(begin), Some(end)) => (begin.parse()?, Some(end.parse()?)),
        (Some(end), _) => (0, Some(end.parse()?)),
        _ => (0, None),
    };

    let k = match args.value_of("k") {
        Some(k) => Some(k.parse()?),
        None => None,
    };

    let result = Args {
        program: args.value_of("name").unwrap().to_string(),
        begin,
        end,
        k,
    };

    Ok((result, config))
}

fn main() -> Result<()> {
    let (args, config) = parse_args()?;

    lightning::initialize_logger();
    lightning::api::execute(config, |ctx| {
        let n = args.end;

        match &*args.program {
            "blackscholes" => {
                black_scholes::run(ctx, n)?;
            }
            "stencil" => {
                stencil::run(ctx, n)?;
            }
            "saxpy" => {
                saxpy::run(ctx, n)?;
            }
            "kmeans" => {
                kmeans::run(ctx, args)?;
            }
            "kmeans_tuning" => {
                kmeans::run_k_tuning(ctx, n)?;
            }
            "kmeans_chunksize" => {
                kmeans::run_chunksize_tuning(ctx, n)?;
            }
            "nbody" => {
                nbody::run(ctx, n)?;
            }
            "gemm" => {
                gemm::run(ctx, args)?;
            }
            "hotspot" => {
                hotspot::run(ctx, args)?;
            }
            "mandelbrot" => {
                mandelbrot::run(ctx, n)?;
            }
            "mandelbrot_tilesize" => {
                mandelbrot::run_tilesize_tuning(ctx)?;
            }
            "correlator" => {
                correlator::run(ctx, n)?;
            }
            "spmv" => {
                spmv::run(ctx, args)?;
            }
            "md5hash" => {
                md5hash::run(ctx, args)?;
            }
            "bandwidth" => {
                bandwidth::run(ctx, args)?;
            }
            "cgc" => {
                cgc::run(ctx, args)?;
            }
            s => {
                eprintln!("invalid program: {:?}", s);
            }
        }

        Ok(())
    })?;

    Ok(())
}
