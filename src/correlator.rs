use crate::util::benchmark;
use anyhow::Result;
use lightning::api::distribution::TileDist;
use lightning::api::*;
use lightning::types::util::{div_ceil, round_up};
use lightning::types::{
    float, float4, CastError, DataType, DataValue, HasDataType, DTYPE_FLOAT4, DTYPE_INT,
};
use std::convert::{TryFrom, TryInto};
use std::mem;
use std::mem::{align_of, size_of};

#[derive(Debug, Default, Clone, Copy)]
#[repr(C, align(16))]
struct Polarizations([float; 8]);

impl HasDataType for Polarizations {
    fn data_type() -> DataType {
        DataType::custom(
            "polarizations_t",
            align_of::<Polarizations>(),
            size_of::<Polarizations>(),
        )
        .unwrap()
    }
}

impl From<Polarizations> for DataValue {
    fn from(v: Polarizations) -> Self {
        let data: [u8; size_of::<Polarizations>()] = unsafe { mem::transmute_copy(&v) };

        DataValue::from_raw_data(&data, Polarizations::data_type())
    }
}

impl TryFrom<DataValue> for Polarizations {
    type Error = CastError;

    fn try_from(value: DataValue) -> Result<Self, Self::Error> {
        if value.data_type() == Polarizations::data_type() {
            if let Ok(x) = value.as_raw_data().try_into() {
                let x: [u8; size_of::<Polarizations>()] = x;
                return Ok(unsafe { mem::transmute_copy(&x) });
            }
        }

        Err(CastError)
    }
}

fn compile_kernel(ctx: &Context) -> Result<CudaKernel> {
    CudaKernelBuilder::from_file("resources/gpu_correlator_3x2.cu", "correlate_3x2")?
        .param_array("devSamples", DTYPE_FLOAT4)
        .param_array("devVisibilities", Polarizations::data_type())
        .param_value("nrTimes", DTYPE_INT)
        .param_value("nrStations", DTYPE_INT)
        .annotate(
            "global [channel, stat0, stat3] => \
               read devSamples[3*stat0 : 3*stat0+3, channel, :], \
               read devSamples[2*stat3 : 2*stat3+2, channel, :], \
               write devVisibilities[3*stat0 : 3*stat0+3, 2*stat3 : 2*stat3+2, channel]",
        )?
        .compile(ctx)
        .map_err(|e| e.into())
}

fn execute_kernel(
    ctx: &Context,
    nstations: u64,
    nchannels: u64,
    ntimes: u64,
    channels_per_superblock: u64,
    kernel: &CudaKernel,
) -> Result<()> {
    let tile_size = [3, 2];
    let ngpus = ctx.system().devices().len() as u64;
    let num_superblocks = round_up(div_ceil(nchannels, channels_per_superblock), ngpus);
    let channels_per_superblock = div_ceil(nchannels, num_superblocks);

    let samples: Array<float4> = ctx.empty(
        (nstations, nchannels, ntimes),
        TileDist::new((nstations, channels_per_superblock, ntimes)),
    )?;

    samples.fill([1.0, 1.0, 1.0, 1.0])?;
    ctx.synchronize()?;

    let time = benchmark(&ctx, || {
        let visibilities: Array<Polarizations> = ctx.empty(
            (nstations, nstations, nchannels),
            TileDist::new((nstations, nstations, channels_per_superblock)),
        )?;

        kernel.launch(
            (
                nchannels,
                nstations / tile_size[0],
                nstations / tile_size[1],
            ),
            (1, 16, 16),
            TileDist::new((channels_per_superblock, nstations, nstations)),
            (&samples, &visibilities, ntimes, nstations),
        )?;

        Ok(())
    })?;

    println!(
        "n={} {} nstations={} nchannels={} ntimes={} channels_per_superblock={}",
        nchannels, time, nstations, nchannels, ntimes, channels_per_superblock,
    );

    Ok(())
}

pub(crate) fn run(ctx: Context, max_n: Option<u64>) -> Result<()> {
    let kernel = compile_kernel(&ctx)?;
    let mut num_channels = 1;
    let max_channels = max_n.unwrap_or(1024);

    while num_channels <= max_channels {
        execute_kernel(&ctx, 256, num_channels, 768, 64, &kernel)?;

        num_channels *= 2;
    }

    Ok(())
}
