use anyhow::Result;
use lightning::api::*;
use lightning::types::{DataType, HasDataType, DTYPE_U64};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(crate) struct BenchmarkResult {
    submission_times: Vec<Duration>,
    times: Vec<Duration>,
}

impl BenchmarkResult {
    pub(crate) fn times(&self) -> &[Duration] {
        &self.times
    }

    pub(crate) fn average(&self) -> Duration {
        self.times.iter().sum::<Duration>() / (self.times.len() as u32)
    }

    pub(crate) fn stddev(&self) -> Duration {
        let n = self.times.len() as f64;
        let sum: f64 = self.times.iter().map(|e| e.as_secs_f64()).sum();
        let avg = sum / n;
        let dev_sum: f64 = self
            .times
            .iter()
            .map(|e| (e.as_secs_f64() - avg).powi(2))
            .sum();
        Duration::from_secs_f64(f64::sqrt(dev_sum / n))
    }

    pub(crate) fn submission_average(&self) -> Duration {
        self.submission_times.iter().sum::<Duration>() / (self.times.len() as u32)
    }
}

impl Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sorted_times = self.times.clone();
        sorted_times.sort();

        write!(
            f,
            "time={} time_submit={} time_min={} time_max={} time_median={} time_stddev={}",
            self.average().as_secs_f64(),
            self.submission_average().as_secs_f64(),
            sorted_times.first().unwrap().as_secs_f64(),
            sorted_times.last().unwrap().as_secs_f64(),
            sorted_times[sorted_times.len() / 2].as_secs_f64(),
            self.stddev().as_secs_f64(),
        )
    }
}

pub(crate) fn benchmark_options(
    ctx: &Context,
    fun: &mut dyn FnMut() -> Result<()>,
    min_runs: usize,
    max_runs: usize,
    min_duration: Duration,
    max_duration: Duration,
) -> Result<BenchmarkResult> {
    assert!(min_runs <= max_runs);
    assert!(min_duration <= max_duration);
    assert!(min_runs >= 1);

    // Call once just to warm up the system
    fun()?;
    ctx.synchronize()?;

    let mut submission_times = vec![];
    let mut times = vec![];
    let start = Instant::now();
    ctx.synchronize()?;

    loop {
        let before = Instant::now();
        fun()?;
        let middle = Instant::now();
        ctx.synchronize()?;
        let after = Instant::now();

        submission_times.push(middle - before);
        times.push(after - before);

        if start.elapsed() > max_duration || times.len() >= max_runs {
            break;
        }

        if start.elapsed() > min_duration && times.len() > min_runs {
            break;
        }
    }

    // Compute average
    Ok(BenchmarkResult {
        times,
        submission_times,
    })
}

pub(crate) fn benchmark<F>(ctx: &Context, mut fun: F) -> Result<BenchmarkResult>
where
    F: FnMut() -> Result<()>,
{
    benchmark_options(
        ctx,
        &mut fun,
        5, // min 5 runs
        100, // max 100 runs
        Duration::from_secs(5), // min 5 seconds
        Duration::from_secs(60), // max 60 seconds
    )
}

const RANDOM_KERNEL_SOURCE: &'static str = r#"
using namespace lightning;

__device__ void random_init(
        dim3 blockIdx,
        uint64_t width,
        uint64_t height,
        Tensor<float> new_vals,
        float low,
        float high
) {
    uint64_t i = (int64_t)blockDim.x  * (int64_t)blockIdx.x + (int64_t)threadIdx.x;
    uint64_t j = (int64_t)blockDim.y  * (int64_t)blockIdx.y + (int64_t)threadIdx.y;
    uint64_t k = (int64_t)blockDim.z  * (int64_t)blockIdx.z + (int64_t)threadIdx.z;
    if (i >= width || j >= height) return;

    uint64_t a = 6364136223846793005;
    uint64_t b = 1442695040888963407;
    uint64_t seed = i ^ __brevll(j) ^ k;

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        seed = a * seed + b;
    }

    float factor = float(seed % (1 << 16)) / float(1 << 16);
    vals[i][j][k] = (factor * (high - low)) + low;
}
"#;

thread_local! {
    static RANDOM_KERNELS: RefCell<HashMap<DataType, CudaKernel>> = RefCell::new(HashMap::default());
}

fn generate_random_kernel(ctx: &Context, dtype: DataType, is_float: bool) -> Result<CudaKernel> {
    let fun = match is_float {
        true => "random_float",
        false => "random_integer",
    };

    let source = format!(
        "\
using namespace lightning;

template <typename T>
__device__ T random_integer(uint64_t seed, T low, T high) {{
    return (T)(seed % (uint64_t)(high - low)) + low;
}}

template <typename T>
__device__ T random_float(uint64_t seed, T low, T high) {{
    return (T)(double(seed % 1000000000) / double(1000000000)) * (high - low) + low;
}}

__device__ void random(
        dim3 blockIdx,
        uint64_t width,
        uint64_t height,
        uint64_t depth,
        Array3<{dtype}> vals,
        {dtype} low,
        {dtype} high
) {{
    uint64_t i = (uint64_t)blockDim.x  * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    uint64_t j = (uint64_t)blockDim.y  * (uint64_t)blockIdx.y + (uint64_t)threadIdx.y;
    uint64_t k = (uint64_t)blockDim.z  * (uint64_t)blockIdx.z + (uint64_t)threadIdx.z;
    if (i >= width || j >= height || k >= depth) return;

    uint64_t a = 6364136223846793005;
    uint64_t b = 1442695040888963407;
    uint64_t seed = i + a * j + a * a * k;

    #pragma unroll
    for (int i = 0; i < 10; i++) {{
        seed = a * seed + b;
    }}

    vals[i][j][k] = {fun}(seed, low, high);
}}
    ",
        dtype = dtype.ctype(),
        fun = fun
    );

    let kernel = CudaKernelBuilder::new(source, "random")
        .param_value("width", DTYPE_U64)
        .param_value("height", DTYPE_U64)
        .param_value("depth", DTYPE_U64)
        .param_array("vals", dtype)
        .param_value("low", dtype)
        .param_value("high", dtype)
        .annotate("global [i,j,k] => write vals[i,j,k]")?
        .compile(ctx)?;

    Ok(kernel)
}

pub(crate) trait RandomType: HasDataType {
    fn is_float() -> bool;
}

impl RandomType for f32 {
    fn is_float() -> bool {
        true
    }
}

impl RandomType for f64 {
    fn is_float() -> bool {
        true
    }
}

impl RandomType for u64 {
    fn is_float() -> bool {
        false
    }
}

impl RandomType for u32 {
    fn is_float() -> bool {
        false
    }
}

pub(crate) fn random_array<T: RandomType>(array: &Array<T>, low: T, high: T) -> Result<()> {
    use std::collections::hash_map::Entry;
    let ctx = array.context();

    RANDOM_KERNELS.with(|kernels| -> Result<()> {
        let mut kernels = kernels.borrow_mut();
        let kernel = match kernels.entry(T::data_type()) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(generate_random_kernel(
                array.context(),
                T::data_type(),
                T::is_float(),
            )?),
        };

        let block_size = [256, 1, 1];

        for (memory, region) in array.regions()? {
            let subarray = array.slice(region);
            let dim = subarray.extents();

            kernel.launch_one(
                dim,
                block_size,
                memory.best_affinity_executor(),
                (dim[0], dim[1], dim[2], &subarray, low, high),
            )?;
        }

        ctx.synchronize()?;
        Ok(())
    })
}
