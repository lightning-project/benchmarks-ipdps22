// Taken from the SHOC benchmark and modified for Lightning
// source: https://github.com/vetter/shoc/blob/0aea03beba2f09fcb5935cc11737372fe4de9ec0/src/cuda/level1/spmv/Spmv.cu

template <typename fpType>
__device__ void
spmv_ellpackr_kernel(
                     dim3 blockIdx,
                     const lightning::Matrix<fpType> val,
                     const lightning::Matrix<uint64_t> cols,
                     const lightning::Vector<uint64_t>  rowLengths,
                     const uint64_t dim,
                     lightning::Vector<fpType> out,
                     const lightning::Vector<fpType>  in)
{
    uint64_t t = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    if (t < dim)
    {
        fpType result = 0.0f;
        uint64_t max = rowLengths[t];
        for (uint64_t i = 0; i < max; i++) {
            result += val[t][i] * __ldg(&in[cols[t][i]]);
        }
        out[t] = result;
    }
}