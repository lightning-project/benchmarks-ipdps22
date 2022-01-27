#include <cub/cub.cuh>
#include <assert.h>

#ifndef TILE_FACTOR_X
#define TILE_FACTOR_X (1)
#endif
#ifndef TILE_FACTOR_Y
#define TILE_FACTOR_Y (1)
#endif
#ifndef TILE_FACTOR_Z
#define TILE_FACTOR_Z (1)
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y (1)
#endif
#ifndef BLOCK_SIZE_Z
#define BLOCK_SIZE_Z (1)
#endif

#define TILE_SIZE_X (BLOCK_SIZE_X * TILE_FACTOR_X)
#define TILE_SIZE_Y (BLOCK_SIZE_Y * TILE_FACTOR_Y)
#define TILE_SIZE_Z (BLOCK_SIZE_Z * TILE_FACTOR_Z)
#define WARP_SIZE (32)
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#ifdef NDEBUG
#define ASSERT(expr) do { __builtin_assume(expr); } while (0)
#else
#define ASSERT(expr) do { assert(expr); } while (0)
#endif

#define ASSERT_BLOCK_1D \
    do { \
        ASSERT(blockDim.x == BLOCK_SIZE_X); \
        ASSERT(blockDim.y == 1); \
        ASSERT(blockDim.z == 1); \
        ASSERT(threadIdx.x < BLOCK_SIZE_X); \
        ASSERT(threadIdx.y == 0); \
        ASSERT(threadIdx.z == 0); \
    } while (0)

#define ASSERT_BLOCK_2D \
    do { \
        ASSERT(blockDim.x == BLOCK_SIZE_X); \
        ASSERT(blockDim.y == BLOCK_SIZE_Y); \
        ASSERT(blockDim.z == 1); \
        ASSERT(threadIdx.x < BLOCK_SIZE_X); \
        ASSERT(threadIdx.y < BLOCK_SIZE_Y); \
        ASSERT(threadIdx.z == 0); \
    } while (0)

#define ASSERT_NO_TILING \
    ASSERT(TILE_FACTOR_X * TILE_FACTOR_Y * TILE_FACTOR_Z == 1)

#ifndef COCAVG_STRATEGY
#define COCAVG_STRATEGY (0)
#endif

__device__ void calculate_gavg(
        dim3 blockIdx,
        size_t nrows,
        size_t ncols,
        const lightning::Matrix<double> Z,
        lightning::Scalar<double> gavg
) {
    ASSERT_BLOCK_2D;
    ASSERT_NO_TILING;

    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_SIZE_Y>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t row = BLOCK_SIZE_X * blockIdx.x + tx;
    size_t col = BLOCK_SIZE_Y * blockIdx.y + ty;
    double val = 0.0;

    if (row < nrows && col < ncols) {
        val = Z[row][col] / (double)(nrows * ncols);
    }

    val = BlockReduce(temp_storage).Sum(val);

    if (tx == 0 && ty == 0) {
        *gavg = val;
    }
}


__device__ void calculate_bincount(
    dim3 blockIdx,
    size_t nitems,
    size_t nclusters,
    const lightning::Vector<uint32_t> item_clusters,
    lightning::Vector<uint32_t> counts
) {
    ASSERT_BLOCK_1D;
    ASSERT_NO_TILING;

    using BlockReduce = cub::BlockReduce<uint32_t, BLOCK_SIZE_X>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t tx = threadIdx.x;
    size_t gx = BLOCK_SIZE_X * blockIdx.x + tx;

    uint32_t mycluster = ~0;
    if (gx < nitems) {
        mycluster = item_clusters[gx];
    }

    for (size_t cluster = 0; cluster < nclusters; cluster++) {
        uint32_t count = BlockReduce(temp_storage).Sum(mycluster == cluster);
        __syncthreads();

        if (tx == 0) {
            counts[cluster] = count;
        }
    }
}

__device__ void calculate_cocavg(
    dim3 blockIdx,
    size_t nrows,
    size_t ncols,
    size_t nclusters_row,
    size_t nclusters_col,
    const lightning::Matrix<double> Z,
    const lightning::Vector<uint32_t> row_clusters,
    const lightning::Vector<uint32_t> col_clusters,
    lightning::Matrix<double> cocavg
) {
    ASSERT_BLOCK_2D;

    struct key_value_t {
        uint32_t key;
        double value;
    };

    struct ScanOp {
        __device__ key_value_t operator()(
                const key_value_t &a,
                const key_value_t &b
        ) {
            if (a.key != b.key) {
                return {b.key, b.value};
            } else {
                return {b.key, a.value + b.value};
            }
        }
    };


    typedef cub::BlockRadixSort<
        uint32_t,
        BLOCK_SIZE_X,
        TILE_FACTOR_X * TILE_FACTOR_Y,
#if COCAVG_STRATEGY == 1
        short2,
#else
        double,
#endif
        4,
        true,
        cub::BLOCK_SCAN_WARP_SCANS,
        cudaSharedMemBankSizeFourByte,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_Z
    > BlockRadixSort;

    typedef cub::BlockDiscontinuity<
        uint32_t,
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_Z
    > BlockDiscontinuity;

    typedef cub::BlockScan<
        key_value_t,
        BLOCK_SIZE_X,
        cub::BLOCK_SCAN_RAKING,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_Z
    > BlockScan;


    __shared__ union {
        typename BlockRadixSort::TempStorage sort;
        typename BlockDiscontinuity::TempStorage disc;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    for (uint32_t i = threadIdx.x; i < nclusters_row; i += BLOCK_SIZE_X) {
        for (uint32_t j = threadIdx.y; j < nclusters_col; j += BLOCK_SIZE_Y) {
            cocavg[i][j] = 0.0;
        }
    }

    size_t index = 0;
    uint32_t keys[TILE_FACTOR_X * TILE_FACTOR_Y];
    double vals[TILE_FACTOR_X * TILE_FACTOR_Y];

#if COCAVG_STRATEGY == 1
    short2 indices[TILE_FACTOR_X * TILE_FACTOR_Y];
#endif

#pragma unroll
    for (size_t j = 0; j < TILE_FACTOR_Y; j++) {
#pragma unroll
        for (size_t i = 0; i < TILE_FACTOR_X; i++) {
            size_t row = blockIdx.x * TILE_SIZE_X + i * BLOCK_SIZE_X + threadIdx.x;
            size_t col = blockIdx.y * TILE_SIZE_Y + j * BLOCK_SIZE_Y + threadIdx.y;

            if (row < nrows && col < ncols) {
                uint32_t row_cluster = row_clusters[row];
                uint32_t col_cluster = col_clusters[col];

                keys[index] = row_cluster * nclusters_col + col_cluster;
            } else {
                keys[index] = ~0;
            }

#if COCAVG_STRATEGY == 1
            indices[index] = make_short2(
                    (short)(i * BLOCK_SIZE_X + threadIdx.x),
                    (short)(j * BLOCK_SIZE_Y + threadIdx.y)
            );
#else
            if (row < nrows && col < ncols) {
                vals[index] = Z[row][col];
            } else {
                vals[index] = 0.0;
            }
#endif

            index++;
        }
    }

    size_t end_bit = 32 - __clz(nclusters_col * nclusters_row);
#if COCAVG_STRATEGY == 1
    BlockRadixSort(temp_storage.sort).Sort(keys, indices, 0, end_bit);

    index = 0;

#pragma unroll
    for (index = 0; index < TILE_FACTOR_X * TILE_FACTOR_Y; index++) {
        short2 rel_rowcol = indices[index];
        size_t row = blockIdx.x * TILE_SIZE_X + (size_t) rel_rowcol.x;
        size_t col = blockIdx.y * TILE_SIZE_Y + (size_t) rel_rowcol.y;

        if (row < nrows && col < ncols) {
            vals[index] = Z[row * ncols + col];
        } else {
            vals[index] = 0.0;
        }
    }
#else
    BlockRadixSort(temp_storage.sort).Sort(keys, vals, 0, end_bit);
#endif

    __syncthreads();
    bool flags[TILE_FACTOR_X * TILE_FACTOR_Y];
    BlockDiscontinuity(temp_storage.disc).FlagTails(flags, keys, cub::Inequality());


    key_value_t pairs[TILE_FACTOR_X * TILE_FACTOR_Y];
    for (int i = 0; i < TILE_FACTOR_X * TILE_FACTOR_Y; i++) {
        pairs[i] = { keys[i], vals[i] };
    }

    __syncthreads();
    ScanOp scan_op;
    BlockScan(temp_storage.scan).InclusiveScan(pairs, pairs, scan_op);

    for (int i = 0; i < TILE_FACTOR_X * TILE_FACTOR_Y; i++) {
        if (!flags[i]) continue;

        int key = pairs[i].key;
        if (key == ~0) continue;

        double sum = pairs[i].value;
        size_t row_cluster = key / nclusters_col;
        size_t col_cluster = key % nclusters_col;

        cocavg[row_cluster][col_cluster] = sum;
    }
}

__device__ void process_cocavg(
    dim3 blockIdx,
    size_t nclusters_row,
    size_t nclusters_col,
    const lightning::Scalar<double> gavg,
    const lightning::Vector<uint32_t> nel_row_clusters,
    const lightning::Vector<uint32_t> nel_col_clusters,
    lightning::Matrix<double> cocavg,
    lightning::Matrix<double> log_cocavg,
    double epsilon
) {
    ASSERT_BLOCK_2D;
    ASSERT_NO_TILING;

    size_t col = BLOCK_SIZE_X * blockIdx.x + threadIdx.x;
    size_t row = BLOCK_SIZE_Y * blockIdx.y + threadIdx.y;

    if (row < nclusters_row && col < nclusters_col) {
        double result = cocavg[row][col];
        result += (*gavg) * epsilon;
        result /= nel_row_clusters[row] * nel_col_clusters[col] + epsilon;
        cocavg[row][col] = result;
        log_cocavg[row][col] = log(result);
    }
}

#define INVALID_CLUSTER ((uint32_t)~0)

__device__ void calculate_dist(
    dim3 blockIdx,
    size_t nrows,
    size_t ncols,
    size_t nclusters,
    const lightning::Matrix<double> Z,
    const lightning::Matrix<double> log_cocavg,
    const lightning::Vector<uint32_t> col_clusters,
    lightning::Matrix<double> partial_dists
) {
    ASSERT_BLOCK_2D;

#if BLOCK_SIZE_X > WARP_SIZE
    __shared__ float shared_dist[BLOCK_SIZE_Y][BLOCK_SIZE_X];
#endif

    double local_Z[TILE_FACTOR_X];
    uint32_t local_col_clusters[TILE_FACTOR_X];

#pragma unroll
    for (size_t k = 0; k < TILE_FACTOR_Y; k++) {
        size_t row = blockIdx.y * TILE_SIZE_Y + k * BLOCK_SIZE_Y + threadIdx.y;

#pragma unroll
        for (size_t i = 0; i < TILE_FACTOR_X; i++) {
            size_t col = blockIdx.x * TILE_SIZE_X + i * BLOCK_SIZE_X + threadIdx.x;

            if (col < ncols && row < nrows) {
                local_Z[i] = Z[row][col];
                local_col_clusters[i] = col_clusters[col];
            } else {
                local_Z[i] = 0;
                local_col_clusters[i] = INVALID_CLUSTER;
            }
        }

        for (size_t row_cluster = 0; row_cluster < nclusters; row_cluster++) {
            double dist = 0;

#pragma unroll
            for (size_t i = 0; i < TILE_FACTOR_X; i++) {
                size_t col_cluster = local_col_clusters[i];

                if (col_cluster == INVALID_CLUSTER) {
                    break;
                }

                double Yval = log_cocavg[row_cluster][col_cluster];  // [ncluster_col, nclusters_row]
                dist += local_Z[i] * Yval;
            }

#if BLOCK_SIZE_X > WARP_SIZE
            shared_dist[threadIdx.y][threadIdx.x] = dist;
            __syncthreads();

            if (threadIdx.x < WARP_SIZE) {
#pragma unroll
                for (int i = WARP_SIZE; i < BLOCK_SIZE_X; i += WARP_SIZE) {
                    if (i + threadIdx.x < BLOCK_SIZE_X) {
                        dist += shared_dist[threadIdx.y][i + threadIdx.x];
                    }
                }
            }

#pragma unroll
            for (size_t delta = 1; delta < WARP_SIZE; delta *= 2) {
                dist += __shfl_xor_sync(0xffffffff, dist, delta);
            }
#else
            ASSERT(((BLOCK_SIZE_X - 1) & BLOCK_SIZE_X) == 0); // power of two

            for (size_t delta = 1; delta < BLOCK_SIZE_X; delta *= 2) {
                dist += __shfl_xor_sync(0xffffffff, dist, delta, BLOCK_SIZE_X);
            }
#endif

            if (threadIdx.x == 0 && row < nrows) {
                partial_dists[row][row_cluster] = dist;
            }
        }
    }
}

__device__ void sum_y(
    dim3 blockIdx,
    size_t nclusters_row,
    size_t nclusters_col,
    const lightning::Matrix<double> cocavg,
    const lightning::Vector<uint32_t> nel_clusters,
    lightning::Vector<double> Y
) {
    ASSERT_BLOCK_1D;
    ASSERT_NO_TILING;
    size_t row = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    if (row < nclusters_row) {
        double sum = 0;

        for (size_t col = 0; col < nclusters_col; col++) {
            sum += (double)nel_clusters[col] * cocavg[row][col];
        }

        Y[row] = sum;
    }
}


__device__ void select_best_dist(
    dim3 blockIdx,
    size_t nitems,
    size_t nclusters,
    const lightning::Matrix<double> all_dists,
    const lightning::Vector<double> Y,
    lightning::Vector<uint32_t> item_clusters,
    lightning::Vector<double> min_dist
) {
    ASSERT_BLOCK_1D;
    ASSERT_NO_TILING;

    size_t gid = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    if (gid < nitems) {
        uint32_t best_cluster = 0;
        double best_dist = INFINITY;

        for (uint32_t cluster = 0; cluster < nclusters; cluster++) {
            double dist = Y[cluster] - all_dists[gid][cluster];

            if (dist < best_dist) {
                best_cluster = cluster;
                best_dist = dist;
            }
        }

        item_clusters[gid] = best_cluster;
        min_dist[gid] = best_dist;
    }
}

// Reductions kernels are not necessary, they are provided by Lightning
/*
template <typename T>
__device__ void reduce_elements(
    size_t nsegments,
    size_t segment_len,
    const T *partials,
    T *results
) {
    ASSERT_BLOCK_2D;
    ASSERT_NO_TILING;

#if BLOCK_SIZE_Y > 1
    __shared__ float shared_dist[BLOCK_SIZE_Y][BLOCK_SIZE_X];
#endif

    size_t segment_index = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    T local_sum = 0;

    if (segment_index < nsegments) {
        for (size_t i = threadIdx.y; i < segment_len; i+= BLOCK_SIZE_Y) {
            T value;

#if REDUCE_STRIPED
            value = partials[segment_index * segment_len + i];
#else
            value = partials[i * nsegments + segment_index];
#endif

            local_sum += value;
        }
    }

#if BLOCK_SIZE_Y > 1
    shared_dist[threadIdx.y][threadIdx.x] = local_sum;

    __syncthreads();
    if (threadIdx.y == 0 && segment_index < nsegments) {
#pragma unroll
        for (size_t i = 1; i < BLOCK_SIZE_Y; i++) {
            local_sum += shared_dist[i][threadIdx.x];
        }

        results[segment_index] = local_sum;
    }

#else
    if (segment_index < nsegments) {
        results[segment_index] = local_sum;
    }
#endif
}


extern "C"
__global__ void LAUNCH_BOUNDS reduce_uint(
    size_t nsegments,
    size_t segment_len,
    const uint32_t *partials,
    uint32_t *results
) {
    reduce_elements(nsegments, segment_len, partials, results);
}


extern "C"
__global__ void LAUNCH_BOUNDS reduce_double(
    size_t nsegments,
    size_t segment_len,
    const double *partials,
    double *results
) {
    reduce_elements(nsegments, segment_len, partials, results);
}

*/