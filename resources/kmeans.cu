#include <cub.cuh>

using namespace lightning;

__device__ void update_membership(
        dim3 blockIdx,
        int64_t npoints,
        int64_t nclusters,
        int64_t nfeatures,
        const Matrix<float> points,  // [npoints x nfeatures]
        const Matrix<float> centers,  // [nclusters x nfeatures]
        Vector<int> membership, // [npoints]
        Matrix<float> new_centers_sums,  // [nclusters x nfeatures]
        Vector<uint64_t> new_sizes,
        Scalar<uint64_t> num_deltas
) {
    int best_cluster = -1;
    uint64_t point_offset = (uint64_t)blockIdx.x * (uint64_t)block_size;
    uint64_t i = point_offset + threadIdx.x;

    if (i < npoints) {
        float best_dist = 1e99;

        for (int cluster = 0; cluster < nclusters; cluster++) {
            float dist = 0.0;

            for (int j = 0; j < nfeatures; j++) {
                float delta = points[i][j] - centers[cluster][j];
                dist += delta * delta;
            }

            if (dist < best_dist) {
                best_cluster = cluster;
                best_dist = dist;
            }
        }
        membership[i] = best_cluster;
    }

    __shared__ int shared_membership[block_size];
    shared_membership[threadIdx.x] = best_cluster;
    __syncthreads();

    for (int i = threadIdx.x; i < nclusters * nfeatures; i += block_size) {
        int cluster = i / nfeatures;
        int feature = i % nfeatures;
        float accumulator = 0.0;
        uint64_t size = 0;

        for (int z = 0; z < block_size; z++) {
            if (shared_membership[z] == cluster) {
                size += 1;
                accumulator += points[point_offset + z][feature];
            }
        }

        new_centers_sums[cluster][feature] = accumulator;

        if (feature == 0) {
            new_sizes[cluster] = size;
        }
    }
}

__device__ void compute_centers(
        dim3 blockIdx,
        const Matrix<float> center_sums,
        const Vector<uint64_t> sizes,
        Matrix<float> new_centers
) {
    int cluster = blockIdx.x;
    int feature = blockIdx.y;

    if (threadIdx.x == 0) {
        new_centers[cluster][feature] = center_sums[cluster][feature] / sizes[cluster];
    }
}