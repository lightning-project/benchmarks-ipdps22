/**
 * The kernel is assumed to be tuned to each device by selecting
 * the best performing combination of thread block dimensions
 * and tiling factors in X and Y. In this implementation tiling
 * in X increases the amount of work per thread block and tiling
 * in Y increases the amount of work per thread within the block.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 *
 */

/*
 * Optimized CUDA kernel for matrix multiplication
 *
 * This kernel is optimized according to the directions given
 * in: "Better performance at lower occupancy" by V. Volkov,
 * GPU Technology Conference, GTC 2010.
 *
 * The thread block dimensions (block_size_x, block_size_y)
 * and tiling factors (tile_size_x, tile_size_y) are to be
 * tuned towards each GPU. This kernel assumes that
 * block_size_x = block_size_y * tile_size_y.
 *
 * The kernel computes C+=A*B
 */
static_assert(block_size_x == block_size_y * tile_size_y,
        "block_size_x == block_size_y * tile_size_y");

#define float_type float

using namespace lightning;

__device__ void matrix_multiply(
        dim3 blockIdx,
        int N,
        int M,
        int P,
        Matrix<float_type> C, // N x P
        const Matrix<float_type> A, // N x M
        const Matrix<float_type> B // M x P
) {

    __shared__ float_type sA[block_size_y*tile_size_y][block_size_x];
    __shared__ float_type sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y;
    int k, kb;

    float_type sum[tile_size_y][tile_size_x];
#pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
#pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (k = 0; k < M; k += block_size_x) {

        __syncthreads();
#pragma unroll
        for (int i = 0; i < tile_size_y; i++) {
            float_type val = 0;
            if (y + i*block_size_y < N && k + tx < M) {
                val = A[y + i*block_size_y][k + tx];
            }

            sA[ty + i*block_size_y][tx] = val;

#pragma unroll
            for (int j = 0; j < tile_size_x; j++) {
                float_type val = 0;
                if (k + ty + i * block_size_y < M && x + j * block_size_x < P) {
                    val = B[k + ty + i * block_size_y][x + j * block_size_x];
                }

                sB[ty + i * block_size_y][tx + j * block_size_x] = val;
            }
        }
        __syncthreads();

        //compute
#pragma unroll
        for (kb = 0; kb < block_size_x; kb++) {

#pragma unroll
            for (int i = 0; i < tile_size_y; i++) {
#pragma unroll
                for (int j = 0; j < tile_size_x; j++) {
                    sum[i][j] += sA[ty + block_size_y * i][kb] * sB[kb][tx + j * block_size_x];
                }
            }
        }
    }

    //store result
#pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
#pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            if (y + block_size_y * i < N && x + block_size_x * j < P) {
                C[y + block_size_y * i][x + block_size_x * j] += sum[i][j];
            }
        }
    }

}




