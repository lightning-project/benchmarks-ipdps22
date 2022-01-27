/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define FLOAT_TYPE float
#define vec3 float3
#define vec4 float4
#define THREADS_PER_BLOCK ((BODIES_PER_BLOCK) * (THREADS_PER_BODY))

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}


template <typename T>
__device__ T getSofteningSquared()
{
    return (T) 0.001;
}

__device__ vec3
bodyBodyInteraction(vec3 ai,
                    vec4 bi,
                    vec4 bj)
{
    vec3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    auto distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<FLOAT_TYPE>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    auto invDist = rsqrt_T(distSqr);
    auto invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    auto s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ vec3
computeBodyAccel(vec4 bodyPos,
                 const lightning::Vector<vec4> positions,
                 int numBodies,
                 cg::thread_block cta)
{
    __shared__ vec4 sharedPos[THREADS_PER_BLOCK];

    vec3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numBodies; tile += THREADS_PER_BLOCK)
    {
        sharedPos[threadIdx.x] = positions[tile + threadIdx.x];

        cg::sync(cta);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128
        for (
                unsigned int counter = threadIdx.x % THREADS_PER_BODY;
                counter < THREADS_PER_BLOCK;
                counter += THREADS_PER_BODY
        ) {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}

__device__ void
integrateBodies(
                dim3 blockIdx,
                lightning::Vector<vec4> newPos,
                const lightning::Vector<vec4> oldPos,
                lightning::Vector<vec4> vel,
                unsigned int numBodies,
                float deltaTime,
                float damping)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x) / THREADS_PER_BODY;

    if (index >= numBodies) {
        return;
    }

    vec4 position = oldPos[index];
    vec3 accel = computeBodyAccel(position, oldPos, numBodies, cta);

    if (THREADS_PER_BODY > 1) {
#pragma unroll 32
        for (int mask = 1; mask < THREADS_PER_BODY; mask *= 2) {
            accel.x += __shfl_xor(accel.x, mask);
            accel.y += __shfl_xor(accel.y, mask);
            accel.z += __shfl_xor(accel.z, mask);
        }
        if (threadIdx.x % THREADS_PER_BODY != 0) return;
    }

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    vec4 velocity = vel[index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[index] = position;
    vel[index]    = velocity;
}