/*
Copyright (C) 2009 Rob van Nieuwpoort & John Romein
Astron
P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, nieuwpoort@astron.nl

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <array>
using polarizations_t = float4[2];

__device__ void correlate_3x2(
        dim3 blockIdx,
        const lightning::Tensor<float4> devSamples,  // station x channel x time
        lightning::Tensor<polarizations_t> devVisibilities,   // station x station x channel
        unsigned nrTimes,
        unsigned nrStations
) {
    unsigned channel   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stat0  = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned stat3  = blockIdx.z * blockDim.z + threadIdx.z;

    // 48 registers
    float v0x3xr = 0.0f, v0x3xi = 0.0f;
    float v0x3yr = 0.0f, v0x3yi = 0.0f;
    float v0y3xr = 0.0f, v0y3xi = 0.0f;
    float v0y3yr = 0.0f, v0y3yi = 0.0f;
    float v1x3xr = 0.0f, v1x3xi = 0.0f;
    float v1x3yr = 0.0f, v1x3yi = 0.0f;
    float v1y3xr = 0.0f, v1y3xi = 0.0f;
    float v1y3yr = 0.0f, v1y3yi = 0.0f;
    float v2x3xr = 0.0f, v2x3xi = 0.0f;
    float v2x3yr = 0.0f, v2x3yi = 0.0f;
    float v2y3xr = 0.0f, v2y3xi = 0.0f;
    float v2y3yr = 0.0f, v2y3yi = 0.0f;
    float v0x4xr = 0.0f, v0x4xi = 0.0f;
    float v0x4yr = 0.0f, v0x4yi = 0.0f;
    float v0y4xr = 0.0f, v0y4xi = 0.0f;
    float v0y4yr = 0.0f, v0y4yi = 0.0f;
    float v1x4xr = 0.0f, v1x4xi = 0.0f;
    float v1x4yr = 0.0f, v1x4yi = 0.0f;
    float v1y4xr = 0.0f, v1y4xi = 0.0f;
    float v1y4yr = 0.0f, v1y4yi = 0.0f;
    float v2x4xr = 0.0f, v2x4xi = 0.0f;
    float v2x4yr = 0.0f, v2x4yi = 0.0f;
    float v2y4xr = 0.0f, v2y4xi = 0.0f;
    float v2y4yr = 0.0f, v2y4yi = 0.0f;

    if (stat0 + 2 < nrStations && stat3 + 1 < nrStations && stat0 <= stat3) {
        for (unsigned time = 0; time < nrTimes; time++) {
            // 20 registers, 68 total
            float4 sample0 = devSamples[stat0 + 0][channel][time];
            float4 sample1 = devSamples[stat0 + 1][channel][time];
            float4 sample2 = devSamples[stat0 + 2][channel][time];
            float4 sample3 = devSamples[stat3 + 0][channel][time];
            float4 sample4 = devSamples[stat3 + 1][channel][time];

            v0x3xr += sample0.x * sample3.x;
            v0x3xi += sample0.y * sample3.x;
            v0x3yr += sample0.x * sample3.z;
            v0x3yi += sample0.y * sample3.z;
            v0y3xr += sample0.z * sample3.x;
            v0y3xi += sample0.w * sample3.x;
            v0y3yr += sample0.z * sample3.z;
            v0y3yi += sample0.w * sample3.z;
            v0x3xr += sample0.y * sample3.y;
            v0x3xi -= sample0.x * sample3.y;
            v0x3yr += sample0.y * sample3.w;
            v0x3yi -= sample0.x * sample3.w;
            v0y3xr += sample0.w * sample3.y;
            v0y3xi -= sample0.z * sample3.y;
            v0y3yr += sample0.w * sample3.w;
            v0y3yi -= sample0.z * sample3.w;

            v0x4xr += sample0.x * sample4.x;
            v0x4xi += sample0.y * sample4.x;
            v0x4yr += sample0.x * sample4.z;
            v0x4yi += sample0.y * sample4.z;
            v0y4xr += sample0.z * sample4.x;
            v0y4xi += sample0.w * sample4.x;
            v0y4yr += sample0.z * sample4.z;
            v0y4yi += sample0.w * sample4.z;
            v0x4xr += sample0.y * sample4.y;
            v0x4xi -= sample0.x * sample4.y;
            v0x4yr += sample0.y * sample4.w;
            v0x4yi -= sample0.x * sample4.w;
            v0y4xr += sample0.w * sample4.y;
            v0y4xi -= sample0.z * sample4.y;
            v0y4yr += sample0.w * sample4.w;
            v0y4yi -= sample0.z * sample4.w;

            v1x3xr += sample1.x * sample3.x;
            v1x3xi += sample1.y * sample3.x;
            v1x3yr += sample1.x * sample3.z;
            v1x3yi += sample1.y * sample3.z;
            v1y3xr += sample1.z * sample3.x;
            v1y3xi += sample1.w * sample3.x;
            v1y3yr += sample1.z * sample3.z;
            v1y3yi += sample1.w * sample3.z;
            v1x3xr += sample1.y * sample3.y;
            v1x3xi -= sample1.x * sample3.y;
            v1x3yr += sample1.y * sample3.w;
            v1x3yi -= sample1.x * sample3.w;
            v1y3xr += sample1.w * sample3.y;
            v1y3xi -= sample1.z * sample3.y;
            v1y3yr += sample1.w * sample3.w;
            v1y3yi -= sample1.z * sample3.w;

            v1x4xr += sample1.x * sample4.x;
            v1x4xi += sample1.y * sample4.x;
            v1x4yr += sample1.x * sample4.z;
            v1x4yi += sample1.y * sample4.z;
            v1y4xr += sample1.z * sample4.x;
            v1y4xi += sample1.w * sample4.x;
            v1y4yr += sample1.z * sample4.z;
            v1y4yi += sample1.w * sample4.z;
            v1x4xr += sample1.y * sample4.y;
            v1x4xi -= sample1.x * sample4.y;
            v1x4yr += sample1.y * sample4.w;
            v1x4yi -= sample1.x * sample4.w;
            v1y4xr += sample1.w * sample4.y;
            v1y4xi -= sample1.z * sample4.y;
            v1y4yr += sample1.w * sample4.w;
            v1y4yi -= sample1.z * sample4.w;

            v2x3xr += sample2.x * sample3.x;
            v2x3xi += sample2.y * sample3.x;
            v2x3yr += sample2.x * sample3.z;
            v2x3yi += sample2.y * sample3.z;
            v2y3xr += sample2.z * sample3.x;
            v2y3xi += sample2.w * sample3.x;
            v2y3yr += sample2.z * sample3.z;
            v2y3yi += sample2.w * sample3.z;
            v2x3xr += sample2.y * sample3.y;
            v2x3xi -= sample2.x * sample3.y;
            v2x3yr += sample2.y * sample3.w;
            v2x3yi -= sample2.x * sample3.w;
            v2y3xr += sample2.w * sample3.y;
            v2y3xi -= sample2.z * sample3.y;
            v2y3yr += sample2.w * sample3.w;
            v2y3yi -= sample2.z * sample3.w;

            v2x4xr += sample2.x * sample4.x;
            v2x4xi += sample2.y * sample4.x;
            v2x4yr += sample2.x * sample4.z;
            v2x4yi += sample2.y * sample4.z;
            v2y4xr += sample2.z * sample4.x;
            v2y4xi += sample2.w * sample4.x;
            v2y4yr += sample2.z * sample4.z;
            v2y4yi += sample2.w * sample4.z;
            v2x4xr += sample2.y * sample4.y;
            v2x4xi -= sample2.x * sample4.y;
            v2x4yr += sample2.y * sample4.w;
            v2x4yi -= sample2.x * sample4.w;
            v2y4xr += sample2.w * sample4.y;
            v2y4xi -= sample2.z * sample4.y;
            v2y4yr += sample2.w * sample4.w;
            v2y4yi -= sample2.z * sample4.w;
        }

        float4 *dst = &devVisibilities[stat0 + 0][stat3 + 0][channel][0];
        dst[0].x = v0x3xr;
        dst[0].y = v0x3xi;
        dst[0].z = v0x3yr;
        dst[0].w = v0x3yi;
        dst[1].x = v0y3xr;
        dst[1].y = v0y3xi;
        dst[1].z = v0y3yr;
        dst[1].w = v0y3yi;

        dst = &devVisibilities[stat0 + 1][stat3 + 0][channel][0];
        dst[0].x = v1x3xr;
        dst[0].y = v1x3xi;
        dst[0].z = v1x3yr;
        dst[0].w = v1x3yi;
        dst[1].x = v1y3xr;
        dst[1].y = v1y3xi;
        dst[1].z = v1y3yr;
        dst[1].w = v1y3yi;

        dst = &devVisibilities[stat0 + 2][stat3 + 0][channel][0];
        dst[0].x = v2x3xr;
        dst[0].y = v2x3xi;
        dst[0].z = v2x3yr;
        dst[0].w = v2x3yi;
        dst[1].x = v2y3xr;
        dst[1].y = v2y3xi;
        dst[1].z = v2y3yr;
        dst[1].w = v2y3yi;

        dst = &devVisibilities[stat0 + 0][stat3 + 1][channel][0];
        dst[0].x = v0x4xr;
        dst[0].y = v0x4xi;
        dst[0].z = v0x4yr;
        dst[0].w = v0x4yi;
        dst[1].x = v0y4xr;
        dst[1].y = v0y4xi;
        dst[1].z = v0y4yr;
        dst[1].w = v0y4yi;

        dst = &devVisibilities[stat0 + 1][stat3 + 1][channel][0];
        dst[0].x = v1x4xr;
        dst[0].y = v1x4xi;
        dst[0].z = v1x4yr;
        dst[0].w = v1x4yi;
        dst[1].x = v1y4xr;
        dst[1].y = v1y4xi;
        dst[1].z = v1y4yr;
        dst[1].w = v1y4yi;

        dst = &devVisibilities[stat0 + 2][stat3 + 1][channel][0];
        dst[0].x = v2x4xr;
        dst[0].y = v2x4xi;
        dst[0].z = v2x4yr;
        dst[0].w = v2x4yi;
        dst[1].x = v2y4xr;
        dst[1].y = v2y4xi;
        dst[1].z = v2y4yr;
        dst[1].w = v2y4yi;
    }
}