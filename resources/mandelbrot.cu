/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
// Modified on 12th of July 2021 for use as benchmark in Lightning by stijnh

template<class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos, const T xJParam, const T yJParam, const int crunch,
                                     const bool isJulia)
{
    T x, y, xx, yy, xC, yC ;
    if (isJulia)
    {
        xC = xJParam ;
        yC = yJParam ;
        y = yPos;
        x = xPos;
        yy = y * y;
        xx = x * x;
    }
    else
    {
        xC = xPos ;
        yC = yPos ;
        y = 0 ;
        x = 0 ;
        yy = 0 ;
        xx = 0 ;
    }
    int i = crunch;
    while (--i && (xx + yy < T(4.0)))
    {
        y = x * y * T(2.0) + yC ;
        x = xx - yy + xC ;
        yy = y * y;
        xx = x * x;
    }
    return i; // i > 0 ? crunch - i : 0;
} // CalcMandelbrot

// The Mandelbrot CUDA GPU thread function
template<class T>
__device__ void Mandelbrot0(
        dim3 blockIdx,
        lightning::Matrix<uchar4> dst,
        const int imageW,
        const int imageH,
        const int crunch,
        const T xOff,
        const T yOff,
        const T xJP,
        const T yJP,
        const T scale,
        const uchar4 colors,
        const int frame,
        const int animationFrame,
        const bool isJ
) {
    // process this block
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH))
    {
        // Calculate the location
        const T xPos = (T)ix * scale + xOff;
        const T yPos = (T)iy * scale + yOff;

        // Calculate the Mandelbrot index for the current location
        int m = CalcMandelbrot<T>(xPos, yPos, xJP, yJP, crunch, isJ);
        //            int m = blockIdx.x;         // uncomment to see scheduling order
        m = m > 0 ? crunch - m : 0;

        // Convert the Mandelbrot index into a color
        uchar4 color;

        if (m)
        {
            m += animationFrame;
            color.x = m * colors.x;
            color.y = m * colors.y;
            color.z = m * colors.z;
        }
        else
        {
            color.x = 0;
            color.y = 0;
            color.z = 0;
        }

        // Output the pixel

        if (frame == 0)
        {
            color.w = 0;
            dst[iy][ix] = color;
        }
        else
        {
            int frame1 = frame + 1;
            int frame2 = frame1 / 2;
            uchar4 old_color = dst[iy][ix];
            old_color.x = (old_color.x * frame + color.x + frame2) / frame1;
            old_color.y = (old_color.y * frame + color.y + frame2) / frame1;
            old_color.z = (old_color.z * frame + color.z + frame2) / frame1;
            dst[iy][ix] = old_color;
        }
    }

} // Mandelbrot0