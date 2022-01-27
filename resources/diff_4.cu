// Taken from the microhh project and adapted for Lightning
// source: https://github.com/microhh/microhh/blob/master/kernel_tuner/diff_4.cu

__global__
void diff_c_g(
        lightning::Array3<double> at,
        const lightning::Array3<double> a,
        const lightning::Array1<double> dzi4,
        const lightning::Array1<double> dzhi4,
        const double dxi, const double dyi, const double visc,
        //const int jj,     const int kk,
        const int istart, const int jstart, const int kstart,
        const int iend,   const int jend,   const int kend,
        const int icells, const int jcells, const int ngc)
        {
    const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
    const int k = blockIdx.z + kstart;

    constexpr double cg0  =   1./24.;
    constexpr double cg1  = -27./24.;
    constexpr double cg2  =  27./24.;
    constexpr double cg3  =  -1./24.;

    constexpr double bg0  = -23./24.;
    constexpr double bg1  =  21./24.;
    constexpr double bg2  =   3./24.;
    constexpr double bg3  =  -1./24.;

    constexpr double tg0  =   1./24.;
    constexpr double tg1  =  -3./24.;
    constexpr double tg2  = -21./24.;
    constexpr double tg3  =  23./24.;

    constexpr double cdg0 = -1460./576.;
    constexpr double cdg1 =   783./576.;
    constexpr double cdg2 =   -54./576.;
    constexpr double cdg3 =     1./576.;

    if (i < iend && j < jend && k < kend)
    {
        //const int ijk = i + j*jj + k*kk;

        //const int ii1 = 1;
        //const int ii2 = 2;
        //const int ii3 = 3;
        //const int jj1 = 1*jj;
        //const int jj2 = 2*jj;
        //const int jj3 = 3*jj;
        //const int kk1 = 1*kk;
        //const int kk2 = 2*kk;
        //const int kk3 = 3*kk;

        const double dxidxi = dxi*dxi;
        const double dyidyi = dyi*dyi;

        // bottom boundary
        if (k == kstart)
        {
            at[i][j][k] += visc * (cdg3*a[i-3][j][k] + cdg2*a[i-2][j][k] + cdg1*a[i-1][j][k] + cdg0*a[i][j][k] + cdg1*a[i+1][j][k] + cdg2*a[i+2][j][k] + cdg3*a[i+3][j][k])*dxidxi;
            at[i][j][k] += visc * (cdg3*a[i][j-3][k] + cdg2*a[i][j-2][k] + cdg1*a[i][j-1][k] + cdg0*a[i][j][k] + cdg1*a[i][j+1][k] + cdg2*a[i][j+2][k] + cdg3*a[i][j+3][k])*dyidyi;
            at[i][j][k] += visc * ( cg0*(bg0*a[i][j][k-2] + bg1*a[i][j][k-1] + bg2*a[i][j][k] + bg3*a[i][j][k+1]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*a[i][j][k] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*a[i][j][k] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(cg0*a[i][j][k] + cg1*a[i][j][k+1] + cg2*a[i][j][k+2] + cg3*a[i][j][k+3]) * dzhi4[k+2] ) * dzi4[k];
        }
        // top boundary
        else if (k == kend-1)
        {
            at[i][j][k] += visc * (cdg3*a[i-3][j][k] + cdg2*a[i-2][j][k] + cdg1*a[i-1][j][k] + cdg0*a[i][j][k] + cdg1*a[i+1][j][k] + cdg2*a[i+2][j][k] + cdg3*a[i+3][j][k])*dxidxi;
            at[i][j][k] += visc * (cdg3*a[i][j-3][k] + cdg2*a[i][j-2][k] + cdg1*a[i][j-1][k] + cdg0*a[i][j][k] + cdg1*a[i][j+1][k] + cdg2*a[i][j+2][k] + cdg3*a[i][j+3][k])*dyidyi;
            at[i][j][k] += visc * ( cg0*(cg0*a[i][j][k-3] + cg1*a[i][j][k-2] + cg2*a[i][j][k-1] + cg3*a[i][j][k]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*a[i][j][k] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*a[i][j][k] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(tg0*a[i][j][k-1] + tg1*a[i][j][k] + tg2*a[i][j][k+1] + tg3*a[i][j][k+2]) * dzhi4[k+2] ) * dzi4[k];
        }
        // interior
        else
        {
            at[i][j][k] += visc * (cdg3*a[i-3][j][k] + cdg2*a[i-2][j][k] + cdg1*a[i-1][j][k] + cdg0*a[i][j][k] + cdg1*a[i+1][j][k] + cdg2*a[i+2][j][k] + cdg3*a[i+3][j][k])*dxidxi;
            at[i][j][k] += visc * (cdg3*a[i][j-3][k] + cdg2*a[i][j-2][k] + cdg1*a[i][j-1][k] + cdg0*a[i][j][k] + cdg1*a[i][j+1][k] + cdg2*a[i][j+2][k] + cdg3*a[i][j+3][k])*dyidyi;
            at[i][j][k] += visc * ( cg0*(cg0*a[i][j][k-3] + cg1*a[i][j][k-2] + cg2*a[i][j][k-1] + cg3*a[i][j][k]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*a[i][j][k] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*a[i][j][k] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(cg0*a[i][j][k] + cg1*a[i][j][k+1] + cg2*a[i][j][k+2] + cg3*a[i][j][k+3]) * dzhi4[k+2] ) * dzi4[k];
        }
    }
        }



__global__
void diff_c_g_smem(
        lightning::Array3<double> at,
        const lightning::Array3<double> a,
        const lightning::Array1<double> dzi4,
        const lightning::Array1<double> dzhi4,
        const double dxi, const double dyi, const double visc,
        //const int jj,     const int kk,
        const int istart, const int jstart, const int kstart,
        const int iend,   const int jend,   const int kend,
        const int icells, const int jcells, const int ngc)
        {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i  = blockIdx.x*blockDim.x + threadIdx.x + istart;
    const int j  = blockIdx.y*blockDim.y + threadIdx.y + jstart;
    const int k  = blockIdx.z + kstart;
    const int bx = blockIdx.x*blockDim.x;
    const int by = blockIdx.y*blockDim.y;
    const int blockxpad = blockDim.x+2*ngc;
    const int blockypad = blockDim.y+2*ngc;

    // DANGER DANGER ghost cells harcoded..
    __shared__ double as[(block_size_y+6)*(block_size_x+6)];

    constexpr double cg0  =   1./24.;
    constexpr double cg1  = -27./24.;
    constexpr double cg2  =  27./24.;
    constexpr double cg3  =  -1./24.;

    constexpr double bg0  = -23./24.;
    constexpr double bg1  =  21./24.;
    constexpr double bg2  =   3./24.;
    constexpr double bg3  =  -1./24.;

    constexpr double tg0  =   1./24.;
    constexpr double tg1  =  -3./24.;
    constexpr double tg2  = -21./24.;
    constexpr double tg3  =  23./24.;

    constexpr double cdg0 = -1460./576.;
    constexpr double cdg1 =   783./576.;
    constexpr double cdg2 =   -54./576.;
    constexpr double cdg3 =     1./576.;

    // Read horizontal slice to shared memory
    #pragma unroll
    for (int tj=0; tj<block_size_y+6; tj+=block_size_y)
    {
        #pragma unroll
        for (int ti=0; ti<block_size_x+6; ti+=block_size_x)
        {
            const int is = ti+tx;
            const int js = tj+ty;

            const int ig = is+bx;
            const int jg = js+by;

            if (is < blockxpad && js < blockypad && ig < icells && jg < jcells)
            {
                const int ijs = is + js*blockxpad;
                as[ijs] = a[ig][jg][k];
            }
        }
    }

    __syncthreads();

    if (i < iend && j < jend && k < kend)
    {
        const int ijk  = i + j*jj + k*kk;              // index in global memory
        const int ijks = (tx+ngc)+(ty+ngc)*blockxpad;  // Same location in 2d shared mem

        const int ii1 = 1;
        const int ii2 = 2;
        const int ii3 = 3;
        //const int kk1 = 1*kk;
        //const int kk2 = 2*kk;
        //const int kk3 = 3*kk;

        const int jjs1 = 1*blockxpad;
        const int jjs2 = 2*blockxpad;
        const int jjs3 = 3*blockxpad;

        const double dxidxi = dxi*dxi;
        const double dyidyi = dyi*dyi;

        // bottom boundary
        if (k == kstart)
        {
            at[i][j][k] += visc * (cdg3*as[ijks-ii3 ] + cdg2*as[ijks-ii2 ] + cdg1*as[ijks-ii1 ] + cdg0*as[ijks] + cdg1*as[ijks+ii1 ] + cdg2*as[ijks+ii2 ] + cdg3*as[ijks+ii3 ])*dxidxi;
            at[i][j][k] += visc * (cdg3*as[ijks-jjs3] + cdg2*as[ijks-jjs2] + cdg1*as[ijks-jjs1] + cdg0*as[ijks] + cdg1*as[ijks+jjs1] + cdg2*as[ijks+jjs2] + cdg3*as[ijks+jjs3])*dyidyi;
            at[i][j][k] += visc * ( cg0*(bg0*a[i][j][k-2] + bg1*a[i][j][k-1] + bg2*as[ijks  ] + bg3*a[i][j][k+1]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*as[ijks  ] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*as[ijks  ] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(cg0*as[ijks  ] + cg1*a[i][j][k+1] + cg2*a[i][j][k+2] + cg3*a[i][j][k+3]) * dzhi4[k+2] ) * dzi4[k];
        }
        // top boundary
        else if (k == kend-1)
        {
            at[i][j][k] += visc * (cdg3*as[ijks-ii3 ] + cdg2*as[ijks-ii2 ] + cdg1*as[ijks-ii1 ] + cdg0*as[ijks] + cdg1*as[ijks+ii1 ] + cdg2*as[ijks+ii2 ] + cdg3*as[ijks+ii3 ])*dxidxi;
            at[i][j][k] += visc * (cdg3*as[ijks-jjs3] + cdg2*as[ijks-jjs2] + cdg1*as[ijks-jjs1] + cdg0*as[ijks] + cdg1*as[ijks+jjs1] + cdg2*as[ijks+jjs2] + cdg3*as[ijks+jjs3])*dyidyi;
            at[i][j][k] += visc * ( cg0*(cg0*a[i][j][k-3] + cg1*a[i][j][k-2] + cg2*a[i][j][k-1] + cg3*as[ijks  ]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*as[ijks  ] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*as[ijks  ] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(tg0*a[i][j][k-1] + tg1*as[ijks  ] + tg2*a[i][j][k+1] + tg3*a[i][j][k+2]) * dzhi4[k+2] ) * dzi4[k];
        }
        // interior
        else
        {
            at[i][j][k] += visc * (cdg3*as[ijks-ii3 ] + cdg2*as[ijks-ii2 ] + cdg1*as[ijks-ii1 ] + cdg0*as[ijks] + cdg1*as[ijks+ii1 ] + cdg2*as[ijks+ii2 ] + cdg3*as[ijks+ii3 ])*dxidxi;
            at[i][j][k] += visc * (cdg3*as[ijks-jjs3] + cdg2*as[ijks-jjs2] + cdg1*as[ijks-jjs1] + cdg0*as[ijks] + cdg1*as[ijks+jjs1] + cdg2*as[ijks+jjs2] + cdg3*as[ijks+jjs3])*dyidyi;
            at[i][j][k] += visc * ( cg0*(cg0*a[i][j][k-3] + cg1*a[i][j][k-2] + cg2*a[i][j][k-1] + cg3*as[ijks  ]) * dzhi4[k-1]
                    + cg1*(cg0*a[i][j][k-2] + cg1*a[i][j][k-1] + cg2*as[ijks  ] + cg3*a[i][j][k+1]) * dzhi4[k  ]
                    + cg2*(cg0*a[i][j][k-1] + cg1*as[ijks  ] + cg2*a[i][j][k+1] + cg3*a[i][j][k+2]) * dzhi4[k+1]
                    + cg3*(cg0*as[ijks  ] + cg1*a[i][j][k+1] + cg2*a[i][j][k+2] + cg3*a[i][j][k+3]) * dzhi4[k+2] ) * dzi4[k];
        }
    }
                }
