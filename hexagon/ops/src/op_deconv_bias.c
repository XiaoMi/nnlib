
/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


/*
 *
 * This contains implementations for deconvolution with bias OP
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif

//#define DEBUG_PRINT

//#define MEASURE_PERF
#ifdef MEASURE_PERF
#include "HAP_perf.h"
#endif

// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_DECONVBIAS_INPUT_DATA_IDX 0
#define OP_DECONVBIAS_FILT_DATA_IDX 1
#define OP_DECONVBIAS_FILT_DIM_IDX 2
#define OP_DECONVBIAS_BIAS_IDX 3
#define OP_DECONVBIAS_PAD_IDX 4
#define OP_DECONVBIAS_STRIDE_IDX 5
#define OP_DECONVBIAS_DILATION_IDX 6
#define OP_DECONVBIAS_GROUPS_IDX 7
#define OP_DECONVBIAS_INPUT_MIN_IDX 8
#define OP_DECONVBIAS_INPUT_MAX_IDX 9
#define OP_DECONVBIAS_FILT_MIN_IDX 10
#define OP_DECONVBIAS_FILT_MAX_IDX 11
#define OP_DECONVBIAS_BIAS_MIN_IDX 12
#define OP_DECONVBIAS_BIAS_MAX_IDX 13
#define OP_DECONVBIAS_OUT_USE_STATIC_MINMAX_IDX 14
#define OP_DECONVBIAS_OUT_STATIC_MIN_IDX 15
#define OP_DECONVBIAS_OUT_STATIC_MAX_IDX 16
#define OP_DECONVBIAS_NUM_OPS 17



#define ROUNDUP(X, SIZE) (((X) + (SIZE) - 1) & (~((SIZE)-1)))
#define ROUNDDOWN(X, SIZE) (((X)) & (~((SIZE)-1)))

#define ALIGN_SIZE 128

#define PADDED_SIZE(size,pad) ((size + pad - 1) & ~(pad - 1))

// These are requirements imposed by the hvx gemm functions
// These need to be kept in-sync with the CPU side as the
// CPU side transpacks the filter weights before sending
// them to us
#define KPAD 32
#define NPAD 32
#define MPAD 8
// The output depth pad helps with hvx'ed bias add and
// col2im. This padding also need to be applied in the
// transposed kernel to the K_over_g dimension
#define ODPAD 1 // disabled


// For ref impl, controls the maximum col2im buff size
// so that it can fit in the scratch mem
#define REF_MAX_M_BY_N_SIZE 8*1024*1024

// For hvx impl, controls the size of the col2im buff
// that we work on at a time
// If we cannot fit even a single row into this size
// we double it until we hit the absolute max
#define HVX_M_BY_N_SIZE 64*1024
#define HVX_MAX_M_BY_N_SIZE 8*1024*1024


#if defined(__hexagon__)
static int32_t max(int32_t a, int32_t b) {return((a>b)?a:b);}
static int32_t min(int32_t a, int32_t b) {return((a<b)?a:b);}
#endif


// Load/store unaligned HVX_Vector
#if defined(__hexagon__)
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif

// struct to communicate with nn_os_work_for_vector
struct tdata {
    struct nn_node *self;
    nn_sem_t donesem;
    int res;
};

/*
 * HVX unaligned accumulate for <128b (<32 ints) vectors
 * Takes n ints at y_s, accumulates them to n ints at x_s
 * x_s: int32_t*
 * y_s: int32_t*
 * n: number of ints to accumulate, 0 < n < 32
 *
 * Description:
 *   1. Unaligned 128b load at y_s (bytes past n*4 will not be used)
 *   2. Rotate left by x_s%128 so that first useful byte of y_s aligns
 *      with first useful byte of x_s
 *   3. Case 1: x does not cross 128b boundary, only 1 store is needed
 *              Load aligned x_s
 *              Predicated accumulate with rotated y
 *              Store aligned x_s
 *      Case 2: x crosses 128b boundary, 2 stores needed
 *              Load aligned x_s and x_s+128b
 *              Predicated accumulates with rotated y
 *              Store aligned x_s and x_s+128b
 */

#define HVX_UNALIGNED_ACC(x_s, y_s, n)                                              \
    int offset = (uintptr_t)x_s & (ALIGN_SIZE-1);                                   \
    int nbytes = offset + (n)*sizeof(int32_t);                                      \
    HVX_Vector* xw = (HVX_Vector*) ((uintptr_t)x_s & ~(ALIGN_SIZE-1));              \
    HVX_Vector yw = vmemu((unsigned char*)y_s);                                     \
    yw = Q6_V_vror_VR(yw, -offset);                                                 \
    HVX_VectorPred q0 = Q6_Q_vsetq_R(offset);                                       \
    HVX_VectorPred q1 = Q6_Q_vsetq_R(nbytes);                                       \
    if(nbytes < 128) {                                                              \
        HVX_VectorPred notq0_and_q1 = Q6_Q_and_QQn(q1, q0);                         \
        *xw = Q6_Vw_condacc_QVwVw(notq0_and_q1, *xw, yw);                           \
    } else {                                                                        \
        *xw = Q6_Vw_condacc_QnVwVw(q0, *xw, yw);                                    \
        if (nbytes > 128)                                                           \
            *(xw + 1) = Q6_Vw_condacc_QVwVw(q1, *(xw + 1), yw);                     \
    }                                                                               \


static int32_t compute_out_size(int32_t in_size, int32_t kernel, int32_t stride, int32_t pad, int32_t dilation) {
    return stride*(in_size-1) + dilation*(kernel-1)+1 - 2*pad;
}

static int32_t compute_fill_bias_range(
    int32_t m, 
    int32_t Wi, int32_t Wo, 
    int32_t stride_x, int32_t stride_y,
    int32_t filt_x, int32_t filt_y,
    int32_t dilation_x, int32_t dilation_y,
    int32_t pad_x, int32_t pad_y)
{
    int32_t y = m/Wi;
    int32_t x = m - y*Wi;

    int32_t xo = min(Wo, max(0, dilation_x*(filt_x-1) + x*stride_x - pad_x));
    int32_t yo = dilation_y*(filt_y-1) + y*stride_y - pad_y;

    return  (yo*Wo + xo);
}

static int32_t compute_out_range(
    int32_t m, 
    int32_t Wi, int32_t Wo, 
    int32_t stride_x, int32_t stride_y,
    int32_t filt_x, int32_t filt_y,
    int32_t dilation_x, int32_t dilation_y,
    int32_t pad_x, int32_t pad_y)
{
    int32_t y = m/Wi;
    int32_t x = m - y*Wi;

    int32_t xo = min(Wo, max(0, x*stride_x - pad_x));
    int32_t yo = max(0, y*stride_y - pad_y);

    return  (yo*Wo + xo -1);
}

static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
    uintptr_t ptrval = (uintptr_t)(ptr);
    ptrval += minsize + (ALIGN_SIZE-1);
    ptrval &= ~(ALIGN_SIZE-1);
    return (void *)ptrval;
}

/**
 * Maps point (k, m) in a matrix of size KxM
 * to the target point (y, x) in transpacked matrix of size M/32xK*32
 */
static void transmap(size_t k, size_t m, size_t* y, size_t* x) {
    *y = m/32;
    int block = k/4;
    int block_x = m%32;
    int block_y = k%4;
    *x = block*128 + block_x*4 + block_y;
}

/**
 * Reference gemm implementation
 * Expects B matrix to be transpacked
 */
static void gemm_ref(uint8_t* A, uint8_t* B,
                     int32_t A_offset, int32_t B_offset,
                     size_t M, size_t N, size_t K,
                     size_t Kp,
                     int* C) {
    size_t m, n, k;
    int32_t sum;
    size_t a_i, b_i, c_i;
    size_t b_y, b_x;
    int32_t a_data, b_data;
    for(m=0; m<M; m++) {
        for(n=0; n<N; n++) {
            sum = 0;
            for(k=0; k<K; k++) {
                a_i = m*K + k;
                transmap(k, n, &b_y, &b_x);
                b_i = b_y*Kp*32 + b_x;
                a_data = A[a_i] - A_offset;
                b_data = B[b_i] - B_offset;
                sum += a_data*b_data;
            }
            c_i = m*N + n;
            C[c_i] = sum;
        }
    }
}

#if 0
static void gemsuma_cn(uint8_t * x, int N, int K, int *xsum, int y_offset, int z_offset) {
    int i,k, a_val, sum;
    for (i=0; i < N; i++) {
        sum = 0;
        for (k=0; k < K; k++) {
            a_val = x[i*K+k];
            sum += a_val;
        }
        xsum[i] = (sum*y_offset) + z_offset;
    }
}

// M must be multiple of 32
static void gemsumb_cn(uint8_t * y, int *z, int M, int K, int x_offset) {
    int m;
    for(m=0; m<M; m+=32) {

        int j, k, a_val, b_val = x_offset;
        for (j = 0; j < 32; j += 1) {
            z[j] = 0;
            for (k = 0; k < K; k += 4) {
                a_val = y[32 * k + 0 + 4 * j];
                z[j] += a_val * b_val;

                a_val = y[32 * k + 1 + 4 * j];
                z[j] += a_val * b_val;

                a_val = y[32 * k + 2 + 4 * j];
                z[j] += a_val * b_val;

                a_val = y[32 * k + 3 + 4 * j];
                z[j] += a_val * b_val;
            }
        }
        y += 32*K;
        z += 32;
    }
}

static void gemmpybbw_cn(uint8_t * x, uint8_t * y, int * z, int N, int M, int K) {
    int i,j,k, a_val, b_val;
    int m;
    for(m=0; m<M; m+=32) {

        for (i = 0; i < N; i++) {
            for (j = 0; j < 32; j += 1) {
                z[i * M + j] = 0;
                for (k = 0; k < K; k += 4) {
                    a_val = x[i * K + k];
                    b_val = y[32 * k + 0 + 4 * j];
                    z[i * M + j] += a_val * b_val;

                    a_val = x[i * K + k + 1];
                    b_val = y[32 * k + 1 + 4 * j];
                    z[i * M + j] += a_val * b_val;

                    a_val = x[i * K + k + 2];
                    b_val = y[32 * k + 2 + 4 * j];
                    z[i * M + j] += a_val * b_val;

                    a_val = x[i * K + k + 3];
                    b_val = y[32 * k + 3 + 4 * j];
                    z[i * M + j] += a_val * b_val;
                }
            }
        }
        y += 32*K;
        z += 32;
    }
}

static void gemaddvvm_cn(int * xsum, int * ysum, int * z, int N, int M)
{
    int m;
    for(m=0; m<M; m+=32) {

        int i, j, sum;
        for (i = 0; i < N; i++) {
            for (j = 0; j < 32; j++) {
                sum = xsum[i] + ysum[j] + z[i * M + j];
                z[i * M + j] = sum;
            }
        }

        ysum += 32;
        z += 32;
    }
}
#endif


/**
 * GEMM
 * NOTE: meanings of M and N are switched.
 * @param x Matrix A
 * @param x_offset Offset of values in matrix A
 * @param yopt Matrix B, transpacked
 * @param y_offset Offset of values in matrix B
 * @param z Output matrix of size N*M*sizeof(int)
 * @param N Rows in matrix A
 * @param M Cols in matrix B
 * @param K Cols in matrix A and rows in matrix B
 * @param xsum scratch memory of size N*sizeof(int)
 * @param ysum scratch memory of size M*sizeof(int)
 * @param minmax scratch memory of size 64*sizeof(int)
 */
static void gemm_asm_nstep(uint8_t* x, int x_offset,
                           uint8_t* yopt, int y_offset,
                           int32_t* ysum,
                           int32_t* z, int N, int M, int K,
                           int32_t* xsum)
{
    gemsuma_asm(x, N, Q6_R_combine_RlRl(K,K), (int *)xsum, y_offset, K*x_offset*y_offset);
//    gemsuma_cn(x, N, K, xsum, y_offset, K*x_offset*y_offset);

    // These functions only support max of M=32
    // So we do them in a loop of chunks
    int m;
    for(m=0; m<M; m+=32) {
//      gemsumb_asm(&yopt[m*K], &ysum[m], K, x_offset);

        gemmpybbw_asm(x,  &yopt[m*K],  (int *)&z[m], N, M, Q6_R_combine_RlRl(K,K));

        gemaddvvm_asm1((int *)xsum, (int *)&ysum[m], (int *)&z[m], N, M);
    }

//    gemsumb_cn(yopt, ysum, M, K, x_offset);
//    gemmpybbw_cn(x, yopt, z, N, M, K);
//    gemaddvvm_cn(xsum, ysum, z, N, M);

    return;
}


/**
 * Reference col2im
 * @param col_data Columns data
 * @param m_start Start M for chunked input
 * @param m_end End M for chunked input, not inclusive
 * @param H input height
 * @param W input width
 * @param filt_batches filt batches (aka K_over_g)
 * @param R filter height
 * @param S filter width
 * @param filt_batches_padded padded size of filt_batches (aka K_over_g)
 *                            only col2im buff needs to be padded
 * @param pad_y
 * @param pad_x
 * @param stride_y
 * @param stride_x
 * @param dilation_y
 * @param dilation_x
 */
static void col2im_ref(int* col_data,
                       size_t m_start, size_t m_end,
                       size_t H, size_t W, size_t filt_batches, size_t R, size_t S,
                       size_t filt_batches_padded,
                       size_t pad_y, size_t pad_x,
                       size_t stride_y, size_t stride_x,
                       size_t dilation_y, size_t dilation_x,
                       int32_t* im_data) {
    size_t P = compute_out_size(H, R, stride_y, pad_y, dilation_y);
    size_t Q = compute_out_size(W, S, stride_x, pad_x, dilation_x);

    // the col dimensions are H*W x R*S*K
    // there are HxW patches (H patches down, W patches across)
    //size_t num_col_rows = H*W;
    size_t num_col_cols = R*S*filt_batches_padded;

    size_t h, w, r, s, k;
    int32_t col_row, col_col, col_index;
    int32_t im_y, im_x, im_index;
    int32_t patch_pixel_num;
    for(h=0; h<H; h++) {
        for(w=0; w<W; w++) {
            // col row gives us the patch number
            col_row = h*W + w;

            // For chunked input, skip if this is not part of the current input
            if(col_row < m_start || col_row >= m_end) {
                continue;
            }
            col_row = col_row - m_start;

            // each patch has R*S*K pixels (R*S pixels for each of the packed K channels)
            // for each patch loop over the patch pixels
            for(r=0; r<R; r++) {
                // map the patch pixels to im row
                im_y = dilation_y*r + h*stride_y - pad_y;
                if(im_y < 0 || im_y >= P) {
                    // maps to outside of im
                    continue;
                }
                for(s=0; s<S; s++) {
                    patch_pixel_num = r*S + s;
                    col_col = filt_batches_padded*patch_pixel_num;
                    // map the patch pixels to im col
                    im_x = dilation_x*s + w*stride_x - pad_x;
                    if(im_x < 0 || im_x >= Q) {
                        // maps to outside of im
                        continue;
                    }
                    for(k=0; k<filt_batches; k++) {
                        // copy pixels for all K channels
                        col_index = col_row * num_col_cols + col_col + k;
                        im_index = im_y * Q * filt_batches + im_x * filt_batches + k;
                        im_data[im_index] += col_data[col_index];
                    }
                }
            }
        }
    }
}


/**
 * Col2im
 * @param col_data Columns data
 * @param m_start Start M for chunked input
 * @param m_end End M for chunked input, not inclusive
 * @param H input height
 * @param W input width
 * @param filt_batches filt batches (aka K_over_g)
 * @param R filter height
 * @param S filter width
 * @param ldN padded width of matrix
 * @param filt_batches_padded padded size of filt_batches (aka K_over_g)
 *                            both col2im and im_data buffers need to be padded
 * @param pad_y
 * @param pad_x
 * @param stride_y
 * @param stride_x
 * @param dilation_y
 * @param dilation_x
 */
static void col2im(int32_t* col_data,
                   size_t m_start, size_t m_end,
                   size_t H, size_t W, size_t filt_batches, size_t R, size_t S,
                   size_t ldN, size_t filt_batches_padded,
                   size_t pad_y, size_t pad_x,
                   size_t stride_y, size_t stride_x,
                   size_t dilation_y, size_t dilation_x,
                   int32_t* im_data, 
                   size_t im_Hmin, size_t im_Hmax, 
                   size_t im_Wmin, size_t im_Wmax) {
    size_t Q = compute_out_size(W, S, stride_x, pad_x, dilation_x);

    // the col dimensions are H*W x R*S*K
    // there are HxW patches (H patches down, W patches across)
    //size_t num_col_rows = H*W;
    //size_t num_col_cols = R*S*K;
    // matrix padded width is ldN, so we offset by this amount for cols
    size_t num_col_cols = ldN;

    size_t h, w, r, s, k;
    int32_t col_row, col_col, col_index;
    int32_t im_y, im_x, im_index, xmin, xmax;
    int32_t patch_pixel_num;
    // Loop over the M-dim (size HxW), where each row is a patch
    for(col_row=0; col_row<(m_end-m_start); col_row++) {
        // col_row goes from 0 to last row in col2im buffer
        // the actual col row index (i.e. patch number) is given by
        // m_start+col_row
        size_t patch_id = m_start + col_row;
        // We map the patch id to the source h,w pixel
        h = patch_id/W;
        w = patch_id%W;

        // each patch has R*S*K pixels (R*S pixels for each of the packed K channels)
        // for each patch loop over the patch pixels
        for(r=0; r<R; r++) {
            // map the patch pixels to im row
            im_y = dilation_y*r + h*stride_y - pad_y;
            if(im_y < im_Hmin || im_y > im_Hmax) {
                // maps to outside of im
                continue;
            }

            xmin = (im_y==im_Hmin) ? im_Wmin : 0;
            xmax = (im_y==im_Hmax) ? im_Wmax : Q-1;

            for(s=0; s<S; s++) {
                patch_pixel_num = r*S + s;
                // use padded filt batches
                col_col = filt_batches_padded*patch_pixel_num;
                // map the patch pixels to im col
                im_x = dilation_x*s + w*stride_x - pad_x;
                if(im_x < xmin || im_x > xmax) {
                    // maps to outside of im
                    continue;
                }
                col_index = col_row * num_col_cols + col_col;
                im_index = im_y * Q * filt_batches_padded + im_x * filt_batches_padded;

                // copy-add pixels for all K channels
                // We can only do hvx if filt_batches_padded is multiple of 32
                if(filt_batches_padded % 32 == 0) {
                    for (k = 0; k < filt_batches_padded; k += 32) {
                        HVX_Vector *xw = (HVX_Vector * ) & im_data[im_index + k];
                        HVX_Vector *yw = (HVX_Vector * ) & col_data[col_index + k];
                        HVX_Vector *zw = (HVX_Vector * ) & im_data[im_index + k];
                        *zw =  Q6_Vw_vadd_VwVw(*xw, *yw);
                    }
                } else {
                    // Non-hvxed adds
                    for (k = 0; k < filt_batches_padded; k++) {
                        im_data[im_index + k] += col_data[col_index + k];
                    }
                }
            }
        }
    }
}


/**
 * Col2im with no dilation
 * Does not support padded filt batches
 * @param col_data Columns data
 * @param m_start Start M for chunked input
 * @param m_end End M for chunked input, not inclusive
 * @param H input height
 * @param W input width
 * @param filt_batches filt batches (aka K_over_g)
 * @param R filter height
 * @param S filter width
 * @param ldN padded width of matrix
 * @param pad_y
 * @param pad_x
 * @param stride_y
 * @param stride_x
 */
static void col2im_nodil(int32_t* col_data,
                         size_t m_start, size_t m_end,
                         size_t H, size_t W, size_t filt_batches, size_t R, size_t S,
                         size_t ldN,
                         size_t pad_y, size_t pad_x,
                         size_t stride_y, size_t stride_x,
                         int32_t* im_data,
                         size_t Q,
                         size_t im_Hmin, size_t im_Hmax,
                         size_t im_Wmin, size_t im_Wmax) {

    // the col dimensions are H*W x R*S*K
    // there are HxW patches (H patches down, W patches across)
    //size_t num_col_rows = H*W;
    //size_t num_col_cols = R*S*K;
    // matrix padded width is ldN, so we offset by this amount for cols
    size_t num_col_cols = ldN;

    size_t h, w, r;
    size_t i;
    int32_t col_row, col_col_start, col_index_start;
    int32_t im_y, im_index_start, xmin, xend;
    int32_t im_x_start_ub, im_x_end_ub, im_x_start, im_x_end;
    int32_t patch_pixel_num_start;
    int32_t pixel_count, pixel_offset;

    // We map the initial patch id to the source h,w pixel
    h = m_start/W;
    w = m_start - h*W;

    // Loop over the M-dim (size HxW), where each row is a patch
    for(col_row=0; col_row<(m_end-m_start); col_row++) {
        // col_row goes from 0 to last row in col2im buffer
        // the actual col row index (i.e. patch number) is given by
        // m_start+col_row
        //size_t patch_id = m_start + col_row;
        // We map the patch id to the source h,w pixel
        //h = patch_id/W;
        //w = patch_id%W;

        // each patch has R*S*K pixels (R*S pixels for each of the packed K channels)
        // for each patch loop over the patch pixels
        for(r=0; r<R; r++) {
            // map the patch pixels to im row
            im_y = r + h*stride_y - pad_y;
            if(im_y < im_Hmin || im_y > im_Hmax) {
                // maps to outside of im
                continue;
            }

            xmin = (im_y==im_Hmin) ? im_Wmin   : 0;
            xend = (im_y==im_Hmax) ? im_Wmax+1 : Q;

            // Since there's no dilation, entire row of filter pixels will be contiguous in the output image
            // Get unbounded start and end pixels in output image
            im_x_start_ub = w*stride_x - pad_x;
            im_x_end_ub = im_x_start_ub + S;
            // Get the bounded start and end pixels
            im_x_start = max(xmin, im_x_start_ub);
            im_x_end   = min(xend, im_x_end_ub);

            if(im_x_start >= im_x_end || im_x_start >= Q) {
                // patch row is entirely outside the output image
                continue;
            }
            pixel_count = im_x_end-im_x_start;
            // These are the left-sided out-of-bound pixels that we ignored
            pixel_offset = im_x_start - im_x_start_ub;

            patch_pixel_num_start = r*S + pixel_offset;
            col_col_start = filt_batches*patch_pixel_num_start;
            col_index_start = col_row * num_col_cols + col_col_start;
            im_index_start = (im_y * Q + im_x_start) * filt_batches;

            int n = pixel_count * filt_batches; 

            // Copy-add pixel_count*filt_batches ints
            if((filt_batches&31) == 0) {
                // Yaaay, pixels are aligned, HVX'ing is easy peazy
                for (i = 0; i < n; i += 32) {
                    HVX_Vector *xw = (HVX_Vector * ) & im_data[im_index_start + i];
                    HVX_Vector *yw = (HVX_Vector * ) & col_data[col_index_start + i];
                    HVX_Vector *zw = (HVX_Vector * ) & im_data[im_index_start + i];
                    *zw = Q6_Vw_vadd_VwVw(*xw, *yw);
                }
            } else {
                // Not aligned, but there's enough to do a full HVX accumulate
                // Do as many unaligned full HVX accumulates as we can
                for(i = 0; i < ROUNDDOWN(n,32); i+=32) {
                    HVX_Vector xw = vmemu((unsigned char*)(im_data+im_index_start+i));
                    HVX_Vector yw = vmemu((unsigned char*)(col_data+col_index_start+i));
                    vmemu((unsigned char*)(im_data+im_index_start+i)) = Q6_Vw_vadd_VwVw(xw, yw);
                }
                // Do the remaining as a sub-128b hvx accumulate
                if(n&31) {
                    int32_t* x_s = &im_data[im_index_start+i];
                    int32_t* y_s = &col_data[col_index_start+i];
                    HVX_UNALIGNED_ACC(x_s, y_s, n&31);
                }
            }
        }

        if (++w == W) { w = 0;  h++;}
    }
}

/**
 * Adds bias data to out. Bias data is broadcasted to all rows
 * Bias vector:
 *        [                      ]
 *            <--- depth --->
 * Out matrix:
 *        [                      ]
 *     ^  [                      ]
 *   rows [                      ]
 *    V   [                      ]
 *        [                      ]
 *            <--- depth --->
 * @param bias_data
 * @param bias_offset
 * @param bias_mpy_amt
 * @param out
 * @param depth
 * @param rows
 */
static void bias_add_ref(uint8_t* bias_data, int32_t bias_offset, float bias_mpy_amt,
                         int32_t* out,
                         size_t depth, size_t rows) {
    size_t row, k;
    for(row=0; row<rows; row++) {
        for(k=0; k<depth; k++) {
            int32_t bias_val = ((int32_t)bias_data[k]-bias_offset)*bias_mpy_amt;
            out[row*depth+k] += bias_val;
        }
    }
}

static void bias_set(uint8_t* bias_data, int32_t bias_offset, float bias_mpy_amt,
                     size_t depth, 
                     int32_t* bias_buf )
{
    size_t k;
    for(k=0; k<depth; k++) {
        bias_buf[k] = ((int32_t)bias_data[k]-bias_offset)*bias_mpy_amt;
    }

    if (depth < 32) {
        // Padded to 32-elements
        int phase = 0;

        for(k=depth; k<32; k++)
        { 
            bias_buf[k] = bias_buf[phase];
            phase++;
            if (phase == depth) phase = 0;
        }
    }
}

static void fill_bias(int32_t* bias, int32_t* out, int32_t depth, int32_t rows, int32_t phase) 
{
    int k, d;
    HVX_Vector *pout = (HVX_Vector *)out;

    // if depth = 2^k
    if (depth== (1<<Q6_R_ct0_R(depth))) {
        if (depth<32) {
            rows = (rows*depth + 31)>>5;
            depth = 32;
        }

        for(k=0; k<rows; k++) {
            HVX_Vector *pbias = (HVX_Vector *)bias;
            for (d=0; d<depth; d+=32) {
                *pout++ = *pbias++;
            }
        }
    } else if(depth < 32) {
        phase <<= 2;
        HVX_Vector vbias = *(HVX_Vector *)bias;
        HVX_VectorPred mask = Q6_Q_vsetq_R(128-phase);

        rows = (rows*depth + 31)>>5;
        *pout++ = vbias;

        for(k=1; k<rows; k++) {
            HVX_Vector vbias_0 = Q6_V_vror_VR(vbias,phase);
            HVX_Vector vbias_1 = Q6_V_vror_VR(vbias,2*phase);
            vbias = Q6_V_vmux_QVV(mask,vbias_0, vbias_1);
            *pout++ = vbias;
        }
    } else {
        for(k=0; k<rows; k++) {
            vmemcpy_asm(out+k*depth, bias, depth*sizeof(int32_t));
        }
    }
}

#if 0
static void unpad(int32_t* out_buf, size_t rows, size_t out_depth, size_t out_depth_padded,
                  int32_t* out_buf_nopad) {
    size_t row;
    for(row=0; row<rows; row++) {
        vmemcpy_asm(out_buf_nopad+row*out_depth, out_buf+row*out_depth_padded, out_depth*sizeof(int32_t));
    }
}
#endif

static void find_minmax_hvx(int32_t *pin, HVX_Vector *pminmax, int32_t elements)
{    
    HVX_Vector *in = (HVX_Vector *)pin;

    HVX_Vector vmax = pminmax[0];
    HVX_Vector vmin = pminmax[1];
    HVX_Vector xin;
    int i;

    for (i = 0; i < (elements>>5); i++) {
        xin = *in++;
        vmax = Q6_Vw_vmax_VwVw(vmax,xin);
        vmin = Q6_Vw_vmin_VwVw(vmin,xin);
    } 

    if(elements&31) {
        xin = *in;
        HVX_VectorPred mask = Q6_Q_vsetq_R((elements&31)*4);
        vmax = Q6_Vw_vmax_VwVw(vmax,Q6_V_vmux_QVV(mask,xin,vmax));
        vmin = Q6_Vw_vmin_VwVw(vmin,Q6_V_vmux_QVV(mask,xin,vmin));
    }

    pminmax[0] = vmax;
    pminmax[1] = vmin;
}


#define NUM_THREADS 2

#define BLOCK_SIZE     (16*1024)  

struct deconvnode_info {
    uint8_t*    input;
    uint8_t*    filt;
    uint8_t*    output;
    int32_t*    bias_buf;
    int32_t*    sumb;
    int32_t*    out_buf;
    // buffer size
    size_t      input_scr_size;
    size_t      col2im_buf_size;
    size_t      suma_size;
    size_t      sumb_size;
    // parameters
    size_t      in_width; 
    size_t      in_height; 
    size_t      in_depth; 
    size_t      filt_width;
    size_t      filt_height;
    size_t      filt_batches;
    size_t      stride_width;
    size_t      stride_height; 
    size_t      dilation_width;
    size_t      dilation_height;
    size_t      pad_height; 
    size_t      pad_width;
    size_t      out_width; 
    size_t      out_height; 
    size_t      out_depth; 
    size_t      out_batches;
    size_t      in_offset;
    size_t      filt_offset;
    int         Np;
    int         Kp;
    int         Mc;
    // control parameters
    int         need_compute_minmax;
};

struct tdata_deconv {
	struct deconvnode_info *info;	// same as self->opaque
    nn_sem_t    *donesem;
    int32_t     *in;
    uint8_t     *out;
    uint8_t     *input_scr;
    int32_t     *col2im_buf;
    int32_t     *suma;
    int32_t     *minmax;
    int32_t     *out_buf;
    size_t      start;
    size_t      end;
    int32_t     elements;
    int         gain;
    int         offset;
};

static int workitem_execute_sumb(struct nn_graph *nn, void *cinfo) 
{
    struct deconvnode_info *info = cinfo;
    int N = info->Np;
    int K = info->Kp;
    int x_offset = -info->in_offset;
    uint8_t *filt = info->filt;
    int32_t *sumb = info->sumb;

    int m;
    for(m=0; m<N; m+=32) {
        gemsumb_asm(&filt[m*K], (int *)&sumb[m], K, x_offset);
    }
    return 0;
}

static void sumb_set_hvx(struct nn_graph *nn, struct deconvnode_info *cinfo) 
{
    nn_os_vector_call(nn, workitem_execute_sumb, (void *)cinfo);
}

static void workitem_execute_quantize(struct nn_graph *nn, void *info)
{
    struct tdata_deconv *work = info;
    int32_t *in = work->in;
    uint8_t *out = work->out;
    int32_t elements = work->elements;
    int n;

    int block = Q6_R_min_RR(elements, BLOCK_SIZE/sizeof(int32_t));
    l2fetch(in, 128, 128, block>>5);

    for (n = 0; n < elements; n += BLOCK_SIZE/sizeof(int32_t)) {
        int next_block = Q6_R_min_RR(elements-n-block, BLOCK_SIZE/sizeof(int32_t));
        wait_for_l2fetch();
        if (next_block > 0) l2fetch(&in[block], 128, 128, next_block>>5);
        quantize_asm(&in[n], work->offset, work->gain, &out[n], block);
        block = next_block;
    }
    nn_sem_post(work->donesem);
}
static void workitem_execute_copy(struct nn_graph *nn, void *info)
{
    struct tdata_deconv *work = info;
    uint8_t *in = (uint8_t*) work->in;
    uint8_t *out = work->out;
    int32_t elements = work->elements;
    vmemcpy_asm(out,in,elements);
    nn_sem_post(work->donesem);

}
static void quantize_int32_to_u8(
    struct nn_graph *nn, 
    int32_t *in,
    int     elements,
    int     offseti,
    int     gaini,
    uint8_t *out)
{
    struct tdata_deconv work[NUM_THREADS];
    nn_sem_t donesem;
    nn_sem_init(&donesem,0);

    int t_elements = ((elements + NUM_THREADS-1)/NUM_THREADS + 127)&~127;
    int n_threads = (elements + t_elements -1)/t_elements;
    int i, start;
    int32_t is_output_aligned = (((int32_t)out) & 127) == 0;
    uint8_t* output_quantize = (is_output_aligned) ? out : (uint8_t*)nn->scratch; //scratch size verified already in deconv_bias_execute_hvx
    // setup parameters for multi-thread 
    for (i=0; i < n_threads; i++) {
        start = i * t_elements;
        work[i].in  = in  + start;
        work[i].out = output_quantize + start;
        work[i].elements = Q6_R_min_RR(elements-start, t_elements);
        work[i].offset = offseti;
        work[i].gain = gaini;
        work[i].donesem = &donesem;
        nn_os_work_for_vector(nn, workitem_execute_quantize, &work[i]);
    }
    for (i=0; i < n_threads; i++) { nn_sem_wait(&donesem); }
    if (!is_output_aligned){//output unaligned copy over from scratch
        work[0].in = (int32_t *)output_quantize;
        work[0].out = out;
        work[0].elements = elements;
        nn_os_work_for_vector(nn, workitem_execute_copy, work);
        nn_sem_wait(&donesem);
    }
    return;
}

static void workitem_execute_deconv(struct nn_graph *nn, void *tinfo) 
{
    struct tdata_deconv *work = tinfo;
    struct deconvnode_info *info = work->info;

    // common parameters
    int32_t in_width = info->in_width;
    int32_t in_height = info->in_height;
    int32_t in_depth = info->in_depth;
    int32_t filt_height = info->filt_height;
    int32_t filt_width = info->filt_width;
    int32_t filt_batches = info->filt_batches;
    int32_t pad_width  = info->pad_width;
    int32_t pad_height = info->pad_height;
    int32_t stride_width  = info->stride_width;
    int32_t stride_height = info->stride_height;
    int32_t dilation_width = info->dilation_width;
    int32_t dilation_height = info->dilation_height;
    int32_t out_width = info->out_width;
    int32_t out_depth = info->out_depth;
    //size_t M = in_height * in_width;
    size_t K = in_depth;
    size_t Np = info->Np;
    size_t Kp = info->Kp;
    size_t  Mc = info->Mc;
    size_t col2im_buf_size = info->col2im_buf_size;
    int32_t in_offset = info->in_offset;
    int32_t filt_offset = info->filt_offset;
    int need_compute_minmax = info->need_compute_minmax;
    size_t filt_batches_padded = ROUNDUP(filt_batches, ODPAD);
    int32_t phase = 32%out_depth;
    int32_t n2k = 32>>(min(5,Q6_R_ct0_R(out_depth)));
    // pointers
    uint8_t *input = info->input;
    uint8_t *filt = info->filt;
    int32_t *bias_buf = info->bias_buf;
    int32_t *out_buf_nopad = info->out_buf;
    int32_t *sumb = info->sumb;
    uint8_t *input_scr = work->input_scr;
    int32_t *col2im_buf = work->col2im_buf;
    int32_t *suma = work->suma;
    HVX_Vector *minmax = (HVX_Vector *)work->minmax;
    // MT parameters
    int32_t out_start = work->start;
    int32_t out_end   = work->end;

    int32_t outHmin =  out_start /out_width;
    int32_t outHmax = (out_end-1)/out_width;
    int32_t outWmin =  out_start  - outHmin*out_width;
    int32_t outWmax = (out_end-1) - outHmax*out_width;
    int32_t inHmin = max((outHmin + pad_height - dilation_height*(filt_height-1))/stride_height,           0);
    int32_t inHmax = min((outHmax + pad_height - dilation_height*0              )/stride_height+1, in_height);
    int32_t start = (inHmin*in_width)&~(MPAD-1); 
    int32_t end   = inHmax*in_width;

    // prefetch initial inputs
    if (end > start) {
        l2fetch(input+start*K, K, K, min(end-start, Mc));
    }

    minmax[0] = Q6_V_vsplat_R(0x80000000);
    minmax[1] = Q6_V_vsplat_R(0x7FFFFFFF);

    int32_t b_start = out_start;
    int32_t b_end   = out_start;
    int32_t o_start = out_start;
    int32_t o_end   = out_start;

    int32_t m_start, m_end, m_end_next;
    for(m_start=start; m_start<end; m_start+=Mc) {
        m_end = min(m_start+Mc, end);

        // Copy input to scratch mem with padding
        int32_t in_row, scr_row;
        for (in_row=m_start, scr_row=0; in_row<m_end; in_row++, scr_row++) {
            vmemcpy_asm(input_scr + scr_row*Kp, input + in_row*K, K);
            if (Kp - K > 0) memset(input_scr + scr_row*Kp + K, in_offset, Kp-K);
        }
        // Note: don't need to pad in the M dim because we won't use those rows
        //       in the output matrix

        // Zero-out the col2im buff
        vmemset_asm(col2im_buf, 0, col2im_buf_size);

        gemm_asm_nstep(input_scr, -in_offset,
                       filt, -filt_offset,
                       sumb, 
                       col2im_buf,
                       ROUNDUP(m_end-m_start, MPAD), Np, Kp,
                       suma);

        // prefetch next inputs
        m_end_next = min(m_end+Mc, end);
        if (m_end_next > m_end) {
            l2fetch(input+m_end*K, K, K, m_end_next-m_end);
        }

        // Col2im
        b_end = compute_fill_bias_range(
                    m_end, in_width, out_width, 
                    stride_width, stride_height,
                    filt_width, filt_height,
                    dilation_width, dilation_height,
                    pad_width, pad_height);

        b_end = min(ROUNDUP(b_end,n2k), out_end);

        // Since output buff is used to accumulate, we can
        // fill it with bias values for a free bias add
        if (b_end > b_start) {
            fill_bias(bias_buf, out_buf_nopad+b_start*out_depth, out_depth, b_end-b_start, phase);
            b_start = b_end;
        } 

        if(dilation_height == 1 && dilation_width == 1) {
            col2im_nodil(
                    col2im_buf,
                    m_start, m_end,
                    in_height, in_width, filt_batches, filt_height, filt_width,  /* H, W, K_over_g, R, S */
                    Np,
                    pad_height, pad_width,
                    stride_height, stride_width,
                    out_buf_nopad, out_width,
                    outHmin, outHmax, outWmin, outWmax
            );
        } else {
            col2im(
                    col2im_buf,
                    m_start, m_end,
                    in_height, in_width, filt_batches, filt_height, filt_width,  /* H, W, K_over_g, R, S */
                    Np, filt_batches_padded,
                    pad_height, pad_width,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    out_buf_nopad,
                    outHmin, outHmax, outWmin, outWmax
            );
        }

        //find max/min of outputs
        if (need_compute_minmax) {
            o_end = compute_out_range(
                    m_end, in_width, out_width, 
                    stride_width, stride_height,
                    filt_width, filt_height,
                    dilation_width, dilation_height,
                    pad_width, pad_height);

            o_end = min(ROUNDDOWN(o_end,n2k), out_end);

            if (o_end > o_start) {
                find_minmax_hvx(out_buf_nopad+o_start*out_depth, minmax, (o_end-o_start)*out_depth);
                o_start = o_end;
            }
        }
    }

    if (need_compute_minmax) {
        if (out_end > o_start) {
            find_minmax_hvx(out_buf_nopad + o_start*out_depth, minmax, (out_end-o_start)*out_depth);
        }

        HVX_Vector vmax = minmax[0];
        HVX_Vector vmin = minmax[1];
        int nrot = 64;
        int i;
        for (i=0; i<5; i++) {
            vmax = Q6_Vw_vmax_VwVw(vmax,Q6_V_vror_VR(vmax,nrot));
            vmin = Q6_Vw_vmin_VwVw(vmin,Q6_V_vror_VR(vmin,nrot));
            nrot >>= 1;
        }
        minmax[0] = Q6_V_lo_W(Q6_W_vshuff_VVR(vmin,vmax,-4));
    }

    nn_sem_post(work->donesem);
}

static int deconv_hvx(struct nn_graph *nn, struct deconvnode_info *cinfo, 
                uint8_t *input_scr, int32_t *col2im_buf, int32_t *suma, int32_t *minmaxbuf)
{
    struct tdata_deconv work[NUM_THREADS];
    nn_sem_t donesem;
    nn_sem_init(&donesem,0);
    
    int out_width  = cinfo->out_width;
    int out_height = cinfo->out_height;
    int out_depth  = cinfo->out_depth;
    int n2k = 32 >> (min(5, Q6_R_ct0_R(out_depth)));

    int elements = out_height * out_width;
    int t_elements = ROUNDUP((elements + NUM_THREADS-1)/NUM_THREADS,n2k); 
    int n_threads = (elements + t_elements -1)/t_elements;
    int i; 

    // setup parameters for multi-thread 
    for (i=0; i < n_threads; i++) {
        work[i].info = cinfo;
        work[i].start = i * t_elements;
        work[i].end   = Q6_R_min_RR(i * t_elements + t_elements, elements);
        work[i].input_scr  = input_scr  + i*cinfo->input_scr_size;
        work[i].col2im_buf = col2im_buf + i*cinfo->col2im_buf_size;
        work[i].suma       = suma       + i*cinfo->suma_size;
        work[i].minmax     = minmaxbuf  + i*64;
        work[i].donesem = &donesem;
        nn_os_work_for_vector(nn, workitem_execute_deconv, &work[i]);
    }

    for (i=0; i < n_threads; i++) { nn_sem_wait(&donesem); }

    return n_threads;
}

static int deconv_bias_execute_hvx(struct nn_node *self, struct nn_graph *nn) {
    const struct tensor *in_tensor = self->inputs[OP_DECONVBIAS_INPUT_DATA_IDX];
    const struct tensor *filt_data_tensor = self->inputs[OP_DECONVBIAS_FILT_DATA_IDX];
    const struct tensor *filt_dim_tensor = self->inputs[OP_DECONVBIAS_FILT_DIM_IDX];
    const struct tensor *bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_IDX];
    const struct tensor *pad_tensor = self->inputs[OP_DECONVBIAS_PAD_IDX];
    const struct tensor *stride_tensor = self->inputs[OP_DECONVBIAS_STRIDE_IDX];
    const struct tensor *dilation_tensor = self->inputs[OP_DECONVBIAS_DILATION_IDX];
    const struct tensor *groups_tensor = self->inputs[OP_DECONVBIAS_GROUPS_IDX];
    const struct tensor *min_in_tensor = self->inputs[OP_DECONVBIAS_INPUT_MIN_IDX];
    const struct tensor *max_in_tensor = self->inputs[OP_DECONVBIAS_INPUT_MAX_IDX];
    const struct tensor *min_filt_tensor = self->inputs[OP_DECONVBIAS_FILT_MIN_IDX];
    const struct tensor *max_filt_tensor = self->inputs[OP_DECONVBIAS_FILT_MAX_IDX];
    const struct tensor *min_bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_MIN_IDX];
    const struct tensor *max_bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_MAX_IDX];

    const struct tensor *out_use_static_minmax_tensor = self->inputs[OP_DECONVBIAS_OUT_USE_STATIC_MINMAX_IDX];
    const struct tensor *out_static_min_tensor = self->inputs[OP_DECONVBIAS_OUT_STATIC_MIN_IDX];
    const struct tensor *out_static_max_tensor = self->inputs[OP_DECONVBIAS_OUT_STATIC_MAX_IDX];

    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_depth = in_tensor->shape.depth;

    // Note: deconv filter is passed in as (depth, height, width, batches)
    //       but filt shape is defined as (height, width, depth, batches)
    // Note: filt_batches is also known as K_over_g dimension
    int32_t filt_height = filt_dim_tensor->shape.filt_width;
    int32_t filt_width = filt_dim_tensor->shape.filt_depth;
    int32_t filt_batches = filt_dim_tensor->shape.filt_batches;
    int32_t filt_depth = filt_dim_tensor->shape.filt_height;

    int32_t filt_k_size = filt_data_tensor->shape.height;
    int32_t filt_n_size = filt_data_tensor->shape.width;

    int32_t bias_batches = bias_tensor->shape.batches;
    int32_t bias_height = bias_tensor->shape.height;
    int32_t bias_width = bias_tensor->shape.width;
    int32_t bias_depth = bias_tensor->shape.depth;

    int32_t pad_width = pad_tensor->shape.width;
    int32_t pad_height = pad_tensor->shape.height;

    int32_t stride_width = stride_tensor->shape.width;
    int32_t stride_height = stride_tensor->shape.height;

    int32_t dilation_width = dilation_tensor->shape.width;
    int32_t dilation_height = dilation_tensor->shape.height;

    int32_t groups = tensor_get_int32(groups_tensor, 0);

    uint8_t *input = (uint8_t *) in_tensor->data;
    uint8_t *filt = (uint8_t *) filt_data_tensor->data;
    uint8_t *bias = (uint8_t *) bias_tensor->data;
    uint8_t *output = (uint8_t *) out_tensor->data;

    float in_max_float = tensor_get_float(max_in_tensor,0);
    float in_min_float = tensor_get_float(min_in_tensor,0);
    float filt_max_float = tensor_get_float(max_filt_tensor,0);
    float filt_min_float = tensor_get_float(min_filt_tensor,0);
    float bias_max_float = tensor_get_float(max_bias_tensor,0);
    float bias_min_float = tensor_get_float(min_bias_tensor,0);

    int32_t out_use_static_minmax = tensor_get_int32(out_use_static_minmax_tensor,0);
    float out_static_min = tensor_get_float(out_static_min_tensor,0);
    float out_static_max = tensor_get_float(out_static_max_tensor,0);

    int32_t out_batches = in_batches;
    int32_t out_height = compute_out_size(in_height, filt_height, stride_height, pad_height, dilation_height);
    int32_t out_width = compute_out_size(in_width, filt_width, stride_width, pad_width, dilation_width);
    int32_t out_depth = filt_batches * groups;
    int32_t out_elements_per_batch = out_height * out_width * out_depth;
    int32_t out_elements_total = out_elements_per_batch * out_batches;
    int32_t out_size = out_elements_total*sizeof(uint8_t);

    float in_level_size = (in_max_float - in_min_float) / 255.0f;
    float filt_level_size = (filt_max_float - filt_min_float) / 255.0f;
    float bias_level_size = (bias_max_float - bias_min_float) / 255.0f;
    float out_level_size = in_level_size * filt_level_size;

    /* input_offset is 0.0f quantized to in min/max */
    /* filt_offset is 0.0f quantized to filt min/max */
    /* bias_offset is 0.0f quantized to bias min/max */
    int32_t in_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
    int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
    int32_t bias_offset = quantize_uint8(0.0f,bias_min_float,bias_max_float);

    /*
     * For bias addition, we need to convert values from bias space to the out space
     */
    float bias_mpy_amt = (bias_level_size / out_level_size);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias HVX IMPLEMENTATION");
    logmsg(nn,2, "deconv-bias execute node=%p id=%x",self,self->node_id);
    logmsg(nn,2, "deconv-bias input min/max=%f/%f",in_min_float,in_max_float);
    logmsg(nn,2, "deconv-bias input offset %d",in_offset);
    logmsg(nn,2, "deconv-bias input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
    logmsg(nn,2, "deconv-bias filt min/max=%f/%f",filt_min_float, filt_max_float);
    logmsg(nn,2, "deconv-bias filt offset %d",filt_offset);
    logmsg(nn,2, "deconv-bias filt data %dx%d",filt_k_size,filt_n_size);
    logmsg(nn,2, "deconv-bias filt dim %dx%dx%dx%d",filt_height,filt_width,filt_batches,filt_depth);
    logmsg(nn,2, "deconv-bias bias min/max=%f/%f",bias_min_float,bias_max_float);
    logmsg(nn,2, "deconv-bias bias offset %d",bias_offset);
    logmsg(nn,2, "deconv-bias bias %dx%dx%dx%d",bias_batches,bias_height,bias_width,bias_depth);
    logmsg(nn,2, "deconv-bias bias mpy %f",bias_mpy_amt);
    logmsg(nn,2, "deconv-bias padding %dx%d",pad_height, pad_width);
    logmsg(nn,2, "deconv-bias stride %dx%d",stride_height,stride_width);
    logmsg(nn,2, "deconv-bias dilation %dx%d",dilation_height,dilation_width);
    logmsg(nn,2, "deconv-bias groups %d",groups);
    logmsg(nn,2, "deconv-bias expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
    logmsg(nn,2, "deconv-bias out use_static=%d min/max=%f/%f",out_use_static_minmax,out_static_min,out_static_max);
#endif

    // Padded output depth and filt batches
    // Note: these are different when groups > 1
    //       out_depth = filt_batches * groups
    size_t filt_batches_padded = ROUNDUP(filt_batches, ODPAD);
    //size_t out_depth_padded = ROUNDUP(out_depth, ODPAD);

    // Gemmmpy requirements:
    //   M%4, K%8 and N%128
    size_t M = in_height * in_width;
    size_t N = filt_height * filt_width * filt_batches_padded;
    size_t K = in_depth;
    size_t Mp = ROUNDUP(M, MPAD);
    size_t Np = ROUNDUP(N, NPAD);
    size_t Kp = ROUNDUP(K, KPAD);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias filt_batches=%d, out_depth=%d", filt_batches, out_depth);
    logmsg(nn,2, "deconv-bias filt_batches_padded=%d, out_depth_padded=%d", filt_batches_padded, out_depth_padded);
    logmsg(nn,2, "deconv-bias M=%d, N=%d, K=%d", M, N, K);
    logmsg(nn,2, "deconv-bias Mp=%d, Np=%d, Kp=%d", Mp, Np, Kp);
#endif

    /* Assert in shape */
    if (in_batches < 1) return errlog(nn,"input shape batches %d", in_batches);
    if (in_height < 1) return errlog(nn,"input shape height %d", in_height);
    if (in_width < 1) return errlog(nn,"input shape width %d", in_width);
    if (in_depth < 1) return errlog(nn,"input shape depth %d", in_depth);

    /* Assert filt shape */
    if (filt_batches < 1) return errlog(nn,"filt shape batches %d", filt_batches);
    if (filt_height < 1) return errlog(nn,"filt shape height %d", filt_height);
    if (filt_width < 1) return errlog(nn,"filt shape width %d", filt_width);
    if (filt_depth < 1) return errlog(nn,"filt shape depth %d", filt_depth);
    if (filt_depth != in_depth) {
        return errlog(nn,"filt depth mismatches input depth %d vs %d", filt_depth, in_depth);
    }

    /* Assert filt size */
    if(filt_k_size != Kp) return errlog(nn,"filt k size %d, expected %d", filt_k_size, Kp);
    if(filt_n_size != Np) return errlog(nn,"filt n size %d, expected %d", filt_n_size, Np);

    /* Assert bias shape */
    if (bias_batches != 1) return errlog(nn,"bias shape batches %d", bias_batches);
    if (bias_height != 1) return errlog(nn,"bias shape height %d", bias_height);
    if (bias_width != 1) return errlog(nn,"bias shape width %d", bias_width);
    if (bias_depth != out_depth) {
        return errlog(nn,"bias depth mismatches output depth %d vs %d", bias_depth, out_depth);
    }

    /* Assert pad */
    if (pad_height < 0) return errlog(nn,"pad_height %d", pad_height);
    if (pad_width < 0) return errlog(nn,"pad_width %d", pad_width);

    /* Assert stride */
    if (stride_height < 1) return errlog(nn,"stride_height %d", stride_height);
    if (stride_width < 1) return errlog(nn,"stride_width %d", stride_width);

    /* Assert dilation */
    if (dilation_height < 1) return errlog(nn,"dilation_height %d", dilation_height);
    if (dilation_width < 1) return errlog(nn,"dilation_width %d", dilation_width);

    /* Assert groups */
    if (groups < 1) return errlog(nn,"groups %d", groups);

    if (groups != 1) {
        // TODO; add groups support
        return errlog(nn, "groups > 1 is not yet supported");
    }
    if (out_size > (out_tensor->max_size)) {
        return errlog(nn,"output too small, %d < %d", out_tensor->max_size, out_size);
    }
    tensor_set_shape(out_tensor, out_batches, out_height, out_width, out_depth);
    out_tensor->data_size = out_tensor->shape.batches *
                            out_tensor->shape.height *
                            out_tensor->shape.width *
                            out_tensor->shape.depth;
    tensor_set_shape(out_min_tensor, 1, 1, 1, 1);
    tensor_set_shape(out_max_tensor, 1, 1, 1, 1);
    out_min_tensor->data_size = sizeof(float);
    out_max_tensor->data_size = sizeof(float);

    // Let's determine a chunk size Mc s.t. a chunk of col2im buff of size Mc*Np
    // fits within the recommended size of scratch memory
#if 0
    size_t col2buf_target_size = HVX_M_BY_N_SIZE;
    while(col2buf_target_size < Np*sizeof(int) && col2buf_target_size < HVX_MAX_M_BY_N_SIZE) {
        col2buf_target_size *= 2;
    }
    if(col2buf_target_size > HVX_MAX_M_BY_N_SIZE) {
        // Np is too big
        return errlog(nn,"Np=%d is too large (more than %d bytes)", Np,HVX_M_BY_N_SIZE);
    }
    // Note: by rounding up here Mc*Np may go above max recommended size, but oh well
    size_t Mc = ROUNDUP((col2buf_target_size/sizeof(int)) / Np, MPAD);
    Mc = min(Mc, Mp);
#else
    if(Np*sizeof(int) > HVX_MAX_M_BY_N_SIZE) {
        // Np is too big
        return errlog(nn,"Np=%d is too large (more than %d bytes)", Np,HVX_M_BY_N_SIZE);
    }
    // make l2fetch size of input less than 16K if possible
    size_t Mc = ROUNDDOWN(64*1024/(Np*sizeof(int)),MPAD);
    Mc = max(Mc, MPAD);
    Mc = min(Mc, Mp);
#endif

#ifdef MEASURE_PERF
    uint64_t time_gemm = 0;
//  uint64_t time_col2im = 0;
//  uint64_t time_bias = 0;
//  uint64_t time_unpad = 0;
    uint64_t time_requantize = 0;
    uint64_t start;
#endif

    size_t input_scr_size = Mc*Kp;
    size_t input_scr_size_padded = PADDED_SIZE(input_scr_size, ALIGN_SIZE/sizeof(uint8_t));

    size_t col2im_buf_size = Mc*Np;
    size_t col2im_buf_size_padded = PADDED_SIZE(col2im_buf_size, ALIGN_SIZE/sizeof(int32_t));

    size_t out_buf_nopad_size_per_batch = out_elements_per_batch;
    size_t out_buf_nopad_size_per_batch_padded = PADDED_SIZE(out_buf_nopad_size_per_batch, ALIGN_SIZE/sizeof(int32_t));

    size_t bias_buf_size = out_depth;
    size_t bias_buf_size_padded = PADDED_SIZE(bias_buf_size, ALIGN_SIZE/sizeof(int32_t));

    size_t suma_size = Mc;
    size_t suma_size_padded = PADDED_SIZE(suma_size, ALIGN_SIZE/sizeof(int32_t));
    size_t sumb_size = Np;
    size_t sumb_size_padded = PADDED_SIZE(sumb_size, ALIGN_SIZE/sizeof(int32_t));

    size_t minmax_size = 64;  // 2x HVX vector length
    size_t minmax_size_padded = PADDED_SIZE(minmax_size, ALIGN_SIZE/sizeof(int32_t));

    int32_t is_every_batch_aligned = (in_batches == 1) || ((out_elements_per_batch & 127) == 0);
    size_t output_quantize_size = out_elements_per_batch;
    size_t output_quantize_size_padded = (is_every_batch_aligned) ? 0 : PADDED_SIZE(output_quantize_size, ALIGN_SIZE/sizeof(uint8_t));

    size_t total_scratch_size = bias_buf_size_padded*sizeof(int32_t) +
                                out_buf_nopad_size_per_batch_padded*sizeof(int32_t)*in_batches +
                                sumb_size_padded*sizeof(int32_t) + 
                                NUM_THREADS*(
                                input_scr_size_padded*sizeof(uint8_t) + 
                                col2im_buf_size_padded*sizeof(int32_t) +
                                suma_size_padded*sizeof(int32_t) + 
                                minmax_size_padded*sizeof(int32_t) +
                                output_quantize_size_padded*sizeof(uint8_t));

    if(nn_scratch_grow(nn, total_scratch_size)) {
        return errlog(nn,"scratch couldn't reallocate (need=%db, have=%db)", total_scratch_size, nn->scratch_size);
    }

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias Mc=%d", Mc);
    logmsg(nn,2, "deconv-bias col2imbuf size=%db", col2im_buf_size);
    logmsg(nn,2, "deconv-bias using %db of scratch mem", total_scratch_size);
#endif
    //Buffs for quantizing
    uint8_t* out_quantize =(uint8_t *)nn->scratch;
    //Buffs for deconv
    uint8_t* input_scr = (uint8_t *)(out_quantize + output_quantize_size_padded);
    int32_t* col2im_buf = (int32_t*)(input_scr + input_scr_size_padded*NUM_THREADS);
    int32_t* bias_buf = (int32_t*)(col2im_buf + col2im_buf_size_padded*NUM_THREADS);

    // Buffs used in gemm
    int32_t* suma = (int32_t*)(bias_buf + bias_buf_size_padded);
    int32_t* sumb = (int32_t*)(suma + suma_size_padded*NUM_THREADS);
    int32_t* minmax = (int32_t*)(sumb + sumb_size_padded);
    int32_t* out_buf_nopad = (int32_t*)(minmax + minmax_size_padded*NUM_THREADS);

    float out_max_val = -INFINITY;
    float out_min_val = INFINITY;
    if(out_use_static_minmax != 0) {
        out_max_val = out_static_max;
        out_min_val = out_static_min;
    }
    for (int b = 0; b < in_batches; b++){
        struct deconvnode_info info;

        info.input = input + (b*in_height*in_width*in_depth);
        info.filt = filt;
        info.output = output + out_elements_per_batch * b;
        info.bias_buf = bias_buf;
        info.out_buf = out_buf_nopad + out_buf_nopad_size_per_batch_padded * b;
        info.sumb = sumb;

        info.out_height = out_height;
        info.out_width = out_width;
        info.out_depth = out_depth;
        info.out_batches = out_batches;
        info.in_offset = in_offset;
        info.filt_offset = filt_offset;

        info.in_width = in_width;
        info.in_height = in_height;
        info.in_depth = in_depth;
        info.filt_height = filt_height;
        info.filt_width = filt_width;
        info.filt_batches = filt_batches;
        info.pad_width = pad_width;
        info.pad_height = pad_height;
        info.stride_width = stride_width;
        info.stride_height = stride_height;
        info.dilation_width = dilation_width;
        info.dilation_height = dilation_height;

        info.input_scr_size = input_scr_size_padded;
        info.col2im_buf_size = col2im_buf_size_padded;
        info.suma_size = suma_size_padded;
        info.Np = Np;
        info.Kp = Kp;
        info.Mc = Mc;
        info.need_compute_minmax = !out_use_static_minmax;

        bias_set(bias, bias_offset, bias_mpy_amt, out_depth, bias_buf);

        sumb_set_hvx(nn, &info);

    #ifdef MEASURE_PERF
        // PERF: start of bias_add + gemm + col2im + minmax
        start = HAP_perf_get_time_us();
    #endif
        int ntd = deconv_hvx(nn, &info, input_scr, col2im_buf, suma, minmax);

    #ifdef MEASURE_PERF
        // PERF: end of bias_add
        time_gemm += HAP_perf_get_time_us() - start;
    #endif

        // Requantize out buf down to 8
        float out_min_val_for_batch = 0;
        float out_max_val_for_batch = 0;

        if(out_use_static_minmax == 0) {
            int32_t in_max_val = INT32_MIN;
            int32_t in_min_val = INT32_MAX;
            int i;
            for (i=0; i < ntd; i++) {
                in_max_val = Q6_R_max_RR(in_max_val,minmax[i*64+0]);
                in_min_val = Q6_R_min_RR(in_min_val,minmax[i*64+1]);
            }

            out_min_val_for_batch = out_level_size * (float)in_min_val;
            out_max_val_for_batch = out_level_size * (float)in_max_val;
            if (out_min_val_for_batch < out_min_val) out_min_val = out_min_val_for_batch;
            if (out_max_val_for_batch > out_max_val) out_max_val = out_max_val_for_batch;
        }
    }
#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias out min/max=%f/%f",out_min_val,out_max_val);
#endif

    float stepsize;
    float recip_stepsize;
    quantize_adjust_range(
            &out_min_val,&out_max_val,
            &stepsize,&recip_stepsize,
            out_min_val,out_max_val);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias out adjusted min/max=%f/%f",out_min_val,out_max_val);
#endif
    tensor_set_float(out_min_tensor, 0, out_min_val);
    tensor_set_float(out_max_tensor, 0, out_max_val);

#ifdef MEASURE_PERF
    // PERF: start of requantize
    start = HAP_perf_get_time_us();
#endif
    int32_t i;
    // Taken from op_requantize.c
    for (int b = 0; b < in_batches; b++){
        if(out_elements_per_batch < 128) {
            for (i=0; i<out_elements_per_batch; i++) {
                output[out_elements_per_batch * b + i] =
                quantize_uint8(out_level_size *
                    (float)out_buf_nopad[out_buf_nopad_size_per_batch_padded * b + i],
                    out_min_val,out_max_val);
            }
        } else {
            float gain = (255.f * out_level_size) / (out_max_val - out_min_val);
            int gaini = (int) (gain * 0x1.0p31f + 0.5f);
            int offseti = out_min_val / out_level_size;
            quantize_int32_to_u8(nn,
                out_buf_nopad + out_buf_nopad_size_per_batch_padded * b,
                out_elements_per_batch,offseti,gaini,
                output + out_elements_per_batch * b);
        }
    }

#ifdef MEASURE_PERF
    // PERF: end of requantize
    time_requantize += HAP_perf_get_time_us() - start;

    logmsg(nn,2, "deconv-bias PERF gemm=%d us", time_gemm);
//  logmsg(nn,2, "deconv-bias PERF col2im=%d us", time_col2im);
//  logmsg(nn,2, "deconv-bias PERF bias=%d us", time_bias);
//  logmsg(nn,2, "deconv-bias PERF unpad=%d us", time_unpad);
    logmsg(nn,2, "deconv-bias PERF requantize=%d us", time_requantize);
#endif

    return 0;
}



static int deconv_bias_execute_ref(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[OP_DECONVBIAS_INPUT_DATA_IDX];
    const struct tensor *filt_data_tensor = self->inputs[OP_DECONVBIAS_FILT_DATA_IDX];
    const struct tensor *filt_dim_tensor = self->inputs[OP_DECONVBIAS_FILT_DIM_IDX];
    const struct tensor *bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_IDX];
    const struct tensor *pad_tensor = self->inputs[OP_DECONVBIAS_PAD_IDX];
    const struct tensor *stride_tensor = self->inputs[OP_DECONVBIAS_STRIDE_IDX];
    const struct tensor *dilation_tensor = self->inputs[OP_DECONVBIAS_DILATION_IDX];
    const struct tensor *groups_tensor = self->inputs[OP_DECONVBIAS_GROUPS_IDX];
    const struct tensor *min_in_tensor = self->inputs[OP_DECONVBIAS_INPUT_MIN_IDX];
    const struct tensor *max_in_tensor = self->inputs[OP_DECONVBIAS_INPUT_MAX_IDX];
    const struct tensor *min_filt_tensor = self->inputs[OP_DECONVBIAS_FILT_MIN_IDX];
    const struct tensor *max_filt_tensor = self->inputs[OP_DECONVBIAS_FILT_MAX_IDX];
    const struct tensor *min_bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_MIN_IDX];
    const struct tensor *max_bias_tensor = self->inputs[OP_DECONVBIAS_BIAS_MAX_IDX];

    const struct tensor *out_use_static_minmax_tensor = self->inputs[OP_DECONVBIAS_OUT_USE_STATIC_MINMAX_IDX];
    const struct tensor *out_static_min_tensor = self->inputs[OP_DECONVBIAS_OUT_STATIC_MIN_IDX];
    const struct tensor *out_static_max_tensor = self->inputs[OP_DECONVBIAS_OUT_STATIC_MAX_IDX];

    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_depth = in_tensor->shape.depth;

    // Note: deconv filter is passed in as (depth, height, width, batches)
    //       but filt shape is defined as (height, width, depth, batches)
    int32_t filt_height = filt_dim_tensor->shape.filt_width;
    int32_t filt_width = filt_dim_tensor->shape.filt_depth;
    int32_t filt_batches = filt_dim_tensor->shape.filt_batches;
    int32_t filt_depth = filt_dim_tensor->shape.filt_height;

    int32_t filt_k_size = filt_data_tensor->shape.height;
    int32_t filt_n_size = filt_data_tensor->shape.width;

    int32_t bias_batches = bias_tensor->shape.batches;
    int32_t bias_height = bias_tensor->shape.height;
    int32_t bias_width = bias_tensor->shape.width;
    int32_t bias_depth = bias_tensor->shape.depth;

    int32_t pad_width = pad_tensor->shape.width;
    int32_t pad_height = pad_tensor->shape.height;

    int32_t stride_width = stride_tensor->shape.width;
    int32_t stride_height = stride_tensor->shape.height;

    int32_t dilation_width = dilation_tensor->shape.width;
    int32_t dilation_height = dilation_tensor->shape.height;

    int32_t groups = tensor_get_int32(groups_tensor, 0);

    uint8_t *input = (uint8_t *) in_tensor->data;
    uint8_t *filt = (uint8_t *) filt_data_tensor->data;
    uint8_t *bias = (uint8_t *) bias_tensor->data;
    uint8_t *output = (uint8_t *) out_tensor->data;

    float in_max_float = tensor_get_float(max_in_tensor,0);
    float in_min_float = tensor_get_float(min_in_tensor,0);
    float filt_max_float = tensor_get_float(max_filt_tensor,0);
    float filt_min_float = tensor_get_float(min_filt_tensor,0);
    float bias_max_float = tensor_get_float(max_bias_tensor,0);
    float bias_min_float = tensor_get_float(min_bias_tensor,0);

    int32_t out_use_static_minmax = tensor_get_int32(out_use_static_minmax_tensor,0);
    float out_static_min = tensor_get_float(out_static_min_tensor,0);
    float out_static_max = tensor_get_float(out_static_max_tensor,0);

    int32_t out_batches = in_batches;
    int32_t out_height = compute_out_size(in_height, filt_height, stride_height, pad_height, dilation_height);
    int32_t out_width = compute_out_size(in_width, filt_width, stride_width, pad_width, dilation_width);
    int32_t out_depth = filt_batches * groups;
    int32_t out_elements_per_batch = out_height * out_width * out_depth;
    int32_t out_elements_total = out_elements_per_batch * out_batches;

    int32_t out_size = out_elements_total*sizeof(uint8_t);

    float in_level_size = (in_max_float - in_min_float) / 255.0f;
    float filt_level_size = (filt_max_float - filt_min_float) / 255.0f;
    float bias_level_size = (bias_max_float - bias_min_float) / 255.0f;
    float out_level_size = in_level_size * filt_level_size;

    /* input_offset is 0.0f quantized to in min/max */
    /* filt_offset is 0.0f quantized to filt min/max */
    /* bias_offset is 0.0f quantized to bias min/max */
    int32_t in_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
    int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
    int32_t bias_offset = quantize_uint8(0.0f,bias_min_float,bias_max_float);

    /*
     * For bias addition, we need to convert values from bias space to the out space
     */
    float bias_mpy_amt = (bias_level_size / out_level_size);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias REF IMPLEMENTATION");
    logmsg(nn,2, "deconv-bias execute node=%p id=%x",self,self->node_id);
    logmsg(nn,2, "deconv-bias input min/max=%f/%f",in_min_float,in_max_float);
    logmsg(nn,2, "deconv-bias input offset %d",in_offset);
    logmsg(nn,2, "deconv-bias input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
    logmsg(nn,2, "deconv-bias filt min/max=%f/%f",filt_min_float, filt_max_float);
    logmsg(nn,2, "deconv-bias filt offset %d",filt_offset);
    logmsg(nn,2, "deconv-bias filt data %dx%d",filt_k_size,filt_n_size);
    logmsg(nn,2, "deconv-bias filt dim %dx%dx%dx%d",filt_height,filt_width,filt_batches,filt_depth);
    logmsg(nn,2, "deconv-bias bias min/max=%f/%f",bias_min_float,bias_max_float);
    logmsg(nn,2, "deconv-bias bias offset %d",bias_offset);
    logmsg(nn,2, "deconv-bias bias %dx%dx%dx%d",bias_batches,bias_height,bias_width,bias_depth);
    logmsg(nn,2, "deconv-bias bias mpy %f",bias_mpy_amt);
    logmsg(nn,2, "deconv-bias padding %dx%d",pad_height, pad_width);
    logmsg(nn,2, "deconv-bias stride %dx%d",stride_height,stride_width);
    logmsg(nn,2, "deconv-bias dilation %dx%d",dilation_height,dilation_width);
    logmsg(nn,2, "deconv-bias groups %d",groups);
    logmsg(nn,2, "deconv-bias expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
    logmsg(nn,2, "deconv-bias out use_static=%d min/max=%f/%f",out_use_static_minmax,out_static_min,out_static_max);
#endif

    // Padded filt batches size
    // We receive the filter with the K_over_g dimension padded to this size
    size_t filt_batches_padded = ROUNDUP(filt_batches, ODPAD);

    // Gemmmpy requirements:
    //   M%4, K%8 and N%128
    size_t M = in_height * in_width;
    size_t N = filt_height * filt_width * filt_batches_padded;
    size_t K = in_depth;
    //size_t Mp = ROUNDUP(M, MPAD);
    size_t Np = ROUNDUP(N, NPAD);
    size_t Kp = ROUNDUP(K, KPAD);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias filt_batches=%d, filt_batches_padded=%d", filt_batches, filt_batches_padded);
    logmsg(nn,2, "deconv-bias M=%d, N=%d, K=%d", M, N, K);
    logmsg(nn,2, "deconv-bias Mp=%d, Np=%d, Kp=%d", Mp, Np, Kp);
#endif

    /* Assert in shape */
    if (in_batches < 1) return errlog(nn,"input shape batches %d", in_batches);
    if (in_height < 1) return errlog(nn,"input shape height %d", in_height);
    if (in_width < 1) return errlog(nn,"input shape width %d", in_width);
    if (in_depth < 1) return errlog(nn,"input shape depth %d", in_depth);

    /* Assert filt shape */
    if (filt_batches < 1) return errlog(nn,"filt shape batches %d", filt_batches);
    if (filt_height < 1) return errlog(nn,"filt shape height %d", filt_height);
    if (filt_width < 1) return errlog(nn,"filt shape width %d", filt_width);
    if (filt_depth < 1) return errlog(nn,"filt shape depth %d", filt_depth);
    if (filt_depth != in_depth) {
        return errlog(nn,"filt depth mismatches input depth %d vs %d", filt_depth, in_depth);
    }

    /* Assert filt size */
    if(filt_k_size != Kp) return errlog(nn,"filt k size %d, expected %d", filt_k_size, Kp);
    if(filt_n_size != Np) return errlog(nn,"filt n size %d, expected %d", filt_n_size, Np);

    /* Assert bias shape */
    if (bias_batches != 1) return errlog(nn,"bias shape batches %d", bias_batches);
    if (bias_height != 1) return errlog(nn,"bias shape height %d", bias_height);
    if (bias_width != 1) return errlog(nn,"bias shape width %d", bias_width);
    if (bias_depth != out_depth) {
        return errlog(nn,"bias depth mismatches output depth %d vs %d", bias_depth, out_depth);
    }

    /* Assert pad */
    if (pad_height < 0) return errlog(nn,"pad_height %d", pad_height);
    if (pad_width < 0) return errlog(nn,"pad_width %d", pad_width);

    /* Assert stride */
    if (stride_height < 1) return errlog(nn,"stride_height %d", stride_height);
    if (stride_width < 1) return errlog(nn,"stride_width %d", stride_width);

    /* Assert dilation */
    if (dilation_height < 1) return errlog(nn,"dilation_height %d", dilation_height);
    if (dilation_width < 1) return errlog(nn,"dilation_width %d", dilation_width);

    /* Assert groups */
    if (groups < 1) return errlog(nn,"groups %d", groups);

    if (out_size > (out_tensor->max_size)) {
        return errlog(nn,"output too small, %d < %d", out_tensor->max_size, out_size);
    }
    tensor_set_shape(out_tensor, out_batches, out_height, out_width, out_depth);
    out_tensor->data_size = out_tensor->shape.batches *
                            out_tensor->shape.height *
                            out_tensor->shape.width *
                            out_tensor->shape.depth;
    tensor_set_shape(out_min_tensor, 1, 1, 1, 1);
    tensor_set_shape(out_max_tensor, 1, 1, 1, 1);
    out_min_tensor->data_size = sizeof(float);
    out_max_tensor->data_size = sizeof(float);

    // Do the gemm+col2im in chunks of M so that we fit into
    // scratch mem
    if(N*sizeof(int) > REF_MAX_M_BY_N_SIZE) {
        // N is way too big to fit in our scratch mem
        return errlog(nn,"N=%d is too large to fit in scratch mem", N);
    }
    size_t Mc = (REF_MAX_M_BY_N_SIZE / sizeof(int)) / N;
    Mc = min(Mc, M);

    size_t col2im_buf_size = Mc*N*sizeof(int);
    size_t col2im_buf_size_padded = PADDED_SIZE(col2im_buf_size, ALIGN_SIZE);
    size_t out_buf_size = out_elements_per_batch*sizeof(int32_t);
    size_t out_buf_size_padded = PADDED_SIZE(out_buf_size, ALIGN_SIZE);

    size_t total_scratch_size = col2im_buf_size_padded + out_buf_size_padded*out_batches;
    if(nn_scratch_grow(nn, total_scratch_size)) {
        return errlog(nn,"scratch couldn't reallocate (need=%db, have=%db)", total_scratch_size, nn->scratch_size);
    }

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias Mc=%d", Mc);
    logmsg(nn,2, "deconv-bias using %db of scratch mem", total_scratch_size);
#endif

    int* col2im_buf = (int*) nn->scratch;
    int32_t* out_buf_all = (int32_t*) pad_and_align(col2im_buf, col2im_buf_size) ;
    float out_max_val = -INFINITY;
    float out_min_val = INFINITY;
    for (int b = 0; b < out_batches; b++){
        int32_t* out_buf_for_batch = out_buf_all + out_buf_size_padded * b;
        // We use output buffer to accumulate
        memset(out_buf_for_batch, 0, out_buf_size);

        size_t m_start;
        for(m_start=0; m_start<M; m_start+=Mc) {
            size_t m_end = min(m_start+Mc, M);
            size_t Mc_i = m_end - m_start;

            uint8_t* input_chunk = input + m_start*K;
            if (groups == 1) {
                gemm_ref(
                        input_chunk, filt,
                        in_offset, filt_offset,
                        Mc_i, N, K,
                        Kp,
                        (int *) col2im_buf
                );
            } else {
                // TODO; add groups support
                return errlog(nn, "groups > 1 is not yet supported");
            }


            // Col2im
            col2im_ref(
                    col2im_buf,
                    m_start, m_end,
                    in_height, in_width, filt_batches, filt_height, filt_width,  /* H, W, K_over_g, R, S */
                    filt_batches_padded,
                    pad_height, pad_width,
                    stride_height, stride_width,
                    dilation_height, dilation_width,
                    out_buf_for_batch
            );
        }

        // Bias add
        bias_add_ref(bias, bias_offset, bias_mpy_amt,
                 out_buf_for_batch,
                 out_depth, out_width*out_height);

        // Requantize out buf down to 8
        int32_t i;
        int32_t inval;
        float out_min_val_for_batch;
        float out_max_val_for_batch;
        if(out_use_static_minmax == 0) {
            // Don't have static min/max, need to find it ourselves
            int32_t in_max_val = INT32_MIN;
            int32_t in_min_val = INT32_MAX;
            for (i=0; i<out_elements_per_batch; i++) {
                inval = out_buf_for_batch[i];
                if (inval > in_max_val) in_max_val = inval;
                if (inval < in_min_val) in_min_val = inval;
            }
            out_min_val_for_batch = out_level_size * (float)in_min_val;
            out_max_val_for_batch = out_level_size * (float)in_max_val;
            if (out_min_val_for_batch < out_min_val) out_min_val = out_min_val_for_batch;
            if (out_max_val_for_batch > out_max_val) out_max_val = out_max_val_for_batch;
        } else {
            // Use given static min/max
            out_max_val = out_static_max;
            out_min_val = out_static_min;
        }
    }
#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias out min/max=%f/%f",out_min_val,out_max_val);
#endif

    float stepsize;
    float recip_stepsize;
    quantize_adjust_range(
            &out_min_val,&out_max_val,
            &stepsize,&recip_stepsize,
            out_min_val,out_max_val);

#ifdef DEBUG_PRINT
    logmsg(nn,2, "deconv-bias out adjusted min/max=%f/%f",out_min_val,out_max_val);
#endif
    tensor_set_float(out_min_tensor, 0, out_min_val);
    tensor_set_float(out_max_tensor, 0, out_max_val);
    for (int b = 0; b < out_batches; b++){
        int32_t* out_buf_for_batch = out_buf_all + out_buf_size_padded * b;
        for (int i=0; i<out_elements_per_batch; i++) {
            output[out_elements_per_batch*b + i] = quantize_uint8(out_level_size * (float)out_buf_for_batch[i],out_min_val,out_max_val);
        }
    }

    return 0;
}

struct nn_node_ops nn_ops_for_DeconvBias_8x8to32 = {
        .execute = deconv_bias_execute_hvx,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_DECONVBIAS_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_DeconvBias_8x8to32_ref = {
        .execute = deconv_bias_execute_ref,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_DECONVBIAS_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};
