/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains hvx functions used by reduce sum / reduce mean
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#include "hvx_inlines.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
#define MAX_Q_VAL 255
#define NUM_DIMS 4

//To find the sum, perform parallel prefix using rotations - log2 steps
static inline HVX_Vector compute_reduced_sum(HVX_Vector in)
{
    HVX_Vector in_rot = Q6_V_vror_VR(in, 1 * sizeof(uint32_t));
    HVX_Vector cur_sum = Q6_Vw_vadd_VwVw_sat(in, in_rot);
    in_rot = Q6_V_vror_VR(cur_sum, 2 * sizeof(uint32_t));
    cur_sum = Q6_Vw_vadd_VwVw_sat(cur_sum, in_rot);
    in_rot = Q6_V_vror_VR(cur_sum, 4 * sizeof(uint32_t));
    cur_sum = Q6_Vw_vadd_VwVw_sat(cur_sum, in_rot);
    in_rot = Q6_V_vror_VR(cur_sum, 8 * sizeof(uint32_t));
    cur_sum = Q6_Vw_vadd_VwVw_sat(cur_sum, in_rot);
    in_rot = Q6_V_vror_VR(cur_sum, 16 * sizeof(uint32_t));
    cur_sum = Q6_Vw_vadd_VwVw_sat(cur_sum, in_rot);
    return cur_sum;
}

static inline void reduce_sum_all_axes_hvx(const uint8_t * in_data, int32_t * out_data, const int blob_size)
{
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;

    HVX_Vector cur_sum;
    cur_sum = q6op_Vb_vsplat_R(0); // initialize to all 0

    for (xd = 0; xd + 128 <= blob_size; xd += 128)
    {
        //reduce the vector to a word sum vector
        cur_sum = Q6_Vuw_vrmpyacc_VuwVubRub(cur_sum, *(HVX_Vector *)&in_data[xd], 0x01010101);
    }
    // need to mux the leftovers with all zeros because the input could be corrupted at the end
    if (leftovers)
    {
        HVX_VectorPred mux_pred = Q6_Q_vsetq_R(leftovers);
        HVX_Vector in = *(HVX_Vector *)&in_data[xd];
        HVX_Vector last_in = Q6_V_vmux_QVV(mux_pred, in, q6op_Vb_vsplat_R(0));
        cur_sum = Q6_Vuw_vrmpyacc_VuwVubRub(cur_sum, last_in, 0x01010101);
    }

    cur_sum = compute_reduced_sum(cur_sum);
    out_data[0] = *(uint32_t *)&cur_sum;
}

//8 bit input
static inline void reduce_sum_single_axis_hvx_8(const uint8_t * in_data, int32_t * out_data,
                                        const int num_batches, const int num_blobs, const int blob_size)
{
    const int leftovers = blob_size % (sizeof(HVX_Vector) / sizeof(int32_t));
    int xd = 0;
    HVX_Vector cur_sum;
    HVX_Vector sum0;
    HVX_Vector sum1;
    HVX_Vector sum2;
    HVX_Vector sum3;

    for (int n = 0; n < num_batches; n++)
    {
        // computes the sum for one section of output
        for (xd = 0; xd + 32 <= blob_size; xd += 32)
        {
            cur_sum = q6op_Vb_vsplat_R(0);
            sum0 = q6op_Vb_vsplat_R(0);
            sum1 = q6op_Vb_vsplat_R(0);
            sum2 = q6op_Vb_vsplat_R(0);
            sum3 = q6op_Vb_vsplat_R(0);
            /*The loop below computes partial sums:
             * Sum0 = [sum(w0),sum(w4), sum(w8)...]
             * Sum1 = [sum(w1),sum(w5), sum(w9)...]
             * Sum2 = [sum(w2),sum(w6), sum(w10)...]
             * Sum3 = [sum(w3),sum(w7), sum(w11)...]
             */

            for (int i = 0; i < num_blobs; i++)
            {
                HVX_Vector in = vmemu(&in_data[i * blob_size + xd]);
                sum0 = Q6_Vuw_vrmpyacc_VuwVubRub(sum0, in, 0x00000001);
                sum1 = Q6_Vuw_vrmpyacc_VuwVubRub(sum1, in, 0x00000100);
                sum2 = Q6_Vuw_vrmpyacc_VuwVubRub(sum2, in, 0x00010000);
                sum3 = Q6_Vuw_vrmpyacc_VuwVubRub(sum3, in, 0x01000000);
            }
            //Shuffle the sums computed above into the correct order [sum(w0), sum(w1)....sum(127)]
            // In : [0, 4,  8 ... 124]
            // In : [2, 6, 10 ... 126]
            // Out : [ 0,  2,  4, ...  62]
            // Out : [64, 66, 68, ... 126]
            HVX_VectorPair sum0_2 = Q6_W_vshuff_VVR(sum2, sum0, -4);

            // In : [1, 5,  9 ... 125]
            // In : [3, 7, 11 ... 127]
            // Out : [ 1,  3,  5, ...  63]
            // Out : [65, 67, 69, ... 127]
            HVX_VectorPair sum1_3 = Q6_W_vshuff_VVR(sum3, sum1, -4);

            // In : [ 0,  2,  4, ...  62]
            // In : [ 1,  3,  5, ...  63]
            // Out : [ 0,   1,  2, ...  31]
            // Out : [32,  33, 34, ...  63]
            HVX_VectorPair sum_low = Q6_W_vshuff_VVR(Q6_V_lo_W(sum1_3), Q6_V_lo_W(sum0_2), -4);

            // In : [64, 66, 68, ... 126]
            // In : [65, 67, 69, ... 127]
            // Out : [64, 65, 66, ...  95]
            // Output : [96, 97, 98, ... 127]
            HVX_VectorPair sum_hi = Q6_W_vshuff_VVR(Q6_V_hi_W(sum1_3), Q6_V_hi_W(sum0_2), -4);

            vmemu(&out_data[xd + 0]) = Q6_V_lo_W(sum_low);
            vmemu(&out_data[xd + 32]) = Q6_V_hi_W(sum_low);
            vmemu(&out_data[xd + 64]) = Q6_V_lo_W(sum_hi);
            vmemu(&out_data[xd + 96]) = Q6_V_hi_W(sum_hi);
        }
        if (leftovers)
        {
            cur_sum = q6op_Vb_vsplat_R(0);
            sum0 = q6op_Vb_vsplat_R(0);
            sum1 = q6op_Vb_vsplat_R(0);
            sum2 = q6op_Vb_vsplat_R(0);
            sum3 = q6op_Vb_vsplat_R(0);
            for (int i = 0; i < num_blobs; i++)
            {
                HVX_Vector in = vmemu(&in_data[i * blob_size + xd]);
                sum0 = Q6_Vuw_vrmpyacc_VuwVubRub(sum0, in, 0x00000001);
                sum1 = Q6_Vuw_vrmpyacc_VuwVubRub(sum1, in, 0x00000100);
                sum2 = Q6_Vuw_vrmpyacc_VuwVubRub(sum2, in, 0x00010000);
                sum3 = Q6_Vuw_vrmpyacc_VuwVubRub(sum3, in, 0x01000000);
            }
            HVX_VectorPair abcd_even = Q6_W_vshuff_VVR(sum2, sum0, -4);
            HVX_VectorPair abcd_odd = Q6_W_vshuff_VVR(sum3, sum1, -4);
            HVX_VectorPair abcd_low = Q6_W_vshuff_VVR(Q6_V_lo_W(abcd_odd), Q6_V_lo_W(abcd_even), -4);
            HVX_Vector *outp = &vmemu(&out_data[xd]);
            q6op_vstu_variable_ARV(outp, leftovers * sizeof(uint32_t), Q6_V_lo_W(abcd_low));
        }
        in_data += num_blobs * blob_size;
        out_data += blob_size;
    }
}

//32 bit input
static inline void reduce_sum_single_axis_hvx_32(const int32_t * in_data, int32_t * out_data,
                                         const int num_batches, const int num_blobs, const int blob_size)
{
    const int leftovers = blob_size % (sizeof(HVX_Vector));

    int xd = 0;
    HVX_Vector cur_sum;

    for (int n = 0; n < num_batches; n++)
    {
        // computes the sum for one section of output
        for (xd = 0; xd + 32 <= blob_size; xd += 32)
        {
            cur_sum = q6op_Vb_vsplat_R(0);
            for (int i = 0; i < num_blobs; i++)
            {
                HVX_Vector cnv = vmemu(&in_data[i * blob_size + xd]);
                cur_sum = Q6_Vw_vadd_VwVw_sat(cnv, cur_sum);
            }
            *(HVX_Vector *)&out_data[xd] = cur_sum;
        }
        if (leftovers)
        {
            cur_sum = q6op_Vb_vsplat_R(0);
            for (int i = 0; i < num_blobs; i++)
            {
                HVX_Vector in = vmemu(&in_data[i * blob_size + xd]);
                cur_sum = Q6_Vw_vadd_VwVw_sat(in, cur_sum);
            }
            HVX_Vector *outp = (HVX_Vector *)&out_data[xd];
            q6op_vstu_variable_ARV(outp, leftovers * sizeof(uint32_t), cur_sum);
        }
        // step to next chunk of input and output
        in_data += num_blobs * blob_size;
        out_data += blob_size;
    }
}
