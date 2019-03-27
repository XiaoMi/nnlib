/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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
 * This contains implementations for quantized reduce sum node
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


struct t8data
{
    uint8_t *in_data;
    int32_t *out_data;
    int32_t reduction_batches;
    int32_t num_blobs;
    int32_t blob_size;
    nn_sem_t donesem;
};

struct t32data
{
    int32_t *in_data;
    int32_t *out_data;
    int32_t reduction_batches;
    int32_t num_blobs;
    int32_t blob_size;
    nn_sem_t donesem;
};

struct tadjdata
{
    int32_t *in_data;
    int32_t *out_data;
    int32_t out_data_size;
    int32_t adj;
    nn_sem_t donesem;
};

//To find the sum, perform parallel prefix using rotations - log2 steps
static HVX_Vector compute_reduced_sum (HVX_Vector in, struct nn_graph *nn)
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

static void reduce_sum_all_axes_hvx(struct nn_graph *nn, void *vtd)
{
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;

    HVX_Vector cur_sum;
    cur_sum = q6op_Vb_vsplat_R(0); // initialize to all 0

    for (xd = 0; xd + 128 <= blob_size; xd += 128)
    {
        //reduce the vector to a word sum vector
        cur_sum = Q6_Vuw_vrmpyacc_VuwVubRub(cur_sum, *(HVX_Vector*) &in_data[xd], 0x01010101);
    }
    // need to mux the leftovers with all zeros because the input could be corrupted at the end
    if (leftovers)
    {
        HVX_VectorPred mux_pred = Q6_Q_vsetq_R(leftovers);
        HVX_Vector in = *(HVX_Vector * )&in_data[xd];
        HVX_Vector last_in = Q6_V_vmux_QVV(mux_pred, in, q6op_Vb_vsplat_R(0));
        cur_sum = Q6_Vuw_vrmpyacc_VuwVubRub(cur_sum, last_in, 0x01010101);
    }

    cur_sum = compute_reduced_sum(cur_sum, nn);
    out_data[0] = *(uint32_t *)&cur_sum;
    nn_sem_post(&td->donesem);
}

//8 bit input
static void reduce_sum_single_axis_hvx_8(struct nn_graph *nn, void *vtd)
{
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int num_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
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

            *(HVX_Vector *) &out_data[xd + 0] = Q6_V_lo_W(sum_low);
            *(HVX_Vector *) &out_data[xd + 32] = Q6_V_hi_W(sum_low);
            *(HVX_Vector *) &out_data[xd + 64] = Q6_V_lo_W(sum_hi);
            *(HVX_Vector *) &out_data[xd + 96] = Q6_V_hi_W(sum_hi);
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
            HVX_Vector *outp = (HVX_Vector *) &out_data[xd];
            q6op_vstu_variable_ARV( outp, leftovers * sizeof(uint32_t), Q6_V_lo_W(abcd_low));
        }
        in_data += num_blobs * blob_size;
        out_data += blob_size;
    }

    nn_sem_post(&td->donesem);
}

//32 bit input
static void reduce_sum_single_axis_hvx_32(struct nn_graph *nn, void *vtd)
{
    struct t32data *td = vtd;
    int32_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int num_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
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
            *(HVX_Vector *) &out_data[xd] = cur_sum;
        }
        if (leftovers)
        {
            cur_sum = q6op_Vb_vsplat_R(0);
            for (int i = 0; i < num_blobs; i++)
            {
                HVX_Vector in = vmemu(&in_data[i * blob_size + xd]);
                cur_sum = Q6_Vw_vadd_VwVw_sat(in, cur_sum);
            }
                HVX_Vector *outp = (HVX_Vector *) &out_data[xd];
                q6op_vstu_variable_ARV( outp, leftovers * sizeof(uint32_t), cur_sum);
            }
            // step to next chunk of input and output
            in_data += num_blobs * blob_size;
            out_data += blob_size;
        }
        nn_sem_post(&td->donesem);
}

// Adjust output min max so that they are symmetric around zero
static void adjust_output_range(struct nn_graph *nn, void *vtd) {
    struct tadjdata *td = vtd;
    int32_t *in_data = td->in_data;
    int32_t *out_data = (int32_t *)td->out_data;
    int32_t out_data_size = td->out_data_size;
    int32_t adj = td->adj;
    HVX_Vector adjv = Q6_V_vsplat_R(adj);

    for (int i = 0; i < out_data_size; i += sizeof(HVX_Vector) / sizeof(int32_t))
    {
        HVX_Vector outv = *(HVX_Vector *) &in_data[i];
        *(HVX_Vector *) &out_data[i] = Q6_Vw_vsub_VwVw_sat(outv, adjv);
    }
    nn_sem_post(&td->donesem);
}

    static int reducing_sum_execute(struct nn_node *self, struct nn_graph *nn)
    {
        const struct tensor *in_tensor = self->inputs[0];
        const struct tensor *in_min_tensor = self->inputs[1];
        const struct tensor *in_max_tensor = self->inputs[2];
        const struct tensor *axes_tensor = self->inputs[3];
        struct tensor *out_tensor = self->outputs[0];
        struct tensor *out_min_tensor = self->outputs[1];
        struct tensor *out_max_tensor = self->outputs[2];
        int32_t in_batches = in_tensor->shape.batches;
        int32_t in_height = in_tensor->shape.height;
        int32_t in_width = in_tensor->shape.width;
        int32_t in_depth = in_tensor->shape.depth;
        int32_t* axes = (int32_t*)axes_tensor->data;
        int32_t axes_size = axes_tensor->data_size / sizeof(int32_t);
        uint8_t* in_data = in_tensor->data;
        int32_t* out_data = nn->scratch;
        int32_t elemcount = in_batches * in_height * in_width * in_depth;

        int32_t modified_shape_final[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
        int32_t modified_data_size = 1;
        for (int i = 0; i < axes_size; i++)
        {
            modified_shape_final[axes[i]] = 1; // if our axis is being reduced on, it's size is 1
        }
        for (int i = 0; i < NUM_DIMS; i++)
        {
            modified_data_size *= modified_shape_final[i]; // set the final size
        }

        int32_t out_batches = out_tensor->shape.batches;
        int32_t out_height = out_tensor->shape.height;
        int32_t out_width = out_tensor->shape.width;
        int32_t out_depth = out_tensor->shape.depth;
        int32_t out_data_size = out_batches * out_height * out_width * out_depth;

        // sanity checks
        if (out_data_size != modified_data_size)
        {
            return errlog(nn, "Output tensor is of size %d, but expected output tensor of size %d", out_data_size, modified_data_size);
        }
        if (nn_scratch_grow(nn, elemcount * sizeof(int32_t)))
        {
            return errlog(nn,"failed to get scratch");
        }
        if (tensor_out_prepare_normal_fromshape( out_tensor, & out_tensor->shape, NN_TYPE_INT32 )!= 0)
        {
            return errlog(nn, "out too small");
        }

        if (axes_size == 4 || modified_data_size == 1) { // reduce on all dimensions
            struct t8data td = {
                    .in_data = in_data,
                    .out_data = out_data,
                    .reduction_batches = 1,
                    .num_blobs = 1,
                    .blob_size = elemcount
            };
            nn_sem_init(&td.donesem, 0);
            nn_os_work_for_vector(nn, reduce_sum_all_axes_hvx, &td);
            nn_sem_wait(&td.donesem);
        }
        else
        {
            int32_t modified_shape[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
            for (int a = 0; a < axes_size; a++) {
                int32_t axis = axes[a];
                int32_t reduction_batches = 1;

                int32_t blob_size = 1;
                int32_t elemcount = 1;
                for (int i = 0; i < 4; i++) {
                    elemcount *= modified_shape[i];
                }

                if (nn_scratch_grow(nn, elemcount * sizeof(int32_t))){
                    return errlog(nn,"failed to get scratch");
                }

                for (int i = 0; i < axis; i++) {
                    reduction_batches *= modified_shape[i]; // how many groups are reduced to a single value
                }
                for (int i = axis + 1; i < NUM_DIMS; i++) {
                    blob_size *= modified_shape[i]; // tells how many elements are in a group that is reduced to a single value
                }
                if (reduction_batches == 1 && blob_size == 1) {
                    return errlog(nn, "Cannot reduce a single element");
                }

                if (a == 0) { // input is 8bit
                    struct t8data td = {
                            .in_data = in_data,
                            .out_data = out_data,
                            .reduction_batches = reduction_batches,
                            .num_blobs = modified_shape[axis],
                            .blob_size = blob_size
                    };
                    nn_sem_init(&td.donesem, 0);
                    nn_os_work_for_vector(nn, reduce_sum_single_axis_hvx_8, &td);
                    nn_sem_wait(&td.donesem);
                }
                else
                { // input data is 32bit
                    struct t32data td = {
                            .in_data = out_data,
                            .out_data = out_data,
                            .reduction_batches = reduction_batches,
                            .num_blobs = modified_shape[axis],
                            .blob_size = blob_size
                    };
                    nn_sem_init(&td.donesem, 0);
                    nn_os_work_for_vector(nn, reduce_sum_single_axis_hvx_32, &td);
                    nn_sem_wait(&td.donesem);
                }
                modified_shape[axis] = 1;
            }
        }

        // carefully handle min and max to handle going from quint8 with asymetric min and max to
        // qint32 with symetric min and max
        float maxval = tensor_get_float(in_max_tensor, 0);
        float minval = tensor_get_float(in_min_tensor, 0);
        float range = fmaxf(0.0001f,maxval-minval);
        float stepsize = flt_div_255(range);

        uint8_t qzero = saturate_u8(roundf_i32(-minval/stepsize));

        const int32_t input_shape[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
        int32_t elems = 1;
        for (int i = 0; i < axes_size; i++)
        {
            elems *= input_shape[axes[i]];
        }

        int32_t adj = elems * qzero;
        struct tadjdata td = {
                .in_data = out_data,
                .out_data = (int32_t *)out_tensor->data,
                .out_data_size = out_data_size,
                .adj = adj
        };
        nn_sem_init(&td.donesem, 0);
        nn_os_work_for_vector(nn, adjust_output_range, &td);
        nn_sem_wait(&td.donesem);

        float out_level_size = (maxval - minval) / 255;
        float out_max = 2147483648.0f/*0x1.0p31f*/ * out_level_size;
        float out_min = -out_max;

        tensor_set_single_float(out_min_tensor, out_min);
        tensor_set_single_float(out_max_tensor, out_max);

        return 0;
    }

    static int reducing_sum_check(struct nn_node *self, struct nn_graph *nn)
    {
        int k;
        logmsg(nn, 2, "reducing sum node %p", self);
        k = node_check_inputs_outputs_n(self, nn, "reducing sum", 4, 3);
        if (k != 0) return k;
        logmsg(nn, 2, "reducing sum %p check OK", self);
        return 0;
    }

    struct nn_node_ops nn_ops_for_QuantizedSum_8to32 = {
            .execute = reducing_sum_execute,
            .check = reducing_sum_check,
            .ctor = node_alloc_common,
            .dtor = node_free_common,
    };
