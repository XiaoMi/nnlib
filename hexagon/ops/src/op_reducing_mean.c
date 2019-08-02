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
 * This contains implementations for quantized reduce mean node
 */

#include <nn_graph.h>
#include <quantize.h>
#include <nn_reduce_utils.h>
#include "nn_axis.h"
#include "hvx_inlines.h"

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
    uint8_t *out_data;
    int32_t out_data_size;
    int32_t divisor;
    float in_min;
    float in_max;
    float out_min;
    float out_max;
    nn_sem_t donesem;
};

static void reduce_sum_all_axes_hvx_wrapper(struct nn_graph *nn, void *vtd)
{
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    reduce_sum_all_axes_hvx(in_data, out_data, blob_size);
    nn_sem_post(&td->donesem);
}

static void reduce_sum_single_axis_hvx_8_wrapper(struct nn_graph *nn, void *vtd)
{
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = (int32_t *)td->out_data;
    const int reduction_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
    reduce_sum_single_axis_hvx_8(in_data, out_data, reduction_batches, num_blobs, blob_size);
    nn_sem_post(&td->donesem);
}

static void reduce_sum_single_axis_hvx_32_wrapper(struct nn_graph *nn, void *vtd)
{
    struct t32data *td = vtd;
    int32_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int reduction_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
    reduce_sum_single_axis_hvx_32(in_data, out_data, reduction_batches, num_blobs, blob_size);
    nn_sem_post(&td->donesem);
}

/*Output scaling for mean is done as follows:
1) Inital sum is computed, result is 32 bits
2) This sum is has an addtional num elements_reduced * in_offset term that we need to subtract to obtain a zero based result
3) Multiply by a quantized muliplier with 31 fractional bits (input scale / output scale / num elements)
4) Add output offset
5) Pack to 8 bits
*/

static void set_output_scaling_for_mean(struct nn_graph *nn, void *vtd)
{
    struct tadjdata *td = vtd;
    int32_t *in_data = td->in_data;
    uint8_t *out_data = td->out_data;
    int32_t out_data_size = td->out_data_size;
    int32_t divisor = td->divisor;
    float in_min = td->in_min;
    float in_max = td->in_max;
    float out_min = td->out_min;
    float out_max = td->out_max;

    float in_stepsize = 0.0f;
    float out_stepsize = 0.0f;
    int in_offset = get_qu8_level_size_zero( in_min, in_max, &in_stepsize);
    int out_offset = get_qu8_level_size_zero( out_min, out_max, &out_stepsize);
    int loop_count = out_data_size / sizeof(HVX_Vector); //Number of full vecs to process
    int leftovers = out_data_size - loop_count * sizeof(HVX_Vector);
    int k = ((leftovers + 31) & -31) >> 5;
    float realMultiplier = fminf(0.99999f, (in_stepsize / (out_stepsize * divisor)));
    int32_t outputMultiplier = 0;
    int32_t outputShift = 0;
    if (QuantizeMultiplierSmallerThanOne(realMultiplier, &outputMultiplier, &outputShift) == -1)
    {
        errlog(nn, "Unable to determine quantized mulipiler for output");
    }

    HVX_Vector adjv = Q6_V_vsplat_R(in_offset * divisor);
    HVX_Vector out_offset_vec = Q6_V_vsplat_R(out_offset);
    HVX_Vector *out_data_vec = (HVX_Vector *)(out_data);
    for (int i = 0; i < loop_count; i++)
    {
        HVX_Vector vin = *(HVX_Vector *)in_data;
        in_data += 32;
        HVX_Vector vin1 = *(HVX_Vector *)in_data;
        in_data += 32;
        HVX_Vector vin2 = *(HVX_Vector *)in_data;
        in_data += 32;
        HVX_Vector vin3 = *(HVX_Vector *)in_data;
        in_data += 32;
        HVX_Vector s1 = Q6_Vw_vsub_VwVw_sat(vin, adjv);
        HVX_Vector s2 = Q6_Vw_vsub_VwVw_sat(vin1, adjv);
        HVX_Vector s3 = Q6_Vw_vsub_VwVw_sat(vin2, adjv);
        HVX_Vector s4 = Q6_Vw_vsub_VwVw_sat(vin3, adjv);

        s1 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s1, outputMultiplier, -outputShift), out_offset_vec);
        s2 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s2, outputMultiplier, -outputShift), out_offset_vec);
        s3 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s3, outputMultiplier, -outputShift), out_offset_vec);
        s4 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s4, outputMultiplier, -outputShift), out_offset_vec);
        s4 = Q6_Vh_vpacke_VwVw(s4, s3); // take upper 16 bits.
        s1 = Q6_Vh_vpacke_VwVw(s2, s1);
        *out_data_vec++ = Q6_Vub_vpack_VhVh_sat(s4, s1); // sat to u8
    }
    //Process any leftovers
    {
        HVX_Vector s1 = Q6_V_vzero();
        HVX_Vector s2 = Q6_V_vzero();
        HVX_Vector s3 = Q6_V_vzero();
        HVX_Vector s4 = Q6_V_vzero();

        HVX_Vector vin = *(HVX_Vector *)in_data;
        in_data += 32;
        s1 = Q6_Vw_vsub_VwVw_sat(vin, adjv);
        s1 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s1, outputMultiplier, -outputShift), out_offset_vec);
        if (k != 1)
        {
            HVX_Vector vin1 = *(HVX_Vector *)in_data;
            in_data += 32;
            s2 = Q6_Vw_vsub_VwVw_sat(vin1, adjv);
            s2 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s2, outputMultiplier, -outputShift), out_offset_vec);
            if (k != 2)
            {
                HVX_Vector vin2 = *(HVX_Vector *)in_data;
                in_data += 32;
                s3 = Q6_Vw_vsub_VwVw_sat(vin2, adjv);
                s3 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s3, outputMultiplier, -outputShift), out_offset_vec);
                if (k != 3)
                {
                    HVX_Vector vin3 = *(HVX_Vector *)in_data;
                    s4 = Q6_Vw_vsub_VwVw_sat(vin3, adjv);
                    s4 = Q6_Vw_vadd_VwVw_sat(MultiplyByQuantizedMultiplier(s4, outputMultiplier, -outputShift), out_offset_vec);
                }
            }
        }
        s4 = Q6_Vh_vpacke_VwVw(s4, s3); // take upper 16 bits.
        s1 = Q6_Vh_vpacke_VwVw(s2, s1);
        q6op_vstu_variable_ARV(out_data_vec, leftovers, Q6_Vub_vpack_VhVh_sat(s4, s1)); // sat to u8
    }

    nn_sem_post(&td->donesem);
}

static int reducing_mean_execute(struct nn_node *self, struct nn_graph *nn)
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
    int32_t const *axes_orig = (int32_t const *)axes_tensor->data;
    int32_t axes_size = axes_tensor->data_size / sizeof(int32_t);
    uint8_t *in_data = in_tensor->data;
    int32_t *out_data = nn->scratch;
    int32_t elemcount = in_batches * in_height * in_width * in_depth;

    int32_t modified_shape_final[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
    int32_t axes[NUM_DIMS];

    int32_t modified_data_size = 1;
    if (axes_size > NUM_DIMS)
        return errlog(nn, "Number of elements in axes tensor is %d, support a maximum of 4", axes_size);
    for (int i = 0; i < axes_size; i++)
    {
        axes[i] = axes_orig[i];
    }
    if (handle_negative_axes(nn, axes, axes_size) != 0)
        return -1;

    //Handle negative axis also checks that the axes are sane, so we don't need to check them here
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
        return errlog(nn, "failed to get scratch");
    }
    if (tensor_out_prepare_normal_fromshape(out_tensor, &out_tensor->shape, NN_TYPE_UINT8) != 0)
    {
        return errlog(nn, "out too small");
    }

    if (axes_size == 4 || modified_data_size == 1)
    { // reduce on all dimensions
        struct t8data td = {
            .in_data = in_data,
            .out_data = out_data,
            .reduction_batches = 1,
            .num_blobs = 1,
            .blob_size = elemcount};
        nn_sem_init(&td.donesem, 0);
        nn_os_work_for_vector(nn, reduce_sum_all_axes_hvx_wrapper, &td);
        nn_sem_wait(&td.donesem);
    }
    else
    {
        int32_t modified_shape[NUM_DIMS] = {in_batches, in_height, in_width, in_depth};
        for (int a = 0; a < axes_size; a++)
        {
            int32_t axis = axes[a];
            int32_t reduction_batches = 1;

            int32_t blob_size = 1;
            int32_t elemcount = 1;
            for (int i = 0; i < 4; i++)
            {
                elemcount *= modified_shape[i];
            }

            if (nn_scratch_grow(nn, elemcount * sizeof(int32_t)))
            {
                return errlog(nn, "failed to get scratch");
            }

            for (int i = 0; i < axis; i++)
            {
                reduction_batches *= modified_shape[i]; // how many groups are reduced to a single value
            }
            for (int i = axis + 1; i < NUM_DIMS; i++)
            {
                blob_size *= modified_shape[i]; // tells how many elements are in a group that is reduced to a single value
            }

            if (a == 0)
            { // input is 8bit
                struct t8data td = {
                    .in_data = in_data,
                    .out_data = out_data,
                    .reduction_batches = reduction_batches,
                    .num_blobs = modified_shape[axis],
                    .blob_size = blob_size};
                nn_sem_init(&td.donesem, 0);
                nn_os_work_for_vector(nn, reduce_sum_single_axis_hvx_8_wrapper, &td);
                nn_sem_wait(&td.donesem);
            }
            else
            { // input data is 32bit
                struct t32data td = {
                    .in_data = out_data,
                    .out_data = out_data,
                    .reduction_batches = reduction_batches,
                    .num_blobs = modified_shape[axis],
                    .blob_size = blob_size};
                nn_sem_init(&td.donesem, 0);
                nn_os_work_for_vector(nn, reduce_sum_single_axis_hvx_32_wrapper, &td);
                nn_sem_wait(&td.donesem);
            }
            modified_shape[axis] = 1;
        }
    }
    
    int32_t elements = 1;

    for (int i = 0; i < axes_size; i++)
    {
        elements *= in_tensor->shape.dimension[axes[i]];
    }
    float maxval = tensor_get_float(in_max_tensor, 0);
    float minval = tensor_get_float(in_min_tensor, 0);
    adjust_minmax_for_zero(&minval, &maxval);
    float out_min = minval;
    float out_max = maxval;
    if (self->n_inputs >= 5)
    {
        out_min = tensor_get_float(self->inputs[4], 0);
    }
    if (self->n_inputs == 6)
    {
        out_max = tensor_get_float(self->inputs[5], 0);
    }
    struct tadjdata td = {
        .in_data = out_data,
        .out_data = out_tensor->data,
        .out_data_size = out_data_size,
        .divisor = elements,
        .in_min = minval,
        .in_max = maxval,
        .out_min = out_min,
        .out_max = out_max};

    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, set_output_scaling_for_mean, &td);
    nn_sem_wait(&td.donesem);

    tensor_set_single_float(out_min_tensor, out_min);
    tensor_set_single_float(out_max_tensor, out_max);
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMean_8 = {
    .execute = reducing_mean_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(4, 6),
    .n_outputs = NN_IOCOUNT(3),
};
