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
 * This contains implementations for quantized reduce sum node
 */

#include <nn_graph.h>
#include <quantize.h>
#include <nn_reduce_utils.h>

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

static void reduce_sum_all_axes_hvx_wrapper(struct nn_graph *nn, void *vtd) {
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    reduce_sum_all_axes_hvx(in_data, out_data, blob_size);
    nn_sem_post(&td->donesem);
}

static void reduce_sum_single_axis_hvx_8_wrapper(struct nn_graph *nn, void *vtd) {
    struct t8data *td = vtd;
    uint8_t *in_data = td->in_data;
    int32_t *out_data = (int32_t *)td->out_data;
    const int reduction_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
    reduce_sum_single_axis_hvx_8(in_data, out_data, reduction_batches, num_blobs, blob_size);
    nn_sem_post(&td->donesem);
}

static void reduce_sum_single_axis_hvx_32_wrapper(struct nn_graph *nn, void *vtd) {
    struct t32data *td = vtd;
    int32_t *in_data = td->in_data;
    int32_t *out_data = td->out_data;
    const int reduction_batches = td->reduction_batches;
    const int num_blobs = td->num_blobs;
    const int blob_size = td->blob_size;
    reduce_sum_single_axis_hvx_32(in_data, out_data, reduction_batches, num_blobs, blob_size);
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
            nn_os_work_for_vector(nn, reduce_sum_all_axes_hvx_wrapper, &td);
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

                if (a == 0) { // input is 8bit
                    struct t8data td = {
                            .in_data = in_data,
                            .out_data = out_data,
                            .reduction_batches = reduction_batches,
                            .num_blobs = modified_shape[axis],
                            .blob_size = blob_size
                    };
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
                            .blob_size = blob_size
                    };
                    nn_sem_init(&td.donesem, 0);
                    nn_os_work_for_vector(nn, reduce_sum_single_axis_hvx_32_wrapper, &td);
                    nn_sem_wait(&td.donesem);
                }
                modified_shape[axis] = 1;
            }
        }

        // carefully handle min and max to handle going from quint8 with asymetric min and max to
        // qint32 with symetric min and max
        float maxval = tensor_get_float(in_max_tensor, 0);
        float minval = tensor_get_float(in_min_tensor, 0);
        //maxval = 0;
        adjust_minmax_for_zero(&minval, &maxval);
        //float range = fmaxf(0.0001f,maxval-minval);
        //float stepsize = flt_div_255(range);

        //uint8_t qzero = saturate_u8(roundf_i32(-minval/stepsize));
        uint8_t qzero = quantize_uint8(0.0f, minval, maxval);

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


    struct nn_node_ops nn_ops_for_QuantizedSum_8to32 = {
            .execute = reducing_sum_execute,
            .check = NULL,
            .ctor = node_alloc_common,
            .dtor = node_free_common,
            .n_inputs = NN_IOCOUNT(4),
            .n_outputs = NN_IOCOUNT(3),
    };
