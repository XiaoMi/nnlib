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

//  L2Normalize reference op
//  Independently normalizes each 1-D slice along the given axis. Axis is depth by default
//  for each axis slice, output = x / sqrt(sum(x*x))
//  the op quantizes the output tensor to [-1,1]
//  3..4 inputs:
//      0           input data       (uint8_t)
//      1           input min val    (scalar float)
//      2           input max val    (scalar float)
//      3(optional) input axis index (int32_t)
//  3 output:
//      0   output data      (uint8_t)
//      1   onput min val    (scalar float)
//      2   onput max val    (scalar float)

#include <nn_graph.h>
#include <math.h>
#include "nn_axis.h"
#include "quantize.h"

#define OP_L2Norm_IN_NUM    3
#define OP_L2Norm_IN_NUM_WITH_OPTIONAL    4
#define OP_L2Norm_OUT_NUM   3
#define AXIS_DEPTH_IDX      3
#define IN_DATA_IDX         0
#define IN_MIN_IDX          1
#define IN_MAX_IDX          2
#define IN_AXIS_IDX         3
#define OUT_DATA_IDX        0
#define OUT_MIN_IDX         1
#define OUT_MAX_IDX         2
#define OUT_MIN             -1
#define OUT_MAX             1

static void l2_norm_along_axis(const uint8_t* in_data, uint8_t in_0_offset, int dim_val, int stride, uint8_t* out_data,
                               float out_min, float out_max) {
    float squared_l2_norm = 0;
    float val = 0;

    int tmp_idx = 0;
    for (int d = 0; d < dim_val; ++d) {
        val = in_data[tmp_idx] - in_0_offset;   //in_data points to the head of the current slice
        squared_l2_norm += val * val;
        tmp_idx += stride;
    }
    const float l2_norm = sqrt(squared_l2_norm);
    float tmp_out = 0;
    tmp_idx = 0;
    for (int d = 0; d < dim_val; ++d) {

        if(0.0 == l2_norm) {
            tmp_out = 0;
        }else {
            tmp_out = (in_data[tmp_idx] - in_0_offset) / l2_norm;
        }
        out_data[tmp_idx] = quantize_uint8(tmp_out, out_min, out_max);
        tmp_idx += stride;
    }
}

static int l2normalize_execute_8_ref(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_data_tensor = self->inputs[IN_DATA_IDX];
    const struct tensor *in_min_tensor = self->inputs[IN_MIN_IDX];
    const struct tensor *in_max_tensor = self->inputs[IN_MAX_IDX];
    struct tensor *out_data_tensor = self->outputs[OUT_DATA_IDX];
    struct tensor *out_min_tensor = self->outputs[OUT_MIN_IDX];
    struct tensor *out_max_tensor = self->outputs[OUT_MAX_IDX];
    const struct shape in_shape = in_data_tensor->shape;

    const uint8_t* in_data = in_data_tensor->data;
    const float in_min = tensor_get_float(in_min_tensor,0);
    const float in_max = tensor_get_float(in_max_tensor,0);

    uint8_t* out_data = out_data_tensor->data;

    int32_t in_axis = AXIS_DEPTH_IDX;
    if (self->n_inputs == OP_L2Norm_IN_NUM_WITH_OPTIONAL) {
        in_axis = tensor_get_int32(self->inputs[IN_AXIS_IDX],0);
        handle_negative_axes(nn, &in_axis, 1);
    }

    uint32_t data_size =  in_data_tensor->data_size;    //1 byte per unit

    if (tensor_out_prepare_normal_fromshape(out_data_tensor, &in_shape, NN_TYPE_UINT8) !=0) return errlog(nn,"op_l2normalize out too small");

    int inner_stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > in_axis; i--)
    {
        inner_stride *= in_shape.dimension[i];
    }

    int dim_val = in_shape.dimension[in_axis];
    uint8_t in_0_offset = quantize_uint8(0.0f, in_min, in_max); //quantized val of 0.0f
    float out_min = OUT_MIN;
    float out_max = OUT_MAX;

    int tmp_outer_idx = 0;
    int count = 0;
    int outer_increment = (dim_val - 1) * inner_stride;
    while (tmp_outer_idx < data_size) {
        if(count < inner_stride) {
            l2_norm_along_axis(in_data+tmp_outer_idx, in_0_offset, dim_val, inner_stride, out_data + tmp_outer_idx, out_min, out_max);
            ++count;
            ++tmp_outer_idx;
        }
        else {
            tmp_outer_idx += outer_increment;
            count = 0;
        }
    }

    tensor_set_single_float( out_min_tensor, out_min);
    tensor_set_single_float( out_max_tensor, out_max);

    return 0;
}

struct nn_node_ops nn_ops_for_L2Normalize_8_ref = {
    .execute = l2normalize_execute_8_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(OP_L2Norm_IN_NUM, OP_L2Norm_IN_NUM_WITH_OPTIONAL),
    .n_outputs = NN_IOCOUNT(OP_L2Norm_OUT_NUM),
};
