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

//  Group and transpose along a given axis
//
//  The operation is equivalent to doing the following (e.g. axis is w)
//  [b,h,wa*wb,d] -> [b,h,wa,wb,d]  (reshape)
//  [b,h,wb,wa,d]                   (transpose)
//  [b,h,wb*wa,d]                   (reshape)
//
//  5 inputs:
//      0: input data       (uint8_t) Need to set up the input shape
//      1: axis             (int32_t)
//      2: number of groups (int32_t)
//      3: input min val    (scalar float)
//      4: input max val    (scalar float)
//  3 output:
//      0: output data      (uint8_t)
//      1: output min val   (scalar float - same as input min)
//      2: output max val   (scalar float - same as input max)
//

#include <stdlib.h>
#include <nn_graph.h>
#include "nn_gentranspose.h"
#include "nn_axis.h"

#define OP_AXISSHUFFLE_INPUT_NUM 5
#define OP_AXISSHUFFLE_OUTPUT_NUM 3
#define DIM_NUM 4                   //number of dimensions

#define INPUT_DATA_IDX 0
#define INPUT_AXIS_IDX 1
#define INPUT_NUMGROUP_IDX 2
#define INPUT_DATA_MIN 3
#define INPUT_DATA_MAX 4

#define OUTPUT_DATA_IDX 0
#define OUTPUT_DATA_MIN 1
#define OUTPUT_DATA_MAX 2

static int axisshuffle_execute_8(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_data_tensor = self->inputs[INPUT_DATA_IDX];
    const struct tensor *in_axis_tensor = self->inputs[INPUT_AXIS_IDX];
    const struct tensor *in_numGroup_tensor = self->inputs[INPUT_NUMGROUP_IDX];
    const struct tensor *in_data_min_tensor = self->inputs[INPUT_DATA_MIN];
    const struct tensor *in_data_max_tensor = self->inputs[INPUT_DATA_MAX];

    struct tensor *out_tensor = self->outputs[OUTPUT_DATA_IDX];
    struct tensor *out_min_tensor = self->outputs[OUTPUT_DATA_MIN];
    struct tensor *out_max_tensor = self->outputs[OUTPUT_DATA_MAX];

    const int32_t in_numGroup = tensor_get_int32(in_numGroup_tensor,0);
    int32_t in_axis = tensor_get_int32(in_axis_tensor,0);
    const struct shape in_shape = in_data_tensor->shape;

    in_axis = handle_negative_axis(in_axis);
    if (-1 == in_axis) return errlog(nn, "AxisShuffle: axis is out of range \n");

    if (in_shape.dimension[in_axis]%in_numGroup != 0 ) return errlog(nn, "AxisShuffle: cannot group elements along the axis \n");
    if (tensor_out_prepare_normal_fromshape(out_tensor, &in_shape, NN_TYPE_QINT8) !=0) return errlog(nn,"out too small");

    struct nn_transpose_desc txdesc;
    //reshape
    uint32_t temp_in_dim[DIM_NUM+1];
    int32_t perm_arr[DIM_NUM+1];

    uint32_t num_elems_per_group = in_shape.dimension[in_axis]/in_numGroup;
    uint32_t temp_dim_idx = 0;
    for (uint32_t i = 0; i < DIM_NUM; ++i, ++temp_dim_idx) {
        if(i == in_axis) {
            temp_in_dim[temp_dim_idx] = in_numGroup;
            temp_in_dim[temp_dim_idx+1] = num_elems_per_group;

            perm_arr[temp_dim_idx] = temp_dim_idx+1;
            perm_arr[temp_dim_idx+1] = temp_dim_idx;
            ++temp_dim_idx;
            continue;
        }
        temp_in_dim[temp_dim_idx] = in_shape.dimension[i];
        perm_arr[temp_dim_idx] = temp_dim_idx;
    }

    //transpose
    int res;
    res = nn_transpose_analyze_direct( &txdesc,sizeof(uint8_t), perm_arr, DIM_NUM+1, temp_in_dim, DIM_NUM+1 );
    if(res) return errlog( nn,"AxisShuffle: transpose analyze error %d", res);
    nn_scratch_grow(nn, txdesc.buffer_needed);
    res = nn_transpose_execute( nn, &txdesc, nn->scratch, (uint8_t*)(out_tensor->data), (uint8_t const*)(in_data_tensor->data));
    if(res) return errlog( nn, "AxisShuffle: transpose exec error %d", res);
    *(float*)(out_min_tensor->data) = *(float*)(in_data_min_tensor->data);
    *(float*)(out_max_tensor->data) = *(float*)(in_data_max_tensor->data);

    return 0;
}

static int axisshuffle_check_8(struct nn_node *self, struct nn_graph *nn)  {

    logmsg(nn,2,"Checking AxisShuffle node %p",self);
    if (self->n_inputs != OP_AXISSHUFFLE_INPUT_NUM) return errlog(nn,"AxisShuffle check: wrong # inputs");
    if (self->n_outputs != OP_AXISSHUFFLE_OUTPUT_NUM) return errlog(nn,"AxisShuffle check: wrong # outputs");
    logmsg(nn,2,"AxisShuffle node %p check OK",self);
    return 0;
}

struct nn_node_ops nn_ops_for_AxisShuffle_8 = {
    .execute = axisshuffle_execute_8,
    .check = axisshuffle_check_8,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

