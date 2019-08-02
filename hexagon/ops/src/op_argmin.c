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

//  ArgMin_8:
//  Returns the indices of the minimum values along an axis.
//  4 inputs:
//      0   input data       (uint8_t)
//      1   axis             (int32_t)
//      2   input min val    (scalar float)
//      3   input max val    (scalar float)
//  1 output:
//      0   output data      (int32_t)

#include <stdlib.h>
#include <nn_graph.h>
#include "nn_gentranspose.h"
#include "nn_axis.h"

#define OP_ARGMIN_INPUT_NUM 4
#define OP_ARGMIN_OUTPUT_NUM 1

#define AXIS_DEPTH_IDX   3

#define INPUT_DATA_IDX 0
#define INPUT_AXIS_IDX 1

#define OUTPUT_DATA_IDX 0

struct argmin_runstate {
    struct argmin_info * info;
    nn_sem_t done_sem;
};

struct argmin_info {
    int32_t outer_size;
    int32_t dim_value;
    int32_t data_stride;
    int32_t * out_data;
    uint8_t const * in_data;
};

static void find_argmin(struct nn_graph *nn,  void * rstpv) {

    struct argmin_runstate *rstp = (struct argmin_runstate *)rstpv;
    struct argmin_info const * info = rstp->info;

    int32_t outer_size = info->outer_size;
    int32_t dim_value = info->dim_value;
    int32_t data_stride = info->data_stride;
    uint8_t const * in_data = info->in_data;
    int32_t * out_data = info->out_data;

    if( 1 == data_stride ) {
        hvx_argmin_or_max_in_rows(in_data, outer_size, dim_value, dim_value, out_data, 0);

    }else {

        for(int32_t i = 0; i < outer_size; ++i) {

            int32_t in_data_p_increment = dim_value * data_stride;
            int32_t out_data_p_increment = data_stride;

            hvx_argmin_or_max_in_cols(in_data+i*in_data_p_increment, dim_value, data_stride, data_stride, out_data+i*out_data_p_increment, 0);
            //errlog(nn, "outdata %d %d", i, *(out_data+i*out_data_p_increment));
        }
    }
    nn_sem_post(& rstp->done_sem);
}

static int argmin_execute_8(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_data_tensor = self->inputs[INPUT_DATA_IDX];
    const struct tensor *in_axis_tensor = self->inputs[INPUT_AXIS_IDX];
    struct tensor *out_data_tensor = self->outputs[OUTPUT_DATA_IDX];

    const uint8_t *in_data = in_data_tensor->data;
    int32_t *out_data = out_data_tensor->data;
    const struct shape in_shape = in_data_tensor->shape;

    int32_t in_axis0 = tensor_get_int32(in_axis_tensor,0);
    int res = handle_negative_axes(nn, &in_axis0, 1);
    if (res)
         return errlog(nn, "ArgMin_8: axis is out of range \n");
    int32_t in_axis = in_axis0;

    struct shape outshape = in_shape;
    outshape.dimension[in_axis] = 1;
    if (tensor_out_prepare_normal_fromshape(out_data_tensor, &outshape, NN_TYPE_INT32) !=0) return errlog(nn,"ArgMin_8 out too small");

    int stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > in_axis; i--)
    {
        stride *= in_shape.dimension[i];
    }

    int outer_size = get_number_elements_between_axes (in_shape, 0, in_axis);
    if(-1 == outer_size) return errlog(nn, "ArgMin_8: axes are invalid \n");
    int dim_value = in_shape.dimension[in_axis];

    struct argmin_info info;
    info.outer_size = outer_size;
    info.dim_value = dim_value;
    info.data_stride = stride;
    info.out_data = out_data;
    info.in_data = in_data;

    struct argmin_runstate rst;
    rst.info = &info;

    nn_sem_init( &rst.done_sem, 0);
    nn_os_work_for_vector(nn,  find_argmin, &rst );
    nn_sem_wait( & rst.done_sem);

    return 0;
}

struct nn_node_ops nn_ops_for_ArgMin_8 = {
    .execute = argmin_execute_8,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ARGMIN_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_ARGMIN_OUTPUT_NUM),
};

