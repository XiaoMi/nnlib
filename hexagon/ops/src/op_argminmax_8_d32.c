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

//  ArgMinMax_8_d32
//  Returns the indices of the minimum/maximum values along an axis.
//  4 inputs:
//      0   input data       (uint8_t)
//      1   axis             (int32_t)
//      2   input min val    (scalar float)
//      3   input max val    (scalar float)
//  1 output:
//      0   output data      (int32_t)

#include <stdlib.h>
#include <nn_graph.h>
#include <quantize.h>
#include "nn_gentranspose.h"
#include "nn_axis.h"

#ifdef HEXAGON_V66
#define ARGMAX_MAX_THREADS 4
#else
#define ARGMAX_MAX_THREADS 2
#endif

#define OP_ARGMINMAX_INPUT_NUM 4
#define OP_ARGMINMAX_OUTPUT_NUM 1

#define AXIS_DEPTH_IDX   3

#define INPUT_DATA_IDX 0
#define INPUT_AXIS_IDX 1

#define OUTPUT_DATA_IDX 0

struct argminmax_8_d32_runstate {
    struct argminmax_8_d32_info * info;
    nn_sem_t done_sem;
};

struct argminmax_8_d32_info {
    int32_t * out_data;
    struct tensor const * in_data_tensor;
    int32_t axis;
    int32_t find_max; //0 - find argmin; 1 - find argmax
    int threads_num;
    int thread_id;
};

static void find_argminmax_8_d32(struct nn_graph *nn,  void * rstpv) {

    struct argminmax_8_d32_runstate *rstp = (struct argminmax_8_d32_runstate *)rstpv;
    struct argminmax_8_d32_info const * info = rstp->info;

    struct tensor  const * in_data_tensor = info->in_data_tensor;
    int32_t * out_data = info->out_data;
    int32_t find_max = info->find_max;
    int32_t  axis = info->axis;
    int threads_num = info->threads_num;
    int thread_id = info->thread_id;

    if( AXIS_DEPTH_IDX == axis ) {
        hvx_argmin_or_max_d_8_d32_mt(in_data_tensor, out_data, find_max, threads_num, thread_id);
    }else {
        hvx_argmin_or_max_whb_8_d32(in_data_tensor,  out_data, axis, find_max);
    }
    nn_sem_post(& rstp->done_sem);
}

static int argminmax_8_d32_execute(struct nn_node *self, struct nn_graph *nn, int32_t find_max) {

    const struct tensor *in_data_tensor = self->inputs[INPUT_DATA_IDX];
    const struct tensor *in_axis_tensor = self->inputs[INPUT_AXIS_IDX];
    struct tensor *out_data_tensor = self->outputs[OUTPUT_DATA_IDX];

    int32_t *out_data = out_data_tensor->data;

    int32_t in_axis = tensor_get_int32(in_axis_tensor,0);
    int res = handle_negative_axes(nn, &in_axis, 1);
    if (res)
        return errlog(nn, "argminmax_8_d32: axis is out of range \n");

    const struct shape in_shape = in_data_tensor->shape;
    struct shape outshape = in_shape;
    outshape.dimension[in_axis] = 1;

    if (tensor_out_prepare_normal_fromshape(out_data_tensor, &outshape, NN_TYPE_INT32) !=0) return errlog(nn,"argminmax_8_d32 out too small");

    struct argminmax_8_d32_info info[ARGMAX_MAX_THREADS];
    struct argminmax_8_d32_runstate rst[ARGMAX_MAX_THREADS];
    int threads_num = 1; // single thread by default

    if (AXIS_DEPTH_IDX == in_axis) {
        // Enabled multi-thread for argmax on depth
        int dim_b = in_data_tensor->shape.batches;
        int dim_w = in_data_tensor->shape.width;
        int dim_h = in_data_tensor->shape.height;

        int dim_w_aligned = (dim_w+3) & ~0x03; //round up dim_w to multiple of 4
        int rows = dim_b * dim_h * dim_w_aligned; //number of rows with paddings

        threads_num = min_i32(rows/8, ARGMAX_MAX_THREADS);
        if (threads_num < 1) {
            threads_num = 1;
        }
    }

    for(int i = 0; i < threads_num; i++) {
        info[i].out_data = out_data;
        info[i].in_data_tensor = in_data_tensor;
        info[i].find_max = find_max;
        info[i].axis = in_axis;
        info[i].threads_num = threads_num;
        info[i].thread_id = i;

        rst[i].info = &info[i];
        nn_sem_init(&rst[i].done_sem, 0);
        nn_os_work_for_vector(nn, find_argminmax_8_d32, &rst[i]);
    }

    for(int i = 0; i < threads_num; i++) {
        nn_sem_wait(&rst[i].done_sem);
    }

    return 0;
}

static int argmax_8_d32_execute(struct nn_node *self, struct nn_graph *nn) {
    return argminmax_8_d32_execute(self, nn, 1);
}

static int argmin_8_d32_execute(struct nn_node *self, struct nn_graph *nn) {
    return argminmax_8_d32_execute(self, nn, 0);
}

struct nn_node_ops nn_ops_for_ArgMax_8_d32 = {

    .execute = argmax_8_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ARGMINMAX_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_ARGMINMAX_OUTPUT_NUM),
};

struct nn_node_ops nn_ops_for_ArgMin_8_d32 = {

    .execute = argmin_8_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_ARGMINMAX_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_ARGMINMAX_OUTPUT_NUM),
};

