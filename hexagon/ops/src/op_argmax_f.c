/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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


#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include "nn_axis.h"

#define AXIS_DEPTH_IDX   3

static inline int maxIdx_ftoInt32 (const float *src, int dimSize, int stride)
{
    float maxVal = *src;
    src += stride;
    int maxIdx = 0;
    for (int i = 1; i < dimSize; i++,src+=stride) {
        if (*src > maxVal) {
            maxVal = *src;
            maxIdx = i;
        }
    }
    return maxIdx;
}

static int argmax_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *axis_tensor = self->inputs[1];
    struct tensor *out_tensor = self->outputs[0];

    int32_t axis_idx0 = tensor_get_int32(axis_tensor,0);
    int res = handle_negative_axes(nn, &axis_idx0, 1);
    if (res)
        return errlog(nn, "ArgMax_ftoInt32: axis is out of range \n");
    int axis_idx = axis_idx0;

    const struct shape inshape = in_tensor->shape;
    struct shape outshape = in_tensor->shape;
    outshape.dimension[axis_idx] = 1;

    int dim = inshape.dimension[axis_idx];
    int stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > axis_idx; i--)
    {
        stride *= inshape.dimension[i];
    }

    if (tensor_out_prepare_normal_fromshape(out_tensor,&outshape,NN_TYPE_INT32) != 0) {
        return errlog(nn,"failed to prepare output");
    }

    int inputIdx = 0;
    int counter = 0;
    int *out_data = out_tensor->data;

    const float *in_base = in_tensor->data;
    int dataLength = in_tensor->data_size/sizeof(float);
    while (inputIdx < dataLength) {
        if (counter < stride) {
            *out_data++ = maxIdx_ftoInt32(in_base+inputIdx,dim,stride);
            counter++;
            inputIdx++;
        } else {
            counter = 0;
            inputIdx += (dim - 1) * stride;
        }
    }

    return 0;
}


struct nn_node_ops nn_ops_for_ArgMax_ftoInt32 = {
    .execute = argmax_execute_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(2),
    .n_outputs = NN_IOCOUNT(1),
};
