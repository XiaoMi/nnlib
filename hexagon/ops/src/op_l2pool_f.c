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
#include <math.h>

static int l2pool_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor     = self->inputs[0];
    const struct tensor *window_tensor = self->inputs[1];
    const struct tensor *stride_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];

    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_width   = in_tensor->shape.width;
    int32_t in_height  = in_tensor->shape.height;
    int32_t in_depth   = in_tensor->shape.depth;

    int32_t stride_width  = stride_tensor->shape.width;
    int32_t stride_height = stride_tensor->shape.height;

    int32_t window_height = window_tensor->shape.height;
    int32_t window_width  = window_tensor->shape.width;

    int32_t out_batches = in_batches;
    int32_t adj_x, adj_y;
    int32_t out_width   = nn_pad_compute_outsize_and_padbefore(in_width, window_width, stride_width, self->padding, & adj_x);
    int32_t out_height  = nn_pad_compute_outsize_and_padbefore(in_height,window_height,stride_height,self->padding, & adj_y);
    int32_t out_depth   = in_depth;

    int32_t batch, out_x, out_y, out_z, in_x, in_y, in_z;

    const float *in = in_tensor->data;
    float *out = out_tensor->data;

    /* check size of output */
    logmsg(nn,2,"fp l2pool execute. self=%p ",self);
    if (  (window_tensor->shape.batches != 1)
       || (window_tensor->shape.depth   != 1)
       || (stride_tensor->shape.batches != 1)
       || (stride_tensor->shape.depth   != 1)) {
        return errlog(nn,"bad window/stride shape");
    }
    if( tensor_out_prepare_normal( out_tensor, out_batches,out_height,out_width,out_depth, NN_TYPE_FLOAT)!=0){
    	return errlog(nn,"out too small");
    }
    if (self->padding == NN_PAD_NA  ) return errlog(nn,"This op might pad");

    for (batch = 0; batch < out_batches; batch++) {
        for (out_y = 0; out_y < out_height; out_y++) {
            int32_t start_y = out_y * stride_height - adj_y;
            int32_t end_y = start_y + window_height;
            if (start_y < 0) start_y = 0;
            if (end_y > in_height) end_y = in_height;
       
            for (out_x = 0; out_x < out_width; out_x++) {
                int32_t start_x = out_x * stride_width - adj_x;
                int32_t end_x   = start_x + window_width;
                if (start_x < 0) start_x = 0;
                if (end_x > in_width) end_x = in_width;
                float count = (end_x-start_x)*(end_y-start_y);
                for (out_z = 0; out_z < out_depth; out_z++) {
                    float sum = 0.0f;
                    in_z = out_z;
            
                    for (in_y = start_y; in_y < end_y; in_y++) {
                        for (in_x = start_x; in_x < end_x; in_x++) {
                            float data = in[in_z + in_depth * (in_x + in_width * ( in_y + in_height * (batch)))];
                            sum += data * data;
                        }
                    }
                    out[out_z + out_depth * (out_x + out_width * (out_y + out_height * (batch)))] = sqrtf(sum/count);
                }
            }
        }
    }
    logmsg(nn,2,"l2pool %p done",self);
    return 0;
}

struct nn_node_ops nn_ops_for_L2Pool_f = {
    .execute = l2pool_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(3),
    .n_outputs = NN_IOCOUNT(1),
};


