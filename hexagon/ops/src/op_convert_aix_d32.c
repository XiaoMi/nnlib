
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
#include <nn_graph.h>
#include <string.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a code to convert to/from aix d32 and dsp d32 format
 * Convert to aix d32 should be considered broken (for now) until there's a good way to test it
 * There is a couple unit tests for it, but these may not match what is passed down to the op
 * Should be easy to refactor these into one function, but leaving it seperate for now until the above works
 */

struct tdata
{
    struct nn_node *self;
    int whoami;
    nn_sem_t donesem;
    int start_batch;
    int end_batch;
    const struct tensor *in_tensor;
    const struct tensor *out_tensor;
    int in_height;
    int out_height;
    int in_width;
    int out_width;
    int in_depth;
    int out_depth;
};

//CONVERSION TO AIX D32 UNTESTED.
static void convert_to_aix_d32_work(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    int start_batch = td->start_batch;
    int end_batch = td->end_batch;
    int out_height = td->out_height;
    int in_width = td->in_width;
    int out_width = td->out_width;
    int out_depth = td->out_depth;
    int depth_padding = 32;
    int in_nd32 = out_depth / depth_padding;
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    int height_stride = tensor_d32_stride_d32(in_tensor);
    int out_height_stride = depth_padding * out_width * in_nd32;
    uint8_t *out_data = out_tensor->data;
    for (int b = start_batch; b < end_batch; b++)
    {
        for (int h = 0; h < out_height; h++)
        {
            uint8_t *start = tensor_location_d32(in_tensor, b, h, 0, 0);
            vmemcpy_2d_general_asm(
                depth_padding * out_width,
                in_nd32,
                out_data + h * out_height_stride,
                depth_padding * in_width,
                start,
                height_stride);
        }
    }
    nn_sem_post(&td->donesem);
}

static void convert_from_aix_d32_work(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    int in_height = td->in_height;
    int in_width = td->in_width;
    int out_width = td->out_width;
    int out_depth = td->out_depth;
    int depth_padding = 32;
    int in_nd32 = out_depth / depth_padding;
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    int height_stride = depth_padding * in_width;
    uint8_t *in_data = in_tensor->data;

    for (int h = 0; h < in_height; h++)
    {
        uint8_t *start = in_data + h * height_stride * in_nd32;
        vmemcpy_2d_general_asm(
            depth_padding * in_width,
            in_nd32,
            tensor_location_d32(out_tensor, 0, h, 0, 0),
            depth_padding * out_width,
            start,
            height_stride);
    }
    nn_sem_post(&td->donesem);
}

//CONVERSION TO AIX D32 UNTESTED.
static int convert_to_aix_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *padding_tensor = self->inputs[3];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_depth = in_tensor->shape.depth;
    int32_t out_batches = out_tensor->shape.batches;
    int32_t out_height = out_tensor->shape.height;
    int32_t out_width = out_tensor->shape.width;
    int32_t out_depth = out_tensor->shape.depth;

    const int32_t b_before = tensor_get_int32(padding_tensor, 0);
    const int32_t b_after = tensor_get_int32(padding_tensor, 1);
    const int32_t h_before = tensor_get_int32(padding_tensor, 2);
    const int32_t h_after = tensor_get_int32(padding_tensor, 3);
    const int32_t w_before = tensor_get_int32(padding_tensor, 4);
    const int32_t w_after = tensor_get_int32(padding_tensor, 5);
    const int32_t d_before = tensor_get_int32(padding_tensor, 6);
    const int32_t d_after = tensor_get_int32(padding_tensor, 7);

    const int32_t b_total = b_before + out_batches + b_after;
    const int32_t h_total = h_before + out_height + h_after;
    const int32_t w_total = w_before + out_width + w_after;
    const int32_t d_total = d_before + out_depth + d_after;
    tensor_set_shape(out_tensor, b_total, h_total, w_total, d_total);
    out_tensor->data_size = b_total * h_total * w_total * d_total;
    if (out_tensor->data_size > out_tensor->max_size)
        return errlog(nn, "output tensor exceeds max data size");
    struct tdata td = {
        .in_tensor = in_tensor,
        .out_tensor = out_tensor,
        .start_batch = 0,
        .end_batch = in_batches,
        .in_height = in_height,
        .out_height = out_height,
        .in_width = in_width,
        .out_width = out_width,
        .in_depth = in_depth,
        .out_depth = out_depth,
    };
    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, convert_to_aix_d32_work, &td);
    nn_sem_wait(&td.donesem);
    tensor_copy(out_min_tensor, in_min_tensor);
    tensor_copy(out_max_tensor, in_max_tensor);
    return 0;
}

static int convert_from_aix_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
    struct tensor const *in_tensor = &nn->inputs[0];
    const struct tensor *in_min_tensor = &nn->inputs[1];
    const struct tensor *in_max_tensor = &nn->inputs[2];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_depth = in_tensor->shape.depth;

    const int b = in_batches;
    const int h = in_height;
    const int w = in_width;
    const int d = in_depth;

    const int h_before_pad = 4;
    const int h_after_pad = 4;
    const int w_min = 0;
    const int w_before_pad = 4;
    const int w_total = (w_before_pad + w + w_min + 3) & ~3;
    const int w_after_pad = w_total - (w_before_pad + w);
    const int d_before_pad = 0;
    const int d_after_pad = (-(d + d_before_pad)) & 31;
    if (tensor_out_prepare_padded_d32(
            out_tensor,
            b,
            h, h_before_pad, h_after_pad,
            w, w_before_pad, w_after_pad,
            d, d_before_pad, d_after_pad,
            NN_TYPE_QUINT8) != 0)
    {
        return errlog(nn, "out prepare fail");
    }
    struct tdata td = {
        .in_tensor = in_tensor,
        .out_tensor = out_tensor,
        .start_batch = 0,
        .end_batch = in_batches,
        .in_height = in_height,
        .out_height = h_before_pad + h + h_after_pad,
        .in_width = in_width,
        .out_width = w_before_pad + w + w_after_pad,
        .in_depth = in_depth,
        .out_depth = d_before_pad + d + d_after_pad,
    };
    nn_sem_init(&td.donesem, 0);
    nn_os_work_for_vector(nn, convert_from_aix_d32_work, &td);
    nn_sem_wait(&td.donesem);

    tensor_copy(out_min_tensor, in_min_tensor);
    tensor_copy(out_max_tensor, in_max_tensor);
    return 0;
}

static int convert_aix_d32_check(struct nn_node *self, struct nn_graph *nn)
{
    int k;
    logmsg(nn, 2, "Convert aix d32 check %p", self);
    k = node_check_inputs_outputs_n(self, nn, "convert aix d32 op", 1, 3);
    if (k != 0)
        return k;
    logmsg(nn, 2, "convert aix d32 op %p check OK", self);
    return 0;
}

struct nn_node_ops nn_ops_for_Convert_to_aix_d32 = {
    .execute = convert_to_aix_d32_execute,
    .check = convert_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .flags = NN_NODE_FLAG_D32_INPUT};

//This is unsued
struct nn_node_ops nn_ops_for_Convert_from_aix = {
    .execute = convert_from_aix_d32_execute,
    .check = convert_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Convert_from_aix_d32 = {
    .execute = convert_from_aix_d32_execute,
    .check = convert_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .flags = NN_NODE_FLAG_D32_OUTPUT};
    
