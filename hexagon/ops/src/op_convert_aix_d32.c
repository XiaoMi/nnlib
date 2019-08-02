
/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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

static void convert_to_aix_d32_work(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    const int start_batch = td->start_batch;
    const int end_batch = td->end_batch;
    const int out_height = td->out_height;
    const int out_width = td->out_width;
    const int out_depth = td->out_depth;
    const int depth_padding = 32;
    const int in_nd32 = out_depth / depth_padding;
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    const int height_stride = tensor_d32_stride_d32(in_tensor);
    const int out_height_stride = depth_padding * out_width;
    const int src_stride = tensor_row_stride_d32(in_tensor);
    const int dst_stride = out_height_stride * in_nd32;
    uint8_t *out_data = out_tensor->data;
    uint8_t *in_data = tensor_location_d32(in_tensor, 0, 0, 0, 0);
    l2fetch(in_data, 128, 128, (out_height * height_stride + 127) / 128u);
    const int rows = (in_nd32 == 1) ? 1 : out_height;
    const int row_bytes = (in_nd32 == 1) ? out_height : in_nd32;
    for (int b = start_batch; b < end_batch; b++)
    {
        in_data = tensor_location_d32(in_tensor, b, 0, 0, 0);
        for (int h = 0; h < rows; h++)
        {
            if (out_width % 4 == 0)
            {
                vmemcpy_2d_asm(
                    depth_padding * out_width,
                    row_bytes,
                    out_data,
                    out_height_stride,
                    in_data,
                    height_stride);
            }
            else
            {
                vmemcpy_2d_general_asm(
                    depth_padding * out_width,
                    row_bytes,
                    out_data,
                    out_height_stride,
                    in_data,
                    height_stride);
            }
            out_data += dst_stride;
            in_data += src_stride;
        }
    }
    nn_sem_post(&td->donesem);
}

static int convert_to_aix_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    int32_t need_convert = tensor_get_int32(self->inputs[3], 0);
    //struct shape in_shape = in_tensor->shape;
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    if (tensor_out_prepare_normal_fromshape(out_tensor, &out_tensor->shape, NN_TYPE_UINT8) != 0)
    {
        return errlog(nn, "out too small");
    }
    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_height = in_tensor->shape.height;
    int32_t in_width = in_tensor->shape.width;
    int32_t in_depth = in_tensor->shape.depth;
    struct shape out_shape = out_tensor->shape;
    int32_t out_height = out_shape.height;
    int32_t out_width = out_shape.width;
    int32_t out_depth = out_shape.depth;

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
    if (need_convert) //Tensor is d32, run d32->aix d32 convert
    {
        nn_os_work_for_vector(nn, convert_to_aix_d32_work, &td);
    }
    else
    {
        return errlog(nn, "Converting from unknown data format.");
    }
    nn_sem_wait(&td.donesem);
    tensor_copy(out_min_tensor, in_min_tensor);
    tensor_copy(out_max_tensor, in_max_tensor);
    return 0;
}

static void convert_from_aix_d32_work(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    const int start_batch = td->start_batch;
    const int end_batch = td->end_batch;
    const int in_height = td->in_height;
    const int in_width = td->in_width;
    const int out_width = td->out_width;
    const int out_depth = td->out_depth;
    const int depth_padding = 32;
    const int in_nd32 = out_depth / depth_padding;
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    int height_stride = depth_padding * in_width;
    const int src_stride = height_stride * in_nd32;
    const int dst_stride = tensor_row_stride_d32(out_tensor);
    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    l2fetch(in_data, 128, 128, (in_height * height_stride + 127) / 128u);
    const int rows = (in_nd32 == 1) ? 1 : in_height;
    const int row_bytes = (in_nd32 == 1) ? in_height : in_nd32;
    for (int b = start_batch; b < end_batch; b++)
    {
        out_data = tensor_location_d32(out_tensor, b, 0, 0, 0);
        for (int h = 0; h < rows; h++)
        {
            if (in_width % 4 == 0)
            {
                vmemcpy_2d_asm(
                    depth_padding * in_width,
                    row_bytes,
                    out_data,
                    depth_padding * out_width,
                    in_data,
                    height_stride);
            }
            else
            {
                vmemcpy_2d_general_asm(
                    depth_padding * in_width,
                    row_bytes,
                    out_data,
                    depth_padding * out_width,
                    in_data,
                    height_stride);
            }
            in_data += src_stride;
            out_data += dst_stride;
        }
    }
    nn_sem_post(&td->donesem);
}

static int convert_from_aix_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
    struct tensor const *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
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

//This should always be converted to the d32 variant
struct nn_node_ops nn_ops_for_Convert_to_aix_d32 = {
    .execute = convert_to_aix_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Convert_to_aix_d32_d32 = {
    .execute = convert_to_aix_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
    .flags = NN_NODE_FLAG_D32_INPUT};

//This should always be converted to the d32 variant
struct nn_node_ops nn_ops_for_Convert_from_aix = {
    .execute = convert_from_aix_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(3),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Convert_from_aix_d32 = {
    .execute = convert_from_aix_d32_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(3),
    .n_outputs = NN_IOCOUNT(3),
    .flags = NN_NODE_FLAG_D32_OUTPUT};
