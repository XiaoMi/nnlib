
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

#define D32_INPUT 1
#define FLAT_INPUT 2
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

static void convert_flat_to_aix_d32(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    int start_batch = td->start_batch;
    int end_batch = td->end_batch;
    int height = td->in_height;
    int width = td->in_width;
    int depth = td->in_depth;
    struct shape input_shape = (struct shape){
        .batches = end_batch - start_batch,
        .height = height,
        .width = width,
        .depth = depth};
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    uint8_t *in_data = in_tensor->data;
    int in_idx = 0;
    int d_valid = 32;

    for (int b = start_batch; b < end_batch; b++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int d = 0;
                while (d < depth)
                {
                    int d_left = depth - d;
                    d_valid = 32;
                    if (d_left < d_valid)
                    {
                        d_valid = d_left;
                    }
                    uint8_t *out_ptr = tensor_location_d32_aix(out_tensor, b, h, w, d, &input_shape);
                    vmemcpy_asm(out_ptr, &in_data[in_idx], d_valid);
                    d += d_valid;
                    in_idx += d_valid;
                }
            }
        }
    }
    nn_sem_post(&td->donesem);
}

static void convert_to_aix_d32_work(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    int start_batch = td->start_batch;
    int end_batch = td->end_batch;
    int out_height = td->out_height;
    int out_width = td->out_width;
    int out_depth = td->out_depth;
    int depth_padding = 32;
    int in_nd32 = out_depth / depth_padding;
    const struct tensor *in_tensor = td->in_tensor;
    const struct tensor *out_tensor = td->out_tensor;
    int height_stride = tensor_d32_stride_d32(in_tensor);
    int out_height_stride = depth_padding * out_width;
    uint8_t *out_data = out_tensor->data;
    for (int b = start_batch; b < end_batch; b++)
    {
        for (int h = 0; h < out_height; h++)
        {
            uint8_t * dst = out_data + h * out_height_stride * in_nd32;
            vmemcpy_2d_general_asm(
                depth_padding * out_width,
                in_nd32,
                dst,
                out_height_stride,
                tensor_location_d32(in_tensor, b, h, 0, 0),
                height_stride);
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
    if (need_convert == D32_INPUT) //Tensor is d32, run d32->aix d32 convert
    {
        nn_os_work_for_vector(nn, convert_to_aix_d32_work, &td);
    }
    else if (need_convert == FLAT_INPUT) //Tensor is flat, run flat->d32 convert
    {
        nn_os_work_for_vector(nn, convert_flat_to_aix_d32, &td);
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

static int convert_from_aix_d32_check(struct nn_node *self, struct nn_graph *nn)
{
    int k;
    logmsg(nn, 2, "Convert from aix d32 check %p", self);
    k = node_check_inputs_range(self, nn, "convert aix d32 op", 3, 3);
    if (k != 0)
        return k;
    k = node_check_outputs_range(self, nn, "convert aix d32 op", 3, 3);
    if (k != 0)
        return k;
    logmsg(nn, 2, "convert to aix d32 op %p check OK", self);
    return 0;
}

static int convert_to_aix_d32_check(struct nn_node *self, struct nn_graph *nn)
{
    int k;
    logmsg(nn, 2, "Convert to aix d32 check %p", self);
    k = node_check_inputs_range(self, nn, "convert aix d32 op", 4, 4);
    if (k != 0)
        return k;
    k = node_check_outputs_range(self, nn, "convert aix d32 op", 3, 3);
    if (k != 0)
        return k;
    logmsg(nn, 2, "convert to aix d32 op %p check OK", self);
    return 0;
}

struct nn_node_ops nn_ops_for_Convert_to_aix_d32 = {
    .execute = convert_to_aix_d32_execute,
    .check = convert_to_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

//This is unsued
struct nn_node_ops nn_ops_for_Convert_from_aix = {
    .execute = convert_from_aix_d32_execute,
    .check = convert_from_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Convert_from_aix_d32 = {
    .execute = convert_from_aix_d32_execute,
    .check = convert_from_aix_d32_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .flags = NN_NODE_FLAG_D32_OUTPUT
};
    
