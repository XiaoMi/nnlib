/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
 * This contains implementations for Resize, using unit square algorithm
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

static size_t min_sizet(size_t a, size_t b) { return ((a < b) ? a : b); }

// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_RESIZE_INPUT_DATA_IDX 0
#define OP_RESIZE_SCALEW_IDX 1
#define OP_RESIZE_SCALEH_IDX 2
#define OP_RESIZE_PADVALUE_IDX 3
#define OP_RESIZE_MAINTAINASPECT_IDX 4
#define OP_RESIZE_OUTPUT_SHAPE_IDX 5
#define OP_RESIZE_INPUT_MIN_IDX 6
#define OP_RESIZE_INPUT_MAX_IDX 7
#define OP_RESIZE_NUM_OPS 8

static int resize_unitsquare_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[OP_RESIZE_INPUT_DATA_IDX];
    const struct tensor *scalew_tensor = self->inputs[OP_RESIZE_SCALEW_IDX];
    const struct tensor *scaleh_tensor = self->inputs[OP_RESIZE_SCALEH_IDX];
    const struct tensor *padvalue_tensor = self->inputs[OP_RESIZE_PADVALUE_IDX];
    const struct tensor *maintainaspect_tensor = self->inputs[OP_RESIZE_MAINTAINASPECT_IDX];
    const struct tensor *output_shape_tensor = self->inputs[OP_RESIZE_OUTPUT_SHAPE_IDX];
    const struct tensor *in_min_tensor = self->inputs[OP_RESIZE_INPUT_MIN_IDX];
    const struct tensor *in_max_tensor = self->inputs[OP_RESIZE_INPUT_MAX_IDX];


    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    uint8_t *input = (uint8_t *) in_tensor->data;
    uint8_t *out = (uint8_t *) out_tensor->data;
    float in_max_float = tensor_get_float(in_max_tensor, 0);
    float in_min_float = tensor_get_float(in_min_tensor, 0);
    float out_min = in_min_float;
    float out_max = in_max_float;

    tensor_set_shape(out_tensor,
                     output_shape_tensor->shape.batches, output_shape_tensor->shape.height,
                     output_shape_tensor->shape.width, output_shape_tensor->shape.depth);
    out_tensor->data_size = out_tensor->shape.batches *
                            out_tensor->shape.height *
                            out_tensor->shape.width *
                            out_tensor->shape.depth;

    tensor_set_shape(out_min_tensor, 1, 1, 1, 1);
    tensor_set_float(out_min_tensor, 0, out_min);
    tensor_set_shape(out_max_tensor, 1, 1, 1, 1);
    tensor_set_float(out_max_tensor, 0, out_max);
    out_min_tensor->data_size = sizeof(float);
    out_max_tensor->data_size = sizeof(float);

    int32_t in_sub_amt = quantize_uint8(0.0f, in_min_float, in_max_float);

    float scaleW = tensor_get_float(scalew_tensor, 0);
    float scaleH = tensor_get_float(scaleh_tensor, 0);
    float padvalue = tensor_get_float(padvalue_tensor, 0);
    uint32_t maintainAspectRatio = tensor_get_int32(maintainaspect_tensor, 0);
    // using unit square formula: f(x,y) = f00 + f10*x + f01*y + f11*x*y
    // where f00 = f(0,0)
    // f10 = f(1,0) - f(0,0)
    // f01 = f(0,1) - f(0,0)
    // f11 = f(1,1) + f(0,0) - (f(1,0) + f(0,1))
    const size_t batch = in_tensor->shape.batches;
    const size_t depth = in_tensor->shape.depth;
    const size_t inputWidth = in_tensor->shape.width;
    const size_t inputHeight = in_tensor->shape.height;
    const size_t outputWidth = output_shape_tensor->shape.width;
    const size_t outputHeight = output_shape_tensor->shape.height;
    const size_t targetWidth = inputWidth * scaleW;
    const size_t targetHeight = inputHeight * scaleH;
    const size_t offsetY = (outputHeight - targetHeight) / 2;
    const size_t offsetX = (outputWidth - targetWidth) / 2;
    uint8_t* imageInput;
    uint8_t* imageOut;
    const size_t inImageSize = inputHeight * inputWidth * depth;
    const size_t outImageSize = outputHeight * outputWidth * depth;

    for (size_t b = 0; b < batch; ++b) {
        imageInput = input + b * inImageSize;
        imageOut = out + b * outImageSize;

        for (size_t y = 0; y < outputHeight; ++y) {
            int padH = maintainAspectRatio &&
                       (((float) y < (outputHeight - targetHeight) *0.5f) ||
                        ((float) y > (outputHeight + targetHeight) *0.5f));

            const size_t y0 = (y - offsetY) * inputHeight / outputHeight;
            const size_t y1 = min_sizet(y0 + 1, inputHeight - 1);
            int32_t py_d = outputHeight;
            int32_t py_n = ((y - offsetY) * inputHeight) % py_d;

            for (size_t x = 0; x < outputWidth; ++x) {
                int padW = maintainAspectRatio &&
                           (((float) x < (outputWidth - targetWidth) *0.5f) ||
                            ((float) x > (outputWidth + targetWidth) *0.5f));

                if (padH || padW) {
                    uint8_t *output = imageOut + depth * (x + targetWidth * y);
                    for (size_t z = 0; z < depth; ++z) {
                        output[z] = padvalue;
                    }
                } else {
                    const size_t x0 = (x - offsetX) * inputWidth / outputWidth;
                    const size_t x1 = min_sizet(x0 + 1, inputWidth - 1);
                    int32_t px_d = outputWidth;
                    int32_t px_n = ((x - offsetX) * inputWidth) % px_d;

                    const uint8_t *f00 = imageInput + depth * (x0 + inputWidth * y0);
                    const uint8_t *f01 = imageInput + depth * (x0 + inputWidth * y1);
                    const uint8_t *f10 = imageInput + depth * (x1 + inputWidth * y0);
                    const uint8_t *f11 = imageInput + depth * (x1 + inputWidth * y1);
                    uint8_t *output = imageOut + depth * (x + targetWidth * y);

                    for (size_t z = 0; z < depth; ++z) {
                        const int32_t a00 = (int32_t) f00[z] - in_sub_amt;
                        const int32_t a01 = (int32_t) f01[z] - in_sub_amt - a00;
                        const int32_t a10 = (int32_t) f10[z] - in_sub_amt - a00;
                        const int32_t a11 = (int32_t) f11[z] - in_sub_amt - a10 - a01 - a00;
                        int32_t output_val = px_d * py_d * a00 +
                                             py_d * a10 * px_n +
                                             px_d * a01 * py_n +
                                             a11 * px_n * py_n;
                        output_val /= px_d * py_d;
                        output_val += in_sub_amt;
                        output_val = (output_val < 0) ? 0 : ((output_val > 255) ? 255 : output_val);
                        output[z] = output_val;
                    }
                }
            }
        }
    }
    return 0;
}



struct nn_node_ops nn_ops_for_ResizeUnitSquare_8 = {
        .execute = resize_unitsquare_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RESIZE_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_ResizeUnitSquare_8_ref = {
        .execute = resize_unitsquare_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RESIZE_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};


