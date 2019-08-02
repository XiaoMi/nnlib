/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (mulject to the limitations in the
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
 * This contains implementations for NV21 to RGB color space conversion
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>


// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_NV21TORGB_INPUT_DATA_IDX 0
#define OP_NV21TORGB_ISBGR_IDX 1
#define OP_NV21TORGB_NUM_OPS 2

#if defined(__hexagon__)
static int32_t min(int32_t a, int32_t b) { return ((a < b) ? a : b); }
static int32_t max(int32_t a, int32_t b) { return ((a > b) ? a : b); }
#endif

// This value is 2^16-1 and is used to clamp the RGB values before the
// range is normalized to 8-bit
static const int32_t MAX_CHANNEL_VALUE = 65535;

#define RGB_CHANNEL_DEPTH 3

static void YuvToRgb_op8
        (
                uint8_t y,
                uint8_t u,
                uint8_t v,
                uint8_t *rgb
        )
{
    int32_t C = y;
    int32_t D = u - 128;
    int32_t E = v - 128;

    int32_t r = (256*C + 359*E + 128);
    int32_t g = (256*C - 88*D - 183*E + 128);
    int32_t b = (256*C + 454*D + 128);

    r = min( max(r, 0), MAX_CHANNEL_VALUE);
    g = min( max(g, 0), MAX_CHANNEL_VALUE);
    b = min( max(b, 0), MAX_CHANNEL_VALUE);

    r >>= 8;
    g >>= 8;
    b >>= 8;

    *rgb++ = r;
    *rgb++ = g;
    *rgb = b;
}

static void YuvToBgr_op8
        (
                uint8_t y,
                uint8_t u,
                uint8_t v,
                uint8_t *bgr
        )
{
    int32_t C = y;
    int32_t D = u - 128;
    int32_t E = v - 128;

    int32_t r = (256*C + 359*E + 128);
    int32_t g = (256*C - 88*D - 183*E + 128);
    int32_t b = (256*C + 454*D + 128);

    r = min( max(r, 0), MAX_CHANNEL_VALUE);
    g = min( max(g, 0), MAX_CHANNEL_VALUE);
    b = min( max(b, 0), MAX_CHANNEL_VALUE);

    r >>= 8;
    g >>= 8;
    b >>= 8;

    *bgr++ = b;
    *bgr++ = g;
    *bgr = r;
}

static int nv21_to_rgb_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[OP_NV21TORGB_INPUT_DATA_IDX];
    const struct tensor *is_bgr_tensor = self->inputs[OP_NV21TORGB_ISBGR_IDX];

    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    uint8_t *input = (uint8_t *) in_tensor->data;
    uint8_t *output = (uint8_t *) out_tensor->data;
    float out_min = 0.0;
    float out_max = 255.0;

    tensor_set_shape(out_tensor,
                     in_tensor->shape.batches, in_tensor->shape.height,
                     in_tensor->shape.width, RGB_CHANNEL_DEPTH);
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
 
    const size_t oBatch = out_tensor->shape.batches;
    const size_t oWidth = out_tensor->shape.width;
    const size_t oHeight = out_tensor->shape.height;
    const size_t oDepth = out_tensor->shape.depth;
    uint32_t isBGR = tensor_get_int32(is_bgr_tensor, 0);

    size_t yDataSize = oWidth*oHeight;
    size_t yuvDataSize = (yDataSize*3)/2;
    size_t rgbDataSize = yDataSize * RGB_CHANNEL_DEPTH;

    size_t uvoffset = yDataSize;
    uint8_t u, v, y1, y2, y3, y4;

    uint8_t * currentInput;
    uint8_t * currentOutput;

    // Unroll the loop by 2 step yDataSize along the horizontal and
    // vertical dimension.
    for (int b = 0; b < oBatch; b++)
    {
        currentInput = input + b * yuvDataSize;
        currentOutput = output + b * rgbDataSize;
        for(size_t i=0, k=0; i < yDataSize; i+=2, k+=2)
        {
            y1 = *(currentInput+ i);
            y2 = *(currentInput+ i + 1);
            y3 = *(currentInput+ oWidth + i);
            y4 = *(currentInput+ oWidth + i + 1);
            v = *(currentInput+ uvoffset + k);
            u = *(currentInput+ uvoffset + k + 1);

            if( !isBGR )
            {
                YuvToRgb_op8(y1, u, v, currentOutput + i*oDepth);
                YuvToRgb_op8(y2, u, v, currentOutput + (i+1)*oDepth);
                YuvToRgb_op8(y3, u, v, currentOutput + (oWidth+i)*oDepth);
                YuvToRgb_op8(y4, u, v, currentOutput + (oWidth+i+1)*oDepth);
            }
            else
            {
                YuvToBgr_op8(y1, u, v, currentOutput + i*oDepth);
                YuvToBgr_op8(y2, u, v, currentOutput + (i+1)*oDepth);
                YuvToBgr_op8(y3, u, v, currentOutput + (oWidth+i)*oDepth);
                YuvToBgr_op8(y4, u, v, currentOutput + (oWidth+i+1)*oDepth);
            }

            if((i+2)%oWidth==0)
                i+=oWidth;
        }
    }

    return 0;
}


struct nn_node_ops nn_ops_for_Nv21ToRgb_8 = {
        .execute = nv21_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_NV21TORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Nv21ToRgb_8_ref = {
        .execute = nv21_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_NV21TORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};
