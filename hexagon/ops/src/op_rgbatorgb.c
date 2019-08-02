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
 * This contains implementations for RGBA to RGB color space conversion
 * and also ARGB32 to RGB color space conversion
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include "quantize.h"
#include "nn_string_map.h"

// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_RGBATORGB_INPUT_DATA_IDX 0
#define OP_RGBATORGB_ISBGR_IDX 1
#define OP_RGBATORGB_INPUT_MIN_IDX 2
#define OP_RGBATORGB_INPUT_MAX_IDX 3
#define OP_RGBATORGB_NUM_OPS 4
#define OP_ARGB32TORGB_NUM_OPS 4

#define RGB_CHANNEL_DEPTH 3
#define RGBA_CHANNEL_DEPTH 4


static int rgba_to_rgb_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[OP_RGBATORGB_INPUT_DATA_IDX];
    const struct tensor *is_bgr_tensor = self->inputs[OP_RGBATORGB_ISBGR_IDX];
    const struct tensor *in_min_tensor = self->inputs[OP_RGBATORGB_INPUT_MIN_IDX];
    const struct tensor *in_max_tensor = self->inputs[OP_RGBATORGB_INPUT_MAX_IDX];

    // This cannot be in _check func because input shape is not available then
    if(in_tensor->shape.depth != RGBA_CHANNEL_DEPTH) {
        return errlog(nn, "OP_RgbaToRgb_8 input channel depth must be %d, %d provided",
                      RGBA_CHANNEL_DEPTH, in_tensor->shape.depth);
    }

    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    float in_max_float = tensor_get_float(in_max_tensor, 0);
    float in_min_float = tensor_get_float(in_min_tensor, 0);
    float out_min = in_min_float;
    float out_max = in_max_float;

    if( tensor_out_prepare_normal( out_tensor,
            in_tensor->shape.batches, in_tensor->shape.height,
            in_tensor->shape.width, RGB_CHANNEL_DEPTH,  NN_TYPE_QUINT8 ) != 0 ){
    	return errlog(nn,"output too small");
    }
    tensor_set_single_float( out_min_tensor, out_min);
    tensor_set_single_float( out_max_tensor, out_max);

    uint8_t *input = (uint8_t *) in_tensor->data;
    uint8_t *output = (uint8_t *) out_tensor->data;


    const size_t iDepth = RGBA_CHANNEL_DEPTH;// in_tensor->shape.depth;
    const size_t oBatch = out_tensor->shape.batches;
    const size_t oWidth = out_tensor->shape.width;
    const size_t oHeight = out_tensor->shape.height;
    const size_t oDepth = RGB_CHANNEL_DEPTH; // out_tensor->shape.depth;
    uint32_t isBGR = tensor_get_int32(is_bgr_tensor, 0);

    // Are we really Argb32ToRgb node?
    int ntype = self->node_type;
    int isA_first = ( ntype == OP_Argb32ToRgb_8 || ntype == OP_Argb32ToRgb_8_ref );

    // four cases can all be done from a 32-bit read with optional swizzle followed by optional shift:
    //
    // (a) RGBA -> RGB:   A:B:G:R ->             *:B:G:R   (no swizzle, >>0)
    // (b) RGBA -> BGR:   A:B:G:R ->  R:G:B:A -> *:R:G:B   (swizzle, >>8)
    // (c) ARGB -> RGB:   B:G:R:A ->             *:B:G:R   (no swizzle, >>8)
    // (d) ARGB -> BGR:   B:G:R:A ->  A:R:G:B -> *:R:G:B   (swizzle, >> 0)

    int need_swiz = isBGR;
    int rsh = isBGR ? 8:0;		// (b):(a)
    if( isA_first){
    	rsh = isBGR? 0:8;		// (d):(c)
    }

    for( size_t b = 0; b < oBatch; ++b )
    {
        size_t imageiPosition = b * oHeight;
        size_t imageoPosition = b * oHeight;
        for( size_t y = 0; y < oHeight; ++y )
        {
            size_t rowiPosition = (imageiPosition + y) * oWidth;
            size_t rowoPosition = (imageoPosition + y) * oWidth;
            for( size_t x = 0; x < oWidth; ++x )
            {
                size_t iPosition = rowiPosition + x;
                size_t oPosition = rowoPosition + x;

                const uint8_t* in = input + iDepth*iPosition;
                uint8_t* out = output + oDepth*oPosition;

                uint32_t val = *(uint32_t const *)in;			// read all 4 bytes
                if( need_swiz) val  = byteswap_u32(val);		// conditional byte swap
                val >>= rsh;										/// >> 0 or 8
                out[0] = (uint8_t) val;
                out[1] = (uint8_t) (val>>8);
                out[2] = (uint8_t) (val>>16);
#if 0
                if( !isBGR )
                {
                    // To RGB
                    for( size_t z = 0; z < oDepth; ++z )
                    {
                        out[z] = in[z];
                    }
                }
                else
                {
                    // To BGR
                    for( size_t z = 0; z < oDepth; ++z )
                    {
                        out[z] = in[iDepth-2-z];
                    }
                }
#endif
            }
        }
    }
    return 0;
}



// must have OP_RGBATORGB_NUM_OPS inputs and 3 outputs

struct nn_node_ops nn_ops_for_RgbaToRgb_8 = {
        .execute = rgba_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RGBATORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_RgbaToRgb_8_ref = {
        .execute = rgba_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RGBATORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};


struct nn_node_ops nn_ops_for_Argb32ToRgb_8 = {
        .execute = rgba_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RGBATORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Argb32ToRgb_8_ref = {
        .execute = rgba_to_rgb_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(OP_RGBATORGB_NUM_OPS),
        .n_outputs = NN_IOCOUNT(3),
};

